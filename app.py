"""
FastAPI backend for the DSA Lecture RAG Q&A system.

Loads all models once at startup (not per-request) for performance, then
serves a single /ask endpoint wrapping the hybrid retrieval + reranking +
Groq generation pipeline from 06_reranked_ask.py.

Run locally:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI.
"""

import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from groq import Groq

COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GROQ_MODEL = "openai/gpt-oss-120b"

VECTOR_TOP_K = 15
BM25_TOP_K = 15
RERANK_CANDIDATE_K = 15
FINAL_TOP_K = 5
RRF_K = 60

# Basic per-IP rate limit: protects free-tier Groq/Qdrant quota from public abuse
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60

SYSTEM_PROMPT = """You are a DSA (Data Structures & Algorithms) study assistant.
You answer questions using ONLY the provided lecture transcript excerpts below.

Rules:
- If the excerpts contain the answer, explain it clearly and concisely, as if teaching a student.
- If the excerpts do NOT contain enough information to answer the question, say plainly:
  "This isn't covered in the lectures I have access to." Do not make up an answer.
- Do not mention "excerpts" or "context" in your answer - just answer naturally, like a tutor.
"""

# Models and clients are loaded once in the lifespan handler, then stashed here
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    print("Connecting to Qdrant...")
    state["qdrant"] = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

    print("Loading chunks from Qdrant and building BM25 index...")
    state["chunks"] = load_chunks_from_qdrant(state["qdrant"])
    state["chunks_by_id"] = {c["chunk_id"]: c for c in state["chunks"]}
    state["bm25"] = build_bm25_index(state["chunks"])

    print("Loading embedding model...")
    state["embed_model"] = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading reranker model...")
    state["reranker"] = CrossEncoder(RERANKER_MODEL)

    print("Connecting to Groq...")
    state["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("Ready.")
    yield
    state.clear()


app = FastAPI(title="DSA Lecture RAG", lifespan=lifespan)

# Allow a simple frontend (e.g. hosted on the same Space, or a separate static page) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limit tracking: {ip: [timestamps]}. Resets on server restart - fine for a demo.
request_log: dict = defaultdict(list)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks_from_qdrant(qdrant_client: QdrantClient) -> list[dict]:
    """Pulls all chunk payloads from Qdrant via scroll, so no local transcript
    file needs to be deployed (avoids publicly exposing raw transcript text
    via the hosting platform's file browser)."""
    chunks = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            chunks.append(p.payload)
        if offset is None:
            break
    return chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_search(bm25, chunks, query, top_k) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked]


def vector_search(qdrant_client, model, query, top_k) -> list[str]:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points
    return [r.payload["chunk_id"] for r in results]


def reciprocal_rank_fusion(ranked_lists, k=RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)


def rerank(reranker, chunks_by_id, query, candidate_ids, top_k) -> list[str]:
    pairs = [(query, chunks_by_id[cid]["text"]) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]


def generate_answer(groq_client, question, context) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Lecture excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


def check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    request_log[client_ip] = [t for t in request_log[client_ip] if t > window_start]
    if len(request_log[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    request_log[client_ip].append(now)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    youtube_link: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    check_rate_limit(client_ip)

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 characters).")

    bm25_ids = bm25_search(state["bm25"], state["chunks"], question, BM25_TOP_K)
    vector_ids = vector_search(state["qdrant"], state["embed_model"], question, VECTOR_TOP_K)
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:RERANK_CANDIDATE_K]
    final_ids = rerank(state["reranker"], state["chunks_by_id"], question, fused_ids, FINAL_TOP_K)

    chunks_by_id = state["chunks_by_id"]
    context = "\n\n".join(
        f"[Excerpt {i} from \"{chunks_by_id[cid]['title']}\"]\n{chunks_by_id[cid]['text']}"
        for i, cid in enumerate(final_ids, 1)
    )
    answer = generate_answer(state["groq"], question, context)

    seen_videos = set()
    sources = []
    for cid in final_ids:
        c = chunks_by_id[cid]
        if c["video_id"] not in seen_videos:
            seen_videos.add(c["video_id"])
            sources.append(Source(title=c["title"], youtube_link=c["youtube_link"]))

    return AskResponse(answer=answer, sources=sources)
