"""
Step 6: Hybrid retrieval (BM25 + vector, RRF-fused) -> BGE reranker -> Groq generation.

Why reranking: RRF gives us a good, broad candidate set, but it's still just
combining two "cheap" retrieval signals. A cross-encoder reranker looks at the
actual (query, chunk) pair together (not just separate embeddings) and scores
relevance much more precisely - it's slower, so we only run it on a small
candidate set (e.g. 15) rather than the whole corpus.

Usage:
    python 06_reranked_ask.py "how do I detect a cycle in a graph"
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GROQ_MODEL = "openai/gpt-oss-120b"

VECTOR_TOP_K = 15
BM25_TOP_K = 15
RERANK_CANDIDATE_K = 15  # how many fused candidates get reranked
FINAL_TOP_K = 5  # chunks actually sent to the LLM after reranking
RRF_K = 60

SYSTEM_PROMPT = """You are a DSA (Data Structures & Algorithms) study assistant.
You answer questions using ONLY the provided lecture transcript excerpts below.

Rules:
- If the excerpts contain the answer, explain it clearly and concisely, as if teaching a student.
- If the excerpts do NOT contain enough information to answer the question, say plainly:
  "This isn't covered in the lectures I have access to." Do not make up an answer.
- Do not mention "excerpts" or "context" in your answer - just answer naturally, like a tutor.
"""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run 02_chunk_transcripts.py first.")
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked]


def vector_search(qdrant_client: QdrantClient, model: SentenceTransformer, query: str, top_k: int) -> list[str]:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points
    return [r.payload["chunk_id"] for r in results]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)


def rerank(reranker: CrossEncoder, chunks_by_id: dict, query: str, candidate_ids: list[str], top_k: int) -> list[str]:
    pairs = [(query, chunks_by_id[cid]["text"]) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]


def build_context(chunks_by_id: dict, ids: list[str]) -> str:
    parts = []
    for i, chunk_id in enumerate(ids, 1):
        c = chunks_by_id[chunk_id]
        parts.append(f"[Excerpt {i} from \"{c['title']}\"]\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(groq_client: Groq, question: str, context: str) -> str:
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


def main():
    if len(sys.argv) < 2:
        print('Usage: python 06_reranked_ask.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    load_dotenv()
    qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("Loading chunks and building BM25 index...")
    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = build_bm25_index(chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading reranker model (first run downloads it, ~280MB)...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print(f"\nQuestion: {question}\n")
    print("Running BM25 search...")
    bm25_ids = bm25_search(bm25, chunks, question, BM25_TOP_K)

    print("Running vector search...")
    vector_ids = vector_search(qdrant_client, model, question, VECTOR_TOP_K)

    print("Fusing results (RRF)...")
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:RERANK_CANDIDATE_K]

    print("Reranking candidates...")
    final_ids = rerank(reranker, chunks_by_id, question, fused_ids, FINAL_TOP_K)

    context = build_context(chunks_by_id, final_ids)

    print("Generating answer...\n")
    answer = generate_answer(groq_client, question, context)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)
    seen_videos = set()
    for chunk_id in final_ids:
        c = chunks_by_id[chunk_id]
        if c["video_id"] not in seen_videos:
            seen_videos.add(c["video_id"])
            mins, secs = divmod(int(c["start_ts"]), 60)
            print(f"- {c['title']} ({mins}:{secs:02d})")
            print(f"  {c['youtube_link']}")


if __name__ == "__main__":
    main()
