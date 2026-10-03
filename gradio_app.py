"""
Gradio frontend for HuggingFace Spaces deployment (free tier).

Docker Spaces now require a paid HF plan, so this wraps the same hybrid
retrieval + reranking + Groq generation pipeline as app.py (the FastAPI
version used locally and shown in the GitHub repo) in a Gradio UI instead,
which free accounts can still deploy.

Secrets (QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY) are set via the Space's
Settings -> Variables and secrets, and are already present as environment
variables at runtime - no .env file needed here.
"""

import os
import re

import spaces  # must be imported before any CUDA-related package (torch, sentence-transformers)
import gradio as gr
from groq import Groq
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GROQ_MODEL = "openai/gpt-oss-120b"

VECTOR_TOP_K = 15
BM25_TOP_K = 15
RERANK_CANDIDATE_K = 15
FINAL_TOP_K = 5
RRF_K = 60

SYSTEM_PROMPT = """You are a DSA (Data Structures & Algorithms) study assistant.
You answer questions using ONLY the provided lecture transcript excerpts below.

Rules:
- If the excerpts contain the answer, explain it clearly and concisely, as if teaching a student.
- If the excerpts do NOT contain enough information to answer the question, say plainly:
  "This isn't covered in the lectures I have access to." Do not make up an answer.
- Do not mention "excerpts" or "context" in your answer - just answer naturally, like a tutor.
"""

print("Connecting to Qdrant...")
qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

print("Loading chunks from Qdrant and building BM25 index...")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks_from_qdrant() -> list[dict]:
    chunks = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=250, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            chunks.append(p.payload)
        if offset is None:
            break
    return chunks


chunks = load_chunks_from_qdrant()
chunks_by_id = {c["chunk_id"]: c for c in chunks}
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker model...")
reranker = CrossEncoder(RERANKER_MODEL)

print("Connecting to Groq...")
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Ready.")


def bm25_search(query: str, top_k: int) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked]


def vector_search(query: str, top_k: int) -> list[str]:
    query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points
    return [r.payload["chunk_id"] for r in results]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda item_id: scores[item_id], reverse=True)


def rerank(query: str, candidate_ids: list[str], top_k: int) -> list[str]:
    pairs = [(query, chunks_by_id[cid]["text"]) for cid in candidate_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]


def generate_answer(question: str, context: str) -> str:
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


@spaces.GPU
def answer_question(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    if len(question) > 500:
        return "Question too long (max 500 characters).", ""

    bm25_ids = bm25_search(question, BM25_TOP_K)
    vector_ids = vector_search(question, VECTOR_TOP_K)
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:RERANK_CANDIDATE_K]
    final_ids = rerank(question, fused_ids, FINAL_TOP_K)

    context = "\n\n".join(
        f"[Excerpt {i} from \"{chunks_by_id[cid]['title']}\"]\n{chunks_by_id[cid]['text']}"
        for i, cid in enumerate(final_ids, 1)
    )
    answer = generate_answer(question, context)

    seen_videos = set()
    source_lines = []
    for cid in final_ids:
        c = chunks_by_id[cid]
        if c["video_id"] not in seen_videos:
            seen_videos.add(c["video_id"])
            source_lines.append(f"- [{c['title']}]({c['youtube_link']})")
    sources_md = "**Sources:**\n" + "\n".join(source_lines)

    return answer, sources_md


with gr.Blocks(title="DSA Lecture RAG") as demo:
    gr.Markdown("# DSA Lecture Q&A\nAsk a Data Structures & Algorithms question - answers are grounded in a transcribed lecture playlist, with direct timestamped YouTube links.")
    question_box = gr.Textbox(label="Your question", placeholder="e.g. how do I detect a cycle in a graph")
    ask_btn = gr.Button("Ask")
    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources")

    ask_btn.click(fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
