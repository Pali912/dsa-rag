"""
Step 7: Evaluate the RAG pipeline with RAGAS.

Metrics used (no ground-truth answers required):
- Faithfulness: does the generated answer stick to what's actually in the
  retrieved chunks, or does it hallucinate beyond them?
- Answer Relevancy: does the answer actually address the question asked?

Usage:
    python 07_eval.py

Output:
    Prints per-question and average scores.
    Saves full results to eval_results.json
"""

import json
import os
import re
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.metrics import AnswerRelevancy
from ragas.run_config import RunConfig
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GROQ_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-120b"

VECTOR_TOP_K = 15
BM25_TOP_K = 15
RERANK_CANDIDATE_K = 15
FINAL_TOP_K = 5  # matches production scripts (04/05/06)
RRF_K = 60

SYSTEM_PROMPT = """You are a DSA (Data Structures & Algorithms) study assistant.
You answer questions using ONLY the provided lecture transcript excerpts below.

Rules:
- If the excerpts contain the answer, explain it clearly and concisely, as if teaching a student.
- If the excerpts do NOT contain enough information to answer the question, say plainly:
  "This isn't covered in the lectures I have access to." Do not make up an answer.
- Do not mention "excerpts" or "context" in your answer - just answer naturally, like a tutor.
"""

# Golden question set - covers a spread of topics present in the Blind 75 dataset.
GOLDEN_QUESTIONS = [
    "How do I detect a cycle in a graph?",
    "What is the sliding window technique used for?",
    "How does the two pointer approach work?",
    "What is dynamic programming and when should I use it?",
    "How do I find the minimum window substring?",
    "What is a greedy algorithm?",
    "How does prefix sum help with subarray problems?",
    "How do I merge k sorted lists?",
    "What is the approach for trapping rain water?",
    "How do I serialize and deserialize a binary tree?",
    "What is Kahn's algorithm used for?",
    "How do I find the first missing positive number?",
    "What is the approach to solve word break using dynamic programming?",
    "How do I detect a cycle in a linked list?",
    "What is the time complexity of BFS on a graph?",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
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


def generate_answer(groq_client, question, context, max_retries=8) -> str:
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Lecture excerpts:\n\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Exceeded max retries due to rate limiting.")


def run_pipeline_for_question(question, chunks, chunks_by_id, bm25, qdrant_client, model, reranker, groq_client):
    bm25_ids = bm25_search(bm25, chunks, question, BM25_TOP_K)
    vector_ids = vector_search(qdrant_client, model, question, VECTOR_TOP_K)
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:RERANK_CANDIDATE_K]
    final_ids = rerank(reranker, chunks_by_id, question, fused_ids, FINAL_TOP_K)
    contexts = [chunks_by_id[cid]["text"] for cid in final_ids]
    context_str = "\n\n".join(
        f"[Excerpt {i} from \"{chunks_by_id[cid]['title']}\"]\n{chunks_by_id[cid]['text']}"
        for i, cid in enumerate(final_ids, 1)
    )
    answer = generate_answer(groq_client, question, context_str)
    return answer, contexts


FAITHFULNESS_PROMPT = """You are grading a study assistant's answer for faithfulness.

Context (lecture excerpts):
{context}

Answer to grade:
{answer}

How well is every claim in the answer supported by the context?
1 = mostly unsupported or contradicts the context
2 = several claims unsupported
3 = about half supported
4 = mostly supported, minor unsupported details
5 = fully supported by the context

Reply with ONLY a single digit from 1 to 5."""


def custom_faithfulness(groq_client, answer, contexts, max_retries=6):
    """One-call, one-digit faithfulness score normalised to 0-1.
    Replaces RAGAS's multi-step faithfulness, which needs more judge output than free-tier limits allow."""
    prompt = FAITHFULNESS_PROMPT.format(context="\n\n".join(contexts), answer=answer)
    for attempt in range(max_retries):
        try:
            resp = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,  # headroom: gpt-oss models spend some tokens reasoning before the digit
            )
            text = resp.choices[0].message.content or ""
            match = re.search(r"[1-5]", text)
            if match:
                return (int(match.group()) - 1) / 4  # 1..5 -> 0..1
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
    return float("nan")


CHECKPOINT_PATH = Path("eval_checkpoint.json")


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data: dict):
    CHECKPOINT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(GOLDEN_QUESTIONS), help="Number of golden questions to run (for quick testing)")
    args = parser.parse_args()
    questions_to_run = GOLDEN_QUESTIONS[: args.limit]

    load_dotenv()
    qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("Loading chunks and building BM25 index...")
    chunks = load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = build_bm25_index(chunks)

    print("Loading embedding + reranker models...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)

    print(f"\nRunning pipeline for {len(questions_to_run)} golden questions...\n")
    checkpoint = load_checkpoint()
    questions, answers, contexts_list = [], [], []
    for i, question in enumerate(questions_to_run, 1):
        if question in checkpoint:
            print(f"[{i}/{len(questions_to_run)}] {question} [cached]")
            questions.append(question)
            answers.append(checkpoint[question]["answer"])
            contexts_list.append(checkpoint[question]["contexts"])
            continue

        print(f"[{i}/{len(questions_to_run)}] {question}")
        answer, contexts = run_pipeline_for_question(
            question, chunks, chunks_by_id, bm25, qdrant_client, model, reranker, groq_client
        )
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        checkpoint[question] = {"answer": answer, "contexts": contexts}
        save_checkpoint(checkpoint)
        time.sleep(70)  # stay comfortably under Groq's free-tier 8000 TPM limit

    print("\nScoring faithfulness (custom single-call scorer)...")
    faithfulness_scores = []
    for i, (answer, contexts) in enumerate(zip(answers, contexts_list), 1):
        print(f"  [{i}/{len(answers)}] scoring...")
        score = custom_faithfulness(groq_client, answer, contexts)
        faithfulness_scores.append(score)
        time.sleep(20)  # stay under free-tier limits between judge calls

    print("\nRunning RAGAS for answer_relevancy...")
    dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts_list})

    judge_llm = ChatGroq(model=JUDGE_MODEL, api_key=os.environ["GROQ_API_KEY"], max_tokens=2048, temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # strictness=1 avoids asking Groq for multiple (n>1) sampled completions in one call, which it rejects
    answer_relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1)

    result = evaluate(
        dataset,
        metrics=[answer_relevancy_metric],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=8),
    )

    df = result.to_pandas()
    df["faithfulness"] = faithfulness_scores
    print("\nColumns in result:", list(df.columns))

    question_col = "user_input" if "user_input" in df.columns else "question"
    relevancy_col = "answer_relevancy" if "answer_relevancy" in df.columns else next(
        (c for c in df.columns if "relevan" in c.lower()), None
    )

    print("\n" + "=" * 60)
    print("PER-QUESTION SCORES")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"- {str(row[question_col])[:60]}...")
        r_score = row[relevancy_col] if relevancy_col else float("nan")
        print(f"    faithfulness: {row['faithfulness']:.2f}  |  answer_relevancy: {r_score:.2f}")

    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)
    print(f"Faithfulness (custom):  {df['faithfulness'].mean():.3f}")
    if relevancy_col:
        print(f"Answer Relevancy (RAGAS): {df[relevancy_col].mean():.3f}")

    df.to_json("eval_results.json", orient="records", indent=2)
    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()
