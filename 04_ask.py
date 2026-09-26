"""
Step 4: Ask a question and get a grounded answer with video + timestamp citations.

Usage:
    python 04_ask.py "how do I detect a cycle in a graph"

How it works:
    1. Embed the question
    2. Retrieve the top-K most similar chunks from Qdrant
    3. Feed those chunks to Groq's LLM as context, instructing it to answer
       ONLY from that context and cite sources
    4. Print the answer + the exact video/timestamp sources used
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

SYSTEM_PROMPT = """You are a DSA (Data Structures & Algorithms) study assistant.
You answer questions using ONLY the provided lecture transcript excerpts below.

Rules:
- If the excerpts contain the answer, explain it clearly and concisely, as if teaching a student.
- If the excerpts do NOT contain enough information to answer the question, say plainly:
  "This isn't covered in the lectures I have access to." Do not make up an answer.
- Do not mention "excerpts" or "context" in your answer - just answer naturally, like a tutor.
"""


def retrieve(client: QdrantClient, model: SentenceTransformer, query: str, top_k: int = TOP_K):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return results


def build_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        p = r.payload
        parts.append(f"[Excerpt {i} from \"{p['title']}\"]\n{p['text']}")
    return "\n\n".join(parts)


def generate_answer(groq_client: Groq, question: str, context: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Lecture excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage: python 04_ask.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    load_dotenv()
    qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nQuestion: {question}\n")
    print("Searching lectures...")
    results = retrieve(qdrant_client, model, question)

    if not results:
        print("No relevant content found.")
        return

    context = build_context(results)

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
    for r in results:
        p = r.payload
        if p["video_id"] not in seen_videos:
            seen_videos.add(p["video_id"])
            mins, secs = divmod(int(p["start_ts"]), 60)
            print(f"- {p['title']} ({mins}:{secs:02d})")
            print(f"  {p['youtube_link']}")


if __name__ == "__main__":
    main()
