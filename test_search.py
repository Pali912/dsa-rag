"""
Quick sanity check: search the Qdrant collection with a test query and print
the top results, so we can visually confirm retrieval is working before
building the full retrieval + generation pipeline in Weekend 2.

Usage:
    python test_search.py "your question here"
"""

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_search.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]

    load_dotenv()
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    model = SentenceTransformer(EMBEDDING_MODEL)

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
    ).points

    print(f"\nTop {TOP_K} results for: {query!r}\n")
    for i, r in enumerate(results, 1):
        p = r.payload
        print(f"{i}. [{r.score:.3f}] {p['title']}")
        print(f"   {p['youtube_link']}")
        print(f"   {p['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
