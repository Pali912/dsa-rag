"""
Step 3: Embed all chunks and upload them to Qdrant Cloud.

Usage:
    python 03_embed_and_upload.py

Input:
    data/chunks.jsonl        (from 02_chunk_transcripts.py)

Output:
    Populates a Qdrant collection named COLLECTION_NAME with one point per chunk:
    vector = 384-dim embedding of chunk text
    payload = video_id, title, url, start_ts, end_ts, youtube_link, text

Idempotent: point IDs are derived deterministically from chunk_id, so rerunning
this script updates existing points instead of duplicating them.
"""

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/chunks.jsonl")
COLLECTION_NAME = "dsa_lectures"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

# A fixed namespace UUID so chunk_id -> point_id is stable across runs
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


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


def main():
    load_dotenv()
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in your .env file")

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"{len(chunks)} chunks loaded.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, upserting into it.")

    print("Embedding and uploading in batches...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for chunk, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=chunk_id_to_point_id(chunk["chunk_id"]),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "video_id": chunk["video_id"],
                        "title": chunk["title"],
                        "url": chunk["url"],
                        "start_ts": chunk["start_ts"],
                        "end_ts": chunk["end_ts"],
                        "youtube_link": chunk["youtube_link"],
                        "text": chunk["text"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {count} points.")


if __name__ == "__main__":
    main()
