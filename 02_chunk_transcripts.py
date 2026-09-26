"""
Step 2: Chunk timestamped transcripts into ~75-second windows, preserving
video_id / title / url / start_ts / end_ts so each chunk can be traced back
to an exact YouTube timestamp later.

Usage:
    python 02_chunk_transcripts.py --window 75

Input:
    data/transcripts/<video_id>.json   (from 01_download_transcribe.py)

Output:
    data/chunks.jsonl   (one JSON object per line, ready for embedding)
"""

import argparse
import json
from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts")
OUTPUT_PATH = Path("data/chunks.jsonl")


def chunk_video(record: dict, window_seconds: float) -> list[dict]:
    """Group a video's segments into fixed-size time windows."""
    segments = record["segments"]
    if not segments:
        return []

    chunks = []
    window_start = segments[0]["start"]
    window_texts = []
    window_end = window_start

    for seg in segments:
        if seg["start"] - window_start >= window_seconds and window_texts:
            chunks.append(_make_chunk(record, window_start, window_end, window_texts))
            window_start = seg["start"]
            window_texts = []

        window_texts.append(seg["text"])
        window_end = seg["end"]

    if window_texts:
        chunks.append(_make_chunk(record, window_start, window_end, window_texts))

    return chunks


def _make_chunk(record: dict, start: float, end: float, texts: list[str]) -> dict:
    text = " ".join(t for t in texts if t).strip()
    start_int = int(start)
    return {
        "chunk_id": f"{record['video_id']}_{start_int}",
        "video_id": record["video_id"],
        "title": record["title"],
        "url": record["url"],
        "start_ts": round(start, 2),
        "end_ts": round(end, 2),
        "youtube_link": f"{record['url']}&t={start_int}s",
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=75.0, help="Chunk window size in seconds (default: 75)")
    args = parser.parse_args()

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.json"))
    if not transcript_files:
        print(f"No transcripts found in {TRANSCRIPT_DIR}. Run 01_download_transcribe.py first.")
        return

    all_chunks = []
    for path in transcript_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_video(record, args.window)
        all_chunks.extend(chunks)
        print(f"{record['video_id']}: {len(chunks)} chunks")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks from {len(transcript_files)} videos to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
