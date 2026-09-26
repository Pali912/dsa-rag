# DSA Lecture RAG

A Retrieval-Augmented Generation (RAG) system that answers DSA (Data Structures & Algorithms)
interview questions by searching across a full playlist of lecture transcripts and citing the
exact video + timestamp the answer came from.

Built to solve a real problem: revising 60+ videos of lecture content without rewatching
everything from scratch, and to demonstrate an end-to-end AI engineering pipeline for a
software engineering resume.

> **Status: Weekend 1 (data pipeline) complete.** Retrieval + LLM generation, evaluation,
> and public deployment are in progress — see [Roadmap](#roadmap).

---

## What it does

1. Takes a YouTube playlist of DSA lecture videos
2. Downloads audio, transcribes it with timestamps
3. Chunks the transcript into ~75-second windows, preserving video + timestamp metadata
4. Embeds each chunk and stores it in a vector database
5. *(In progress)* Given a question, retrieves the most relevant chunks and generates a
   grounded answer with a clickable, timestamped YouTube link back to the source video

**Example (once Weekend 2 ships):**
> **Q:** How do I detect a cycle in a graph?
> **A:** [grounded answer] — *Source: "Course Schedule (LeetCode 207)" at 5:18*
> → https://youtube.com/watch?v=Oa4Srx9mDqs&t=318s

---

## Tech stack (100% free tier)

| Layer | Tool |
|---|---|
| Audio download | `yt-dlp` |
| Transcription | `faster-whisper` (local, CPU) |
| Chunking | custom Python, timestamp-preserving |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | Qdrant Cloud (free tier) |
| Sparse retrieval | BM25 *(planned)* |
| Reranker | BGE reranker *(planned)* |
| LLM | Groq (Llama 3.3 / GPT-OSS-120B) *(planned)* |
| Backend | FastAPI *(planned)* |
| Hosting | HuggingFace Spaces *(planned)* |
| Eval | RAGAS *(planned)* |

---

## Current dataset

Source: [Blind 75 LeetCode Questions](https://www.youtube.com/playlist?list=PLFdAYMIVJQHPfBe9big-PSAepnjOLc7Pz)
by Nikhil Lohia — full 62-video playlist.

- 62 videos transcribed
- 821 timestamped chunks
- Embedded and stored in Qdrant Cloud, verified searchable

---

## Setup

```bash
pip install -r requirements.txt
```

You'll also need `ffmpeg` installed and on your PATH.

Copy `.env` and fill in your own keys (never commit this file):

```
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### Pipeline

```bash
# 1. Download audio + transcribe a playlist
python 01_download_transcribe.py --playlist "PLAYLIST_URL" --limit 20

# 2. Chunk transcripts into timestamped windows
python 02_chunk_transcripts.py --window 75

# 3. Embed chunks and upload to Qdrant
python 03_embed_and_upload.py

# Sanity-check retrieval
python test_search.py "your DSA question here"
```

---

## Roadmap

- [x] Audio download + transcription pipeline (resumable)
- [x] Timestamp-preserving chunking
- [x] Embedding + Qdrant Cloud upload
- [ ] Hybrid retrieval (BM25 + vector)
- [ ] BGE reranking
- [ ] Groq LLM answer generation with citations
- [ ] RAGAS evaluation (faithfulness, context precision, answer relevancy)
- [ ] FastAPI backend + simple frontend
- [ ] Public deployment on HuggingFace Spaces

---

## Notes

- Source videos are not the author's own content; raw audio and transcripts are kept local
  (`.gitignore`'d) and not redistributed. Only code, chunk structure, and eval results are public.
- Pipeline is playlist-agnostic — works with any YouTube playlist, not just this one.

---

## Author

Built by Nitin Dhadwal as a portfolio project targeting software engineering roles.
