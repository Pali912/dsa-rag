# DSA Lecture RAG

A Retrieval-Augmented Generation (RAG) system that answers DSA (Data Structures & Algorithms)
interview questions by searching across a full playlist of lecture transcripts and citing the
exact video + timestamp the answer came from.

Built to solve a real problem: revising 60+ videos of lecture content without rewatching
everything from scratch, and to demonstrate an end-to-end AI engineering pipeline for a
software engineering resume.

> **Status: Complete.** Data pipeline, hybrid retrieval, reranking, generation,
> evaluation, and public deployment are all done.
>
> **Live demo:** https://huggingface.co/spaces/Pali912/dsa-rag

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
| Sparse retrieval | BM25 (via Reciprocal Rank Fusion with vector search) |
| Reranker | BGE cross-encoder (`bge-reranker-base`) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Backend | FastAPI *(planned)* |
| Hosting | HuggingFace Spaces *(planned)* |
| Eval | RAGAS (answer relevancy) + custom faithfulness scorer |

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

## Evaluation

Evaluated against 15 hand-written golden questions spanning the dataset's topics
(graphs, sliding window, DP, greedy, prefix sum, etc.).

| Metric | Score |
|---|---|
| Faithfulness (custom 1-5 LLM-judge scorer, normalized) | **0.783** |
| Answer Relevancy (RAGAS) | **0.739** |

**Methodology note:** RAGAS's built-in `faithfulness` metric requires the judge
LLM to generate detailed multi-step structured output (claim extraction +
verification), which exceeded the output-token-per-minute limits of the
free-tier Groq models available for this project. A lightweight custom
faithfulness scorer (single-call, 1-5 rating) was used instead, which fits
within those limits while still measuring whether answers are grounded in the
retrieved context.

**Known limitation:** 1 of 15 questions ("What is Kahn's algorithm used for?")
returned no answer, since it's only a brief secondary mention within one video
rather than its own topic, and retrieval didn't surface it strongly enough to
pass the grounding bar. This is a real, common RAG failure mode (recall on
under-represented sub-topics) rather than a bug, and a good target for future
retrieval tuning.

See `07_eval.py` for the full evaluation harness, and `eval_results.json` for
per-question scores.

## Roadmap

- [x] Audio download + transcription pipeline (resumable)
- [x] Timestamp-preserving chunking
- [x] Embedding + Qdrant Cloud upload
- [x] Hybrid retrieval (BM25 + vector)
- [x] BGE reranking
- [x] Groq LLM answer generation with citations
- [x] Evaluation (custom faithfulness scorer + RAGAS answer relevancy)
- [x] FastAPI backend (`app.py`) with a built-in HTML frontend, for local use
- [x] Public deployment on HuggingFace Spaces (`gradio_app.py`) - deployed as
      Gradio rather than the FastAPI/Docker version, since HF Spaces now
      requires a paid plan for Docker Spaces; Gradio Spaces remain free

---

## Notes

- Source videos are not the author's own content; raw audio and transcripts are kept local
  (`.gitignore`'d) and not redistributed. Only code, chunk structure, and eval results are public.
- Pipeline is playlist-agnostic — works with any YouTube playlist, not just this one.

---

## Author

Built by Nitin Dhadwal as a portfolio project targeting software engineering roles.
