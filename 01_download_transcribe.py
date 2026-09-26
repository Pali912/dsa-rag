"""
Step 1: Download audio from a YouTube playlist and transcribe it with timestamps.

Usage:
    python 01_download_transcribe.py --playlist "https://youtube.com/playlist?list=XXXX" --limit 20
    python 01_download_transcribe.py --urls urls.txt

Output:
    data/audio/<video_id>.mp3           (downloaded audio)
    data/transcripts/<video_id>.json    (video metadata + timestamped segments)

Resumable: if a video's audio or transcript already exists on disk, it is skipped.
Safe to Ctrl+C and rerun any time.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel
from tqdm import tqdm

AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts")
WHISPER_MODEL_SIZE = "base"  # "tiny"/"base" = fast+free on CPU; "small"/"medium" = better quality, needs GPU for speed


def ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def extract_video_id(url: str) -> str:
    """Handles both youtu.be/ID and watch?v=ID (with or without extra query params)."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract a video ID from URL: {url}")
    return match.group(1)


def get_playlist_video_ids(playlist_url: str, limit: int | None) -> list[dict]:
    """Return a list of {id, title, url} without downloading, so we can filter/limit first."""
    ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    entries = info.get("entries", [])
    if limit:
        entries = entries[:limit]
    videos = []
    for e in entries:
        vid = e.get("id")
        videos.append(
            {
                "id": vid,
                "title": e.get("title", vid),
                "url": f"https://www.youtube.com/watch?v={vid}",
            }
        )
    return videos


def download_audio(video: dict) -> Path:
    """Download audio-only for one video. Skips if already downloaded."""
    out_path = AUDIO_DIR / f"{video['id']}.mp3"
    if out_path.exists():
        print(f"  [skip] audio already downloaded: {video['id']}")
        return out_path

    progress_bar = {"bar": None}

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if progress_bar["bar"] is None and total:
                progress_bar["bar"] = tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading audio")
            if progress_bar["bar"] is not None:
                progress_bar["bar"].n = d.get("downloaded_bytes", 0)
                progress_bar["bar"].refresh()
        elif d["status"] == "finished":
            if progress_bar["bar"] is not None:
                progress_bar["bar"].close()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{video['id']}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video["url"]])
    return out_path


def transcribe_audio(model: WhisperModel, audio_path: Path, video: dict) -> Path:
    """Transcribe one audio file with timestamps. Skips if transcript already exists."""
    out_path = TRANSCRIPT_DIR / f"{video['id']}.json"
    if out_path.exists():
        print(f"  [skip] transcript already exists: {video['id']}")
        return out_path

    segments, info = model.transcribe(str(audio_path), language="en", vad_filter=True)

    segment_list = []
    with tqdm(total=round(info.duration), unit="s", desc="  Transcribing") as pbar:
        last_end = 0.0
        for seg in segments:
            segment_list.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                }
            )
            pbar.update(seg.end - last_end)
            last_end = seg.end

    record = {
        "video_id": video["id"],
        "title": video["title"],
        "url": video["url"],
        "language": info.language,
        "segments": segment_list,
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlist", help="YouTube playlist URL")
    parser.add_argument("--urls", help="Text file with one YouTube video URL per line (alternative to --playlist)")
    parser.add_argument("--limit", type=int, default=20, help="Max videos to process (default 20)")
    parser.add_argument(
        "--model-size",
        default=WHISPER_MODEL_SIZE,
        help="faster-whisper model size: tiny/base/small/medium/large-v3 (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="cpu (default, works everywhere) or cuda (needs a working CUDA install)",
    )
    args = parser.parse_args()

    if not args.playlist and not args.urls:
        print("Provide either --playlist or --urls", file=sys.stderr)
        sys.exit(1)

    ensure_dirs()

    if args.playlist:
        print(f"Fetching playlist metadata: {args.playlist}")
        videos = get_playlist_video_ids(args.playlist, args.limit)
    else:
        lines = Path(args.urls).read_text().splitlines()
        videos = []
        for url in lines:
            url = url.strip()
            if not url:
                continue
            vid = extract_video_id(url)
            clean_url = f"https://www.youtube.com/watch?v={vid}"
            videos.append({"id": vid, "title": vid, "url": clean_url})
        videos = videos[: args.limit]

    print(f"{len(videos)} videos queued.")
    print(f"Loading faster-whisper model '{args.model_size}' (first run downloads the model)...")
    # compute_type="int8" keeps this fast and low-memory on CPU; use "float16" if you switch to --device cuda
    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model_size, device=args.device, compute_type=compute_type)

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video['title']} ({video['id']})")
        try:
            audio_path = download_audio(video)
            transcribe_audio(model, audio_path, video)
        except Exception as e:
            print(f"  [ERROR] {video['id']}: {e}", file=sys.stderr)
            print("  Continuing with next video. Rerun this script later to retry failed ones.")
            continue

    print("\nDone. Transcripts saved to data/transcripts/")


if __name__ == "__main__":
    main()
