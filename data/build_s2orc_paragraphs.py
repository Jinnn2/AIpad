#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_s2orc_paragraphs.py

Use the public (parquet) release of the modular S2ORC corpus
`claran/modular-s2orc-parquet` to extract paragraph-ready paper
text and convert it into LineArt JSON that follows JSON_requirements.md.

Why this dataset:
    * Each record already contains pre-parsed text for a single paper.
    * Data is stored as parquet shards on the Hub, so it works with
      `datasets>=4` without any legacy scripts.
    * Text is split into paragraphs with minimal LaTeX markup.

Example:
    python data/build_s2orc_paragraphs.py \
        --collection "AgriculturalAndFoodSciences,2019-2020" \
        --hf-split train \
        --output-dir data/s2orc_json \
        --max-articles 30
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset

DATASET_REPO = "claran/modular-s2orc-parquet"
DEFAULT_COLLECTION = "AgriculturalAndFoodSciences,2019-2020"

# ========= Layout constants =========
MARGIN_LEFT = 80
MARGIN_TOP = 80
TEXTBOX_WIDTH = 780
PARA_GAP = 22
AVG_CHARS_PER_LINE = 38
MAX_PARAGRAPH_CHARS = 600
DEFAULT_LINE_HEIGHT = 1.2
DEFAULT_PADDING = 8

TITLE_STYLE = {"fontFamily": "sans-serif", "fontSize": 36, "fontWeight": "700"}
ABSTRACT_STYLE = {"fontFamily": "sans-serif", "fontSize": 18, "fontWeight": "500"}
SUBTITLE_STYLE = {"fontFamily": "sans-serif", "fontSize": 26, "fontWeight": "600"}
BODY_STYLE = {"fontFamily": "sans-serif", "fontSize": 16, "fontWeight": "400"}

LATEX_TOKEN_RE = re.compile(r"@xmath\d+")
INLINE_LATEX_RE = re.compile(r"\$[^$]+\$")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\{[^}]+\}")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\s*")


# ========= Text helpers =========
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n")
    cleaned = LATEX_TOKEN_RE.sub(" ", cleaned)
    cleaned = INLINE_LATEX_RE.sub(" ", cleaned)
    cleaned = CITE_RE.sub(" ", cleaned)
    cleaned = COMMAND_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_into_blocks(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    normalized = raw_text.replace("\r\n", "\n")
    blocks = [clean_text(block) for block in normalized.split("\n\n")]
    return [block for block in blocks if block]


def is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 3:
        return False
    ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return ratio >= 0.65


def strip_abstract_prefix(paragraph: str) -> str:
    lowered = paragraph.lower()
    if lowered.startswith("abstract"):
        remainder = paragraph[len("abstract") :].lstrip(" :-–—")
        return remainder.strip() or paragraph
    return paragraph


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.!?。！？])\s+")


def chunk_long_text(text: str) -> List[str]:
    sentences = [
        normalize_whitespace(sentence)
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if normalize_whitespace(sentence)
    ]
    if not sentences:
        sentences = [normalize_whitespace(text)]

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for sentence in sentences:
        length = len(sentence)
        if buffer and buffer_len + length + 1 > MAX_PARAGRAPH_CHARS:
            chunks.append(" ".join(buffer))
            buffer = [sentence]
            buffer_len = length
        else:
            buffer.append(sentence)
            buffer_len += length + (1 if buffer_len else 0)
    if buffer:
        chunks.append(" ".join(buffer))

    final: List[str] = []
    for chunk in chunks:
        if len(chunk) <= MAX_PARAGRAPH_CHARS:
            final.append(chunk)
        else:
            for idx in range(0, len(chunk), MAX_PARAGRAPH_CHARS):
                final.append(chunk[idx : idx + MAX_PARAGRAPH_CHARS].strip())
    return final


def expand_role_chunks(role: str, text: str) -> List[Dict[str, str]]:
    if role in {"title", "subtitle"}:
        return [{"role": role, "text": text}]

    pieces = chunk_long_text(text)
    return [{"role": role, "text": piece} for piece in pieces]


def classify_paragraphs(raw_text: str, max_paragraphs: Optional[int]) -> List[Dict[str, str]]:
    blocks = split_into_blocks(raw_text)
    if not blocks:
        return []

    paragraphs: List[Dict[str, str]] = []
    title = blocks.pop(0)
    paragraphs.append({"role": "title", "text": title})

    if blocks:
        first = blocks[0]
        looks_like_abs = first.lower().startswith("abstract") or len(first) <= 360
        if looks_like_abs:
            abstract_text = strip_abstract_prefix(first)
            for chunk in expand_role_chunks("abstract", abstract_text):
                paragraphs.append(chunk)
            blocks = blocks[1:]

    for block in blocks:
        role = "subtitle" if is_heading(block) else "body"
        for chunk in expand_role_chunks(role, block):
            paragraphs.append(chunk)
            if max_paragraphs is not None and len(paragraphs) >= max_paragraphs:
                return paragraphs

    return paragraphs[:max_paragraphs] if max_paragraphs else paragraphs


# ========= Stroke builders =========
def style_for_role(role: str) -> Dict[str, Any]:
    if role == "title":
        return TITLE_STYLE
    if role == "abstract":
        return ABSTRACT_STYLE
    if role == "subtitle":
        return SUBTITLE_STYLE
    return BODY_STYLE


def estimate_height(text: str, font_size: int) -> int:
    chars = max(1, len(text.replace(" ", "")))
    lines = max(1, math.ceil(chars / AVG_CHARS_PER_LINE))
    return int(lines * font_size * DEFAULT_LINE_HEIGHT + DEFAULT_PADDING * 2)


def make_text_stroke(
    stroke_id: str, role: str, text: str, x0: float, y0: float, width: float
) -> Tuple[Dict[str, Any], int]:
    style = style_for_role(role)
    height = estimate_height(text, style["fontSize"])
    x1 = x0 + width
    y1 = y0 + height
    stroke = {
        "id": stroke_id,
        "tool": "text",
        "points": [[x0, y0], [x1, y1]],
        "style": {"size": "m", "color": "black", "opacity": 1},
        "meta": {
            "text": text,
            "summary": text[:80],
            "fontFamily": style["fontFamily"],
            "fontSize": style["fontSize"],
            "fontWeight": style["fontWeight"],
            "growDir": "down",
            "lineHeight": DEFAULT_LINE_HEIGHT,
            "padding": DEFAULT_PADDING,
            "configuredWidth": width,
            "configuredHeight": height,
        },
    }
    return stroke, height


def build_strokes(paragraphs: Sequence[Dict[str, str]], article_id: str) -> List[Dict[str, Any]]:
    strokes: List[Dict[str, Any]] = []
    x, y = MARGIN_LEFT, MARGIN_TOP
    for idx, para in enumerate(paragraphs):
        stroke_id = f"{article_id}-p{idx:03d}"
        stroke, box_h = make_text_stroke(stroke_id, para["role"], para["text"], x, y, TEXTBOX_WIDTH)
        strokes.append(stroke)
        y += box_h + PARA_GAP
    return strokes


def slugify(text: str, max_len: int = 80) -> str:
    slug = normalize_whitespace(text.lower())
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-_]+", "", slug)
    slug = slug.strip("-")
    return slug[:max_len] or "paper"


def build_data_files_pattern(collection: str, hf_split: str) -> str:
    collection = collection.strip()
    if not collection:
        raise ValueError("collection cannot be empty")
    if hf_split not in {"train", "validation", "test"}:
        raise ValueError("hf_split must be one of train/validation/test")
    return f"hf://datasets/{DATASET_REPO}/{collection}/{hf_split}-*.parquet"


# ========= Processing =========
def process_modular_s2orc(
    output_dir: Path,
    collection: str,
    hf_split: str,
    max_articles: Optional[int],
    min_paragraphs: int,
    max_paragraphs: Optional[int],
    streaming: bool,
) -> None:
    data_files = build_data_files_pattern(collection, hf_split)
    mode = "streaming" if streaming else "full download"
    print(
        f"[INFO] Loading {DATASET_REPO} collection='{collection}' "
        f"split='{hf_split}' via {data_files} ({mode})"
    )
    try:
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=streaming,
        )
    except Exception as exc:  # pragma: no cover - surfacing context helps debugging
        raise RuntimeError(
            f"Failed to load dataset files using pattern {data_files}. "
            f"Please double-check the collection/split name."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    scanned = 0

    for example in dataset:
        scanned += 1
        article_id = str(example.get("id") or example.get("metadata", {}).get("sha1") or scanned)
        raw_text = example.get("text") or ""
        paragraphs = classify_paragraphs(raw_text, max_paragraphs=max_paragraphs)
        if len(paragraphs) < min_paragraphs:
            continue

        title = next((para["text"] for para in paragraphs if para["role"] == "title"), article_id)
        strokes = build_strokes(paragraphs, article_id)
        payload = {"version": 1, "intent": "import", "strokes": strokes}
        filename = f"{article_id}_{slugify(title)}.json"
        with (output_dir / filename).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        exported += 1
        if exported % 20 == 0:
            print(f"[INFO] Exported {exported} papers (scanned {scanned})")
        if max_articles is not None and exported >= max_articles:
            break

    print(f"[DONE] Exported {exported} papers (scanned {scanned}) -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LineArt JSON from modular S2ORC parquet shards"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./s2orc_json",
        help="Directory to store generated JSON files",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Sub-directory name inside the dataset (e.g. 'AgriculturalAndFoodSciences,2019-2020')",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Which shard split to read from inside the collection directory",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of papers to export (None for all)",
    )
    parser.add_argument(
        "--min-paragraphs",
        type=int,
        default=5,
        help="Skip papers that produce fewer than this number of paragraphs",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=180,
        help="Trim each paper to at most this many paragraphs (None for unlimited)",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="Enable streaming (default)",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming and materialize each shard locally",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()
    process_modular_s2orc(
        output_dir=Path(args.output_dir),
        collection=args.collection,
        hf_split=args.hf_split,
        max_articles=args.max_articles,
        min_paragraphs=args.min_paragraphs,
        max_paragraphs=args.max_paragraphs,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
