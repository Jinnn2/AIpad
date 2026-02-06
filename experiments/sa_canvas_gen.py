from __future__ import annotations

import argparse
import json
import random
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.config import DATA_DIR, SA_DIR, N_SA_DEV, N_SA_TEST, N_SA_TRAIN

# Layout constants (measured: font size 18, line height ~28px => multiplier ~1.56, avg char width 10.4)
TEXTBOX_WIDTH = 760.0
TITLE_MAX_WIDTH = 640.0
MIN_BOX_WIDTH = 240.0
CHAR_WIDTH = 10.4
FONT_SIZE = 18.0
LINE_HEIGHT_MULT = 28.0 / 18.0  # ~1.56
MIN_LINES = 1
PADDING = 3.0  # target height ~ line_height_px + 2 * padding (e.g., 28 + 6 = 34)
INNER_BLOCK_GAP = 24
COLUMN_GAP_X = 200
MARGIN_X = 200
MARGIN_Y = 160
MAX_PARAS_PER_COL = 7
MAX_COLS = 2

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def snap(v: float) -> int:
    # Minimal snapping to preserve measured heights.
    return int(round(v))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _coerce_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value if x)
    return str(value or "")


def _bbox_from_points(points: List[List[float]]) -> Optional[List[float]]:
    if not points or len(points) < 2:
        return None
    try:
        (x0, y0), (x1, y1) = points[0], points[1]
    except Exception:
        return None
    return [float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1))]


def _load_blocks_from_strokes(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    blocks: List[Dict] = []
    for stroke in payload.get("strokes", []):
        if stroke.get("tool") != "text":
            continue
        meta = stroke.get("meta") or {}
        text = (meta.get("text") or meta.get("summary") or "").strip()
        if not text:
            continue
        bbox = _bbox_from_points(stroke.get("points") or [])
        block_type = "title" if meta.get("isTitle") else "body"
        blocks.append(
            {
                "block_id": f"b{len(blocks) + 1}",
                "bbox": bbox,
                "text": text,
                "type": block_type,
            }
        )
    return blocks


def _build_queries(blocks: List[Dict], *, max_queries: int = 3, candidate_k: int = 5) -> List[Dict]:
    queries: List[Dict] = []
    if len(blocks) < 2:
        return queries
    pool = blocks[:]
    random.shuffle(pool)

    def first_sentence(text: str) -> str:
        parts = SENT_SPLIT_RE.split(text)
        for p in parts:
            cleaned = p.strip()
            if cleaned:
                return cleaned
        return text.strip()

    for block in pool:
        answer_id = block["block_id"]
        snippet = first_sentence(block["text"])
        if len(snippet) > 240:
            snippet = snippet[:240].rsplit(" ", 1)[0]
        other_ids = [b["block_id"] for b in blocks if b["block_id"] != answer_id]
        random.shuffle(other_ids)
        candidates = [answer_id] + other_ids[: max(0, candidate_k - 1)]
        # dedupe then shuffle for stability
        seen = set()
        candidate_block_ids: List[str] = []
        for cid in candidates:
            if cid in seen:
                continue
            seen.add(cid)
            candidate_block_ids.append(cid)
        random.shuffle(candidate_block_ids)
        queries.append(
            {
                "query_id": f"q{len(queries) + 1}",
                "query_text": snippet,
                "answer_block_id": answer_id,
                "candidate_block_ids": candidate_block_ids,
                "query_block_id": answer_id,  # anchor block for spatial baselines
            }
        )
        if len(queries) >= max_queries:
            break
    return queries


def _iter_source_files(source_dir: Path) -> List[Path]:
    return sorted(p for p in source_dir.glob("*.json") if p.is_file())


def _estimate_height(text_len: int, box_width: float, padding: float) -> int:
    usable_width = max(box_width - 2 * padding, CHAR_WIDTH)
    chars_per_line = max(1.0, usable_width / CHAR_WIDTH)
    line_count = max(MIN_LINES, int((text_len / chars_per_line) + 0.999))
    line_height_px = LINE_HEIGHT_MULT * FONT_SIZE
    height = padding * 2 + line_count * line_height_px
    return snap(height)


def _measure_box(text: str, max_width: float, *, min_width: float, padding: float) -> (float, int):
    text_len = len(text)
    content_width = text_len * CHAR_WIDTH
    width = min(max_width, max(min_width, padding * 2 + content_width))
    usable_width = max(width - 2 * padding, CHAR_WIDTH)
    chars_per_line = max(1.0, usable_width / CHAR_WIDTH)
    line_count = max(MIN_LINES, int((text_len / chars_per_line) + 0.999))
    line_height_px = LINE_HEIGHT_MULT * FONT_SIZE
    height = padding * 2 + line_count * line_height_px
    return width, snap(height)


def _split_into_paragraphs(
    text: str,
    *,
    min_chars: int = 220,
    max_chars: int = 520,
    max_paras: int = MAX_COLS * MAX_PARAS_PER_COL,
) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    paragraphs: List[str] = []
    cur = ""
    for sent in sentences:
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = f"{cur} {sent}"
        elif len(cur) >= min_chars:
            paragraphs.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}"
        if len(paragraphs) >= max_paras:
            break
    if cur and len(paragraphs) < max_paras:
        paragraphs.append(cur)
    if len(paragraphs) >= 2 and len(paragraphs[-1]) < min_chars * 0.5:
        paragraphs[-2] = f"{paragraphs[-2]} {paragraphs[-1]}"
        paragraphs.pop()
    return paragraphs[:max_paras]


def _build_blocks_from_article(article: Dict) -> List[Dict]:
    blocks: List[Dict] = []
    title = clean_text(_coerce_text(article.get("title", "")))
    body_text = _coerce_text(article.get("text", "")) or _coerce_text(article.get("summary", ""))
    paragraphs = _split_into_paragraphs(body_text)
    if not paragraphs:
        return blocks

    y_cursor = MARGIN_Y
    if title:
        title_width, h = _measure_box(title, TITLE_MAX_WIDTH, min_width=MIN_BOX_WIDTH, padding=PADDING)
        blocks.append(
            {
                "block_id": f"b{len(blocks) + 1}",
                "bbox": [MARGIN_X, y_cursor, MARGIN_X + title_width, y_cursor + h],
                "text": title,
                "type": "title",
            }
        )
        y_cursor = y_cursor + h + 2 * INNER_BLOCK_GAP

    col_y = [y_cursor for _ in range(MAX_COLS)]
    col = 0
    for para in paragraphs:
        if len(blocks) >= MAX_COLS * MAX_PARAS_PER_COL + (1 if title else 0):
            break
        if (len(blocks) - (1 if title else 0)) >= MAX_PARAS_PER_COL * (col + 1):
            col += 1
        if col >= MAX_COLS:
            break
        h = _estimate_height(len(para), TEXTBOX_WIDTH, padding=PADDING)
        x0 = MARGIN_X + col * (TEXTBOX_WIDTH + COLUMN_GAP_X)
        y0 = col_y[col]
        blocks.append(
            {
                "block_id": f"b{len(blocks) + 1}",
                "bbox": [x0, y0, x0 + TEXTBOX_WIDTH, y0 + h],
                "text": para,
                "type": "body",
            }
        )
        col_y[col] = y0 + h + INNER_BLOCK_GAP
    return blocks


def _write_canvas(
    *,
            split: str,
            index: int,
            blocks: List[Dict],
            queries: List[Dict],
            source_name: str,
) -> None:
    SA_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = SA_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_id = f"sa_{split}_{index:06d}"

    strokes = []
    for block in blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        x0, y0, x1, y1 = bbox
        strokes.append(
            {
                "id": f"{canvas_id}_{block['block_id']}",
                "tool": "text",
                "points": [[x0, y0], [x1, y1]],
                "style": {"size": "m", "color": "black", "opacity": 1.0},
                "meta": {
                    "text": block.get("text", ""),
                    "summary": block.get("text", ""),
                    "fontFamily": "sans-serif",
                    "fontSize": FONT_SIZE,
                    "fontWeight": 700 if block.get("type") == "title" else 400,
                    "growDir": "down",
                    "lineHeight": LINE_HEIGHT_MULT,  # multiplier expected by frontend
                    "padding": PADDING,
                    "configuredWidth": max(0.0, float(x1 - x0)),
                    "configuredHeight": max(0.0, float(y1 - y0)),
                },
            }
        )

    sample = {
        "version": 1,
        "intent": "import",
        "canvas_id": canvas_id,
        "strokes": strokes,
        "blocks": blocks,
        "retrieval_queries": queries,
        "source": source_name,
    }
    out_path = out_dir / f"{canvas_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


def build_split(paths: List[Path], split: str, target: int, *, seed: int) -> int:
    random.seed(seed + hash(split) % 9973)
    written = 0
    for path in paths:
        if written >= target:
            break
        blocks = _load_blocks_from_strokes(path)
        if len(blocks) < 3:
            continue
        queries = _build_queries(blocks)
        if not queries:
            continue
        _write_canvas(split=split, index=written, blocks=blocks, queries=queries, source_name=path.name)
        written += 1
    return written


def build_from_dataset(
    *,
    dataset_name: str,
    dataset_split: str,
    train: int,
    dev: int,
    test: int,
    seed: int,
    sample: Optional[int] = None,
) -> Dict[str, int]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("`datasets` package is required. Install via `pip install datasets`.") from exc

    total_needed = train + dev + test
    ds = load_dataset(dataset_name, split=dataset_split)
    if sample is None:
        sample = total_needed * 2 + 10
    ds = ds.shuffle(seed=seed).select(range(min(sample, len(ds))))

    counts = {"train": 0, "dev": 0, "test": 0}
    split_order = [("train", train), ("dev", dev), ("test", test)]
    split_idx = 0
    written_idx = {"train": 0, "dev": 0, "test": 0}

    for idx, item in enumerate(ds):
        current_split, target = split_order[split_idx]
        if counts[current_split] >= target:
            split_idx += 1
            if split_idx >= len(split_order):
                break
            current_split, target = split_order[split_idx]

        blocks = _build_blocks_from_article(item)
        if len(blocks) < 3:
            continue
        queries = _build_queries(blocks)
        if not queries:
            continue
        _write_canvas(
            split=current_split,
            index=written_idx[current_split],
            blocks=blocks,
            queries=queries,
            source_name=f"{dataset_name}:{idx}",
        )
        counts[current_split] += 1
        written_idx[current_split] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SA-Canvas samples from HF dataset or prebuilt stroke JSON.")
    parser.add_argument(
        "--mode",
        choices=["dataset", "strokes"],
        default="dataset",
        help="dataset: sample from HF dataset; strokes: read prebuilt stroke JSON files.",
    )
    parser.add_argument("--source_dir", type=Path, default=DATA_DIR / "wikihow_json", help="Directory of source stroke JSON files.")
    parser.add_argument("--dataset_name", type=str, default="gursi26/wikihow-cleaned", help="HF dataset name.")
    parser.add_argument("--dataset_split", type=str, default="train", help="HF dataset split.")
    parser.add_argument("--sample", type=int, default=None, help="Sample size from dataset (defaults to ~2x needed).")
    parser.add_argument("--train", type=int, default=N_SA_TRAIN, help="Number of train canvases.")
    parser.add_argument("--dev", type=int, default=N_SA_DEV, help="Number of dev canvases.")
    parser.add_argument("--test", type=int, default=N_SA_TEST, help="Number of test canvases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.mode == "strokes":
        paths = _iter_source_files(args.source_dir)
        if not paths:
            raise SystemExit(f"No source files found under {args.source_dir}")
        cursor = 0
        counts = {}
        for split, target in (("train", args.train), ("dev", args.dev), ("test", args.test)):
            remaining = paths[cursor:]
            if not remaining:
                counts[split] = 0
                continue
            written = build_split(remaining, split, target, seed=args.seed)
            counts[split] = written
            cursor += written
    else:
        counts = build_from_dataset(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            train=args.train,
            dev=args.dev,
            test=args.test,
            seed=args.seed,
            sample=args.sample,
        )

    summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
    print(f"SA-Canvas generation done -> {SA_DIR} ({summary})")


if __name__ == "__main__":
    main()
