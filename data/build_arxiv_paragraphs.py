#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_arxiv_paragraphs.py

从 HuggingFace 的 ccdv/arxiv-summarization 数据集（收录大量 ArXiv 论文）
中按段落提取正文，转换成符合 JSON_requirements.md 的 LineArt text strokes。

功能特性：
    - 默认启用 streaming，只下载正在处理的样本，避免全量落地。
    - 自动将标题、摘要、正文段落转换成 text stroke，竖排布局。
    - 可限制导出的篇数 / 段落数，方便快速取样。

使用示例：
    python data/build_arxiv_paragraphs.py \\
        --output-dir ./paper_json \\
        --split train \\
        --max-articles 50
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

# ===================== 基本排版参数 =====================

CANVAS_WIDTH = 1200
MARGIN_LEFT = 80
MARGIN_TOP = 80
TEXTBOX_WIDTH = 760
PARA_GAP = 24
AVG_CHARS_PER_LINE = 36
MAX_PARAGRAPH_CHARS = 600

TITLE_STYLE = {
    "fontFamily": "sans-serif",
    "fontSize": 30,
    "fontWeight": "700",
}

ABSTRACT_STYLE = {
    "fontFamily": "sans-serif",
    "fontSize": 18,
    "fontWeight": "500",
}

SUBTITLE_STYLE = {
    "fontFamily": "sans-serif",
    "fontSize": 20,
    "fontWeight": "600",
}

BODY_STYLE = {
    "fontFamily": "sans-serif",
    "fontSize": 16,
    "fontWeight": "400",
}

DEFAULT_LINE_HEIGHT = 1.4
DEFAULT_PADDING = 8

# 句子分割：支持中英文句号
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")

# ===================== 工具函数 =====================


def normalize_whitespace(text: str) -> str:
    """合并多余空白"""
    return re.sub(r"\s+", " ", text).strip()


def chunk_long_text(text: str) -> List[str]:
    """
    将长文本按句子切分为不超过 MAX_PARAGRAPH_CHARS 的块。
    """
    sentences = [
        normalize_whitespace(s)
        for s in SENTENCE_SPLIT_PATTERN.split(text)
        if normalize_whitespace(s)
    ]
    if not sentences:
        sentences = [normalize_whitespace(text)]

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if buffer and buffer_len + sentence_len + 1 > MAX_PARAGRAPH_CHARS:
            chunks.append(" ".join(buffer))
            buffer = [sentence]
            buffer_len = sentence_len
        else:
            buffer.append(sentence)
            buffer_len += sentence_len + (1 if buffer_len else 0)

    if buffer:
        chunks.append(" ".join(buffer))

    # 如果仍有超长块，则按字符硬切
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= MAX_PARAGRAPH_CHARS:
            final_chunks.append(chunk)
            continue
        for i in range(0, len(chunk), MAX_PARAGRAPH_CHARS):
            piece = chunk[i : i + MAX_PARAGRAPH_CHARS]
            final_chunks.append(piece.strip())
    return final_chunks


def split_into_paragraphs(text: str) -> List[str]:
    """
    将论文文本按照空行切分为段落，去除超短内容。
    """
    if not text:
        return []
    normalized_text = text.replace("\r\n", "\n").strip()
    if not normalized_text:
        return []

    # 先尝试按空行
    blocks = re.split(r"\n\s*\n+", normalized_text)
    paras: List[str] = [
        normalize_whitespace(block) for block in blocks if len(normalize_whitespace(block)) > 2
    ]

    # 若按空行几乎没有切开，再尝试按单行
    if len(paras) <= 1 and "\n" in normalized_text:
        single_line_blocks = [
            normalize_whitespace(block)
            for block in normalized_text.split("\n")
            if len(normalize_whitespace(block)) > 2
        ]
        if len(single_line_blocks) > len(paras):
            paras = single_line_blocks

    # 若仍只有一个大块，则按句子/字符切片
    if len(paras) <= 1:
        paras = chunk_long_text(normalized_text)

    return paras


def looks_like_heading(text: str) -> bool:
    """
    粗略判断段落是否是章节标题：
        - 字母字符数 >= 3
        - 包含的字母大部分为大写
        - 长度不超过 80
    """
    candidate = text.strip()
    if not candidate or len(candidate) > 80 or len(candidate) < 3:
        return False
    letters = [ch for ch in candidate if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper_count = sum(1 for ch in letters if ch.isupper())
    ratio = upper_count / len(letters)
    return ratio >= 0.7


def style_for_role(role: str) -> Dict[str, Any]:
    if role == "title":
        return TITLE_STYLE
    if role == "abstract":
        return ABSTRACT_STYLE
    if role == "subtitle":
        return SUBTITLE_STYLE
    return BODY_STYLE


def estimate_box_height(text: str, font_size: int) -> int:
    """根据字符数估算文本框高度"""
    if not text:
        return int(font_size * 1.6) + DEFAULT_PADDING * 2
    stripped = "".join(text.split())
    char_count = max(1, len(stripped))
    lines = max(1, math.ceil(char_count / AVG_CHARS_PER_LINE))
    content_height = lines * font_size * DEFAULT_LINE_HEIGHT
    return int(content_height + DEFAULT_PADDING * 2)


def make_text_stroke(
    stroke_id: str,
    role: str,
    text: str,
    x0: float,
    y0: float,
    width: float,
) -> Tuple[Dict[str, Any], int]:
    """返回 text stroke 及其高度"""
    font_cfg = style_for_role(role)
    font_size = font_cfg["fontSize"]
    box_height = estimate_box_height(text, font_size)
    x1 = x0 + width
    y1 = y0 + box_height
    summary = text[:60]

    stroke: Dict[str, Any] = {
        "id": stroke_id,
        "tool": "text",
        "points": [
            [x0, y0],
            [x1, y1],
        ],
        "style": {
            "size": "m",
            "color": "black",
            "opacity": 1,
        },
        "meta": {
            "text": text,
            "summary": summary,
            "fontFamily": font_cfg["fontFamily"],
            "fontSize": font_cfg["fontSize"],
            "fontWeight": font_cfg["fontWeight"],
            "growDir": "down",
            "lineHeight": DEFAULT_LINE_HEIGHT,
            "padding": DEFAULT_PADDING,
            "configuredWidth": width,
            "configuredHeight": box_height,
        },
    }
    return stroke, box_height


def build_text_column(
    paragraphs_with_roles: Iterable[Dict[str, str]],
    article_id: str,
) -> List[Dict[str, Any]]:
    """将段落列表堆叠成竖版 text strokes"""
    strokes: List[Dict[str, Any]] = []
    x = MARGIN_LEFT
    y = MARGIN_TOP

    for idx, para in enumerate(paragraphs_with_roles):
        role = para["role"]
        text = para["text"]
        stroke_id = f"{article_id}-p{idx:03d}"
        stroke, box_height = make_text_stroke(stroke_id, role, text, x, y, TEXTBOX_WIDTH)
        strokes.append(stroke)
        y += box_height + PARA_GAP
    return strokes


def slugify(text: str, max_len: int = 60) -> str:
    """文件名 slug"""
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    if not text:
        text = "paper"
    return text[:max_len]


def sanitize_identifier(value: str) -> str:
    """保证文件名片段合法"""
    clean = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", value.strip())
    return clean or "paper"


def collect_paragraphs_from_example(
    example: Dict[str, Any],
    max_paragraphs: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    提取标题 + 段落。返回 (paper_id, title, paragraphs_with_roles)
    """
    title = example.get("title") or example.get("paper_title") or ""
    article_id = (
        example.get("article_id")
        or example.get("paper_id")
        or example.get("id")
        or example.get("identifier")
        or ""
    )

    title = normalize_whitespace(title) if title else ""
    if not article_id:
        article_id = title or "paper"
    article_id = sanitize_identifier(str(article_id))
    if not title:
        title = f"ArXiv {article_id}"

    paragraphs: List[Dict[str, str]] = []
    if title:
        paragraphs.append({"role": "title", "text": title})

    abstract = example.get("abstract") or ""
    for para in split_into_paragraphs(abstract):
        paragraphs.append({"role": "abstract", "text": para})

    body_text = example.get("article") or example.get("text") or ""
    for para in split_into_paragraphs(body_text):
        role = "subtitle" if looks_like_heading(para) else "body"
        paragraphs.append({"role": role, "text": para})

    if max_paragraphs is not None and len(paragraphs) > max_paragraphs:
        paragraphs = paragraphs[:max_paragraphs]

    return article_id, title, paragraphs


# ===================== 主处理流程 =====================

def process_arxiv_dataset(
    output_dir: Path,
    split: str = "train",
    max_articles: Optional[int] = None,
    min_paragraphs: int = 3,
    max_paragraphs: Optional[int] = None,
    streaming: bool = True,
) -> None:
    dataset_name = "ccdv/arxiv-summarization"
    mode = "streaming" if streaming else "full download"
    print(f"[INFO] Loading {dataset_name} (split={split}, mode={mode}) ...")

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    total = 0

    for idx, example in enumerate(ds):
        total += 1
        article_id, title, paragraphs = collect_paragraphs_from_example(
            example, max_paragraphs=max_paragraphs
        )
        if len(paragraphs) < min_paragraphs:
            continue

        strokes = build_text_column(paragraphs, article_id=article_id)
        payload = {
            "version": 1,
            "intent": "import",
            "strokes": strokes,
        }

        slug = slugify(title)
        filename = f"{article_id}_{slug}.json"
        out_path = output_dir / filename

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        exported += 1
        if exported % 20 == 0:
            print(f"[INFO] Exported {exported} papers (scanned {total})")

        if max_articles is not None and exported >= max_articles:
            break

    print(f"[DONE] Exported {exported} papers (scanned {total}) -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 ccdv/arxiv-summarization 构建 LineArt JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./paper_json",
        help="输出 JSON 目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集 split，train/validation/test",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="最多导出多少篇论文（None 表示全部）",
    )
    parser.add_argument(
        "--min-paragraphs",
        type=int,
        default=3,
        help="至少包含多少段落才导出该论文",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=120,
        help="每篇最多保留的段落数，避免超长（None 表示不限）",
    )
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="启用 streaming，仅按需下载样本（默认）",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="禁用 streaming，会下载完整 split",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    process_arxiv_dataset(
        output_dir=output_dir,
        split=args.split,
        max_articles=args.max_articles,
        min_paragraphs=args.min_paragraphs,
        max_paragraphs=args.max_paragraphs,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
