#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_structwiki_fragments.py

基于 wikimedia/structured-wikipedia 数据集，生成 LineArt 可导入的 JSON：
- 数据源：structured-wikipedia（结构化 Wikipedia）
- 每条样本（一篇文章）→ 一份 JSON 文件
- 自动从 sections 中抽取 paragraph，当成段落列表
- 标题 + 小标题 + 正文，生成 text 类型 stroke（fragment）
- 竖直排版：一列文本块，对应 text bbox 的 points。

用法示例：
    python build_structwiki_fragments.py \
        --output-dir ./wiki_json \
        --config-name 20240916.en \
        --max-articles 200

注意：config-name 可以在 HF 数据集页面查看，当前官方示例有类似 20240916.en 等配置。:contentReference[oaicite:2]{index=2}
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from datasets import load_dataset


# ===================== 基本排版参数 =====================

CANVAS_WIDTH = 1200          # 虚拟画布宽度（目前只影响 bbox 范围）
MARGIN_LEFT = 80             # 左边距
MARGIN_TOP = 80              # 上边距
TEXTBOX_WIDTH = 760          # 文本框宽度
PARA_GAP = 24                # 段落之间的垂直间距
AVG_CHARS_PER_LINE = 36      # 每行大约容纳的字符数（估算高度用）

# 字体配置：可以按需调整
TITLE_STYLE = {
    "fontFamily": "sans-serif",
    "fontSize": 28,
    "fontWeight": "700",
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


# ===================== 工具函数 =====================

def normalize_whitespace(text: str) -> str:
    """把内部多余空白合并成单空格。"""
    return re.sub(r"\s+", " ", text).strip()


def collect_paragraphs_from_sections(sections: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    从 structured-wikipedia 的 sections 结构中，抽取所有 paragraph 的 value。
    只保留 type == "paragraph" 的内容。
    """
    paras: List[str] = []
    if not sections:
        return paras

    for sec in sections:
        parts = sec.get("has_parts") or []
        for p in parts:
            if p.get("type") == "paragraph" and p.get("value"):
                paras.append(normalize_whitespace(p["value"]))
    return paras


# ===================== 段落角色判定 =====================

SUBTITLE_PATTERN = re.compile(
    r"""^(
        [0-9]+[\.、)]\s*              |  # 1.  2、 3)
        [一二三四五六七八九十]+[\.、)]\s*  # 一、 二.
    )""",
    re.X
)


def classify_paragraphs(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """
    给每个段落打标签：
    - 第一个段落：title
    - 满足“1. / 一、”之类且长度较短的：subtitle
    - 其他：body
    返回：[{ "role": "title/subtitle/body", "text": "..." }, ...]
    """
    result: List[Dict[str, Any]] = []
    for idx, para in enumerate(paragraphs):
        role = "body"
        if idx == 0:
            role = "title"
        else:
            if len(para) <= 25 and SUBTITLE_PATTERN.match(para):
                role = "subtitle"
        result.append({"role": role, "text": para})
    return result


def estimate_box_height(text: str, font_size: int) -> int:
    """用“平均每行字符数 + 行间距”粗略估算文本框高度。"""
    if not text:
        return int(font_size * 1.6) + DEFAULT_PADDING * 2

    stripped = "".join(text.split())
    char_count = max(1, len(stripped))
    lines = max(1, math.ceil(char_count / AVG_CHARS_PER_LINE))
    content_height = lines * font_size * DEFAULT_LINE_HEIGHT
    return int(content_height + DEFAULT_PADDING * 2)


def style_for_role(role: str) -> Dict[str, Any]:
    if role == "title":
        return TITLE_STYLE
    elif role == "subtitle":
        return SUBTITLE_STYLE
    else:
        return BODY_STYLE


def make_text_stroke(
    stroke_id: str,
    role: str,
    text: str,
    x0: float,
    y0: float,
    width: float,
) -> Tuple[Dict[str, Any], int]:
    """
    根据段落信息生成一个 AIStrokeV11 text stroke。
    points: [[x0,y0], [x1,y1]]
    meta 里写入 text/summary/fontSize 等。
    """
    font_cfg = style_for_role(role)
    font_family = font_cfg["fontFamily"]
    font_size = font_cfg["fontSize"]
    font_weight = font_cfg["fontWeight"]

    box_height = estimate_box_height(text, font_size)
    x1 = x0 + width
    y1 = y0 + box_height

    summary = text[:40]

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
            "fontFamily": font_family,
            "fontSize": font_size,
            "fontWeight": font_weight,
            "growDir": "down",
            "lineHeight": DEFAULT_LINE_HEIGHT,
            "padding": DEFAULT_PADDING,
            "configuredWidth": width,
            "configuredHeight": box_height,
        },
    }
    return stroke, box_height


def build_article_strokes(paragraphs_with_roles: List[Dict[str, Any]], article_id: str) -> List[Dict[str, Any]]:
    """
    把整篇文章的段落（已带 role）转换成一组 text strokes，
    按顺序排成竖直的一列。
    """
    strokes: List[Dict[str, Any]] = []
    x = MARGIN_LEFT
    y = MARGIN_TOP

    for idx, para in enumerate(paragraphs_with_roles):
        role = para["role"]
        text = para["text"]
        stroke_id = f"{article_id}-p{idx:03d}"

        stroke, box_height = make_text_stroke(
            stroke_id=stroke_id,
            role=role,
            text=text,
            x0=x,
            y0=y,
            width=TEXTBOX_WIDTH,
        )
        strokes.append(stroke)
        y += box_height + PARA_GAP

    return strokes


def slugify(text: str, max_len: int = 60) -> str:
    """
    用于生成文件名安全的 slug：只保留字母数字和 - _
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    if not text:
        text = "article"
    return text[:max_len]


# ===================== 处理 structured-wikipedia 数据集 =====================

def process_structured_wikipedia(
    output_dir: Path,
    config_name: str,
    split: str = "train",
    max_articles: Optional[int] = None,
    min_paragraphs: int = 1,
    language_filter: Optional[str] = None,
    streaming: bool = True,
) -> None:
    """
    从 wikimedia/structured-wikipedia 读取数据，生成 JSON。
    - config_name: 例如 '20240916.en'
    - split: 默认 'train'
    - max_articles: 最多处理多少篇文章（None 表示全部）
    - min_paragraphs: 至少多少段才导出
    - language_filter: 若非空，则仅保留 in_language['code'] == 此值 的文章
    """
    mode = "streaming" if streaming else "full download"
    print(
        f"[INFO] Loading dataset wikimedia/structured-wikipedia "
        f"({config_name}, split={split}, mode={mode}) ..."
    )
    ds = load_dataset(
        "wikimedia/structured-wikipedia",
        config_name,
        split=split,
        streaming=streaming,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    total = 0

    for idx, ex in enumerate(ds):
        total += 1
        # 语言过滤（如果需要）
        if language_filter:
            in_lang = ex.get("in_language") or {}
            code = in_lang.get("code") or in_lang.get("id")
            if code and code != language_filter:
                continue

        title = ex.get("name") or ""
        sections = ex.get("sections") or []

        paragraphs = collect_paragraphs_from_sections(sections)
        if not paragraphs or len(paragraphs) < min_paragraphs:
            continue

        # 把标题作为第一个段落，交给 classify_paragraphs 决定 role
        layout_paras: List[str] = []
        if title:
            layout_paras.append(title.strip())
        layout_paras.extend(paragraphs)

        labeled_paras = classify_paragraphs(layout_paras)

        # 生成 article_id & 文件名
        identifier = str(ex.get("identifier", "") or "")
        slug = slugify(title) if title else "article"
        if identifier:
            article_id = f"{identifier}"
            filename = f"{identifier}_{slug}.json"
        else:
            article_id = f"idx{idx:07d}"
            filename = f"idx{idx:07d}_{slug}.json"

        strokes = build_article_strokes(labeled_paras, article_id=article_id)

        payload = {
            "version": 1,
            "intent": "import",
            "strokes": strokes,
        }

        out_path = output_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        exported += 1
        if exported % 50 == 0:
            print(f"[INFO] Exported {exported} articles (scanned {total})")

        if max_articles is not None and exported >= max_articles:
            break

    print(f"[DONE] Exported {exported} articles (scanned {total}) → {output_dir}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="从 wikimedia/structured-wikipedia 生成 LineArt text fragment JSON 数据集"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./wiki_json",
        help="输出 JSON 文件目录",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="20251119.en",
        help="HuggingFace 数据集配置名，例如 20240916.en（可在 HF 页面查看）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集 split，默认 train",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="最多导出多少篇文章（默认全部）",
    )
    parser.add_argument(
        "--min-paragraphs",
        type=int,
        default=1,
        help="至少包含多少段落才导出该文章，默认 1",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="按 in_language.code 过滤语言，例如 'en'；默认不过滤",
    )

    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        help="使用 HuggingFace streaming，按需下载样本（默认开启）",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="禁用 streaming，回退到一次性下载完整数据集",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    process_structured_wikipedia(
        output_dir=output_dir,
        config_name=args.config_name,
        split=args.split,
        max_articles=args.max_articles,
        min_paragraphs=args.min_paragraphs,
        language_filter=args.language,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
