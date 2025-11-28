import os
import re
import json
import math
import time
import argparse
import random
from pathlib import Path

from datasets import load_dataset


# =========================
# 基本常量（整洁优先）
# =========================
DATASET_NAME = "gursi26/wikihow-cleaned"
SPLIT = "train"
OUT_DIR = Path("./wikihow_json")

TEXTBOX_WIDTH = 760   # 单列基准宽（px）
CHAR_WIDTH = 18.0     # 每字符占位（px）
LINE_HEIGHT = 22      # 行高（px）
MIN_LINES = 3
PADDING_TOP_BOTTOM = 24  # 文本块上下内边距（px）

INNER_BLOCK_GAP = 24  # 文内：列/段落间距（px）
COLUMN_GAP_X = 200    # 卡片间水平间距（px）
ROW_GAP_Y = 160       # 行间距（px）
BASE = 8              # 吸附基线（px）

MAX_PARAS_PER_COL = 6         # 每列最多段落数
MAX_COLS_PER_ARTICLE = 4      # 单文最多列
TARGET_ASPECT_MIN = 1.6       # 目标宽高比范围（宽/高）
TARGET_ASPECT_MAX = 2.2

# 行最大宽度（批内搁板式）
MAX_ROW_WIDTH = 3 * TEXTBOX_WIDTH + 4 * COLUMN_GAP_X

# JSON 分片上限（按文本字符计）
DEFAULT_MAX_CHARS_PER_FILE = 100_000

# 文章段落硬上限（提速关键：达到即停止切分）
MAX_PARAS_PER_ARTICLE = MAX_COLS_PER_ARTICLE * MAX_PARAS_PER_COL  # 24

# 为避免在超长文章上耗时，正文字符硬截断（再配合段落上限）
# 估算：单段 ~ 450 字；24 段 ~ 10800；再加富余
MAX_TEXT_CHARS_PER_ARTICLE = 14_000

# 分句正则（预编译）
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# =========================
# 工具函数
# =========================
def snap(v: float, base: int = BASE) -> int:
    return int(round(v / base) * base)


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.strip().lower()
    t = re.sub(r"[^\w\s]+$", "", t)
    t = re.sub(r"\s+(part\s+)?\d+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# =========================
# 轻量两遍合并（避免一次性把全库文本装入内存）
# =========================
def index_articles(dataset):
    """
    第一遍只做分组索引，不读取正文，内存占用低。
    返回:
      groups_meta: { base_title: {"base_title", "first_idx", "titles": [...] } }
      per_item: [(idx, base_title, title), ...]  目前未使用，保留扩展
    """
    groups_meta = {}
    per_item = []
    for idx, item in enumerate(dataset):
        title = (item.get("title") or "").strip()
        base = normalize_title(title)
        if not base:
            continue
        g = groups_meta.get(base)
        if not g:
            g = {"base_title": base, "first_idx": idx, "titles": [title]}
            groups_meta[base] = g
        else:
            g["titles"].append(title)
            # 记录最早出现的样本下标用于排序
            g["first_idx"] = min(g["first_idx"], idx)
        per_item.append((idx, base, title))
    return groups_meta, per_item


def materialize_selected(dataset, selected_bases_set):
    """
    第二遍仅为选中的分组合并正文，避免把全库 text/summary 一次性装入内存。
    """
    groups = {b: {"base_title": b, "title": None, "parts": []} for b in selected_bases_set}
    for idx, item in enumerate(dataset):
        title = (item.get("title") or "").strip()
        base = normalize_title(title)
        if base not in groups:
            continue
        summary = (item.get("summary") or "").strip()
        text = (item.get("text") or "").strip()
        g = groups[base]
        if g["title"] is None:
            g["title"] = title
        g["parts"].append({"title": title, "summary": summary, "text": text, "index": idx})

    merged = list(groups.values())
    for g in merged:
        g["parts"].sort(key=lambda p: p["index"])
        if not g["title"] and g["parts"]:
            g["title"] = g["parts"][0]["title"]
    merged.sort(key=lambda g: (g["parts"][0]["index"] if g["parts"] else 1 << 60))
    return merged


# =========================
# 切段、估高、列分配与排布
# =========================
def split_into_chunks_fast(
    text: str,
    min_chars: int = 300,
    max_chars: int = 600,
    max_paras: int = MAX_PARAS_PER_ARTICLE,
):
    """
    加速版切段：
    - 输入建议已限流（MAX_TEXT_CHARS_PER_ARTICLE），此处再清洗
    - 句尾标点切分；达到 max_paras 立即停止
    - 末段过短并入前一段
    """
    if not text:
        return []

    text = clean_text(text)
    if len(text) > MAX_TEXT_CHARS_PER_ARTICLE:
        text = text[:MAX_TEXT_CHARS_PER_ARTICLE]

    sentences = SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    chunks = []
    cur = ""

    for sent in sentences:
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = f"{cur} {sent}"
        elif len(cur) >= min_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}"

        if len(chunks) >= max_paras:
            break

    if cur and (len(chunks) < max_paras):
        chunks.append(cur)

    if len(chunks) >= 2 and len(chunks[-1]) < min_chars * 0.5:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()

    if len(chunks) > max_paras:
        chunks = chunks[:max_paras]

    return chunks


def estimate_height(text_len: int, box_width: float) -> int:
    """
    仅基于字符数估高（避免反复计算 len(text)）。
    """
    if text_len <= 0:
        line_count = MIN_LINES
    else:
        chars_per_line = max(1.0, box_width / CHAR_WIDTH)
        line_count = max(MIN_LINES, math.ceil(text_len / chars_per_line))
    height = PADDING_TOP_BOTTOM + line_count * LINE_HEIGHT
    return snap(height)


def distribute_to_cols_lpt_by_lengths(lengths, num_cols: int, col_width: float):
    """
    列内平衡分配（LPT）
    lengths: [(idx, len_text), ...] 按估高降序分配
    """
    heights = [(i, estimate_height(L, col_width)) for i, L in lengths]
    heights.sort(key=lambda x: x[1], reverse=True)

    cols = [{"h_sum": 0, "items": []} for _ in range(num_cols)]
    for idx, h in heights:
        j = min(range(num_cols), key=lambda c: cols[c]["h_sum"])
        cols[j]["items"].append((idx, h))
        cols[j]["h_sum"] += h + INNER_BLOCK_GAP
    for c in cols:
        if c["items"]:
            c["h_sum"] -= INNER_BLOCK_GAP
        c["h_sum"] = max(0, snap(c["h_sum"]))
    return cols


def decide_cols(num_chunks: int, title_h: int, col_width: float, lengths):
    # 上限由段落数与全局 MAX_COLS_PER_ARTICLE 决定
    max_cols_by_paras = max(1, math.ceil(num_chunks / MAX_PARAS_PER_COL))
    max_cols = min(MAX_COLS_PER_ARTICLE, max_cols_by_paras)

    target_mid = 0.5 * (TARGET_ASPECT_MIN + TARGET_ASPECT_MAX)
    best = None  # (score, cols)

    for cols in range(1, max_cols + 1):
        cols_obj = distribute_to_cols_lpt_by_lengths(lengths, cols, col_width)
        body_h = max((c["h_sum"] for c in cols_obj), default=0)
        total_h = title_h + INNER_BLOCK_GAP + body_h
        total_w = cols * col_width + (cols - 1) * INNER_BLOCK_GAP
        aspect = total_w / max(1, total_h)

        # 先对越界进行强惩罚，再靠近中点
        penalty = 0.0
        if aspect < TARGET_ASPECT_MIN:
            penalty = TARGET_ASPECT_MIN - aspect
        elif aspect > TARGET_ASPECT_MAX:
            penalty = aspect - TARGET_ASPECT_MAX

        # 组合成一个简单分数：越小越好
        score = penalty * 100.0 + abs(aspect - target_mid)
        # 轻微偏好更矮的卡片，避免极端拉高
        score += total_h / 10000.0

        if best is None or score < best[0]:
            best = (score, cols)

    return best[1]



def build_article_card(i, art):
    # ===== 增量拼接 + 限流：避免超长文章在 join 时吃爆内存/时间 =====
    buf = []
    remain = MAX_TEXT_CHARS_PER_ARTICLE + 2000  # 稍加富余，减少句中截断概率
    for part in art["parts"]:
        t = (part.get("text") or "").strip()
        if not t:
            continue
        if len(t) <= remain:
            buf.append(t)
            remain -= len(t)
            if remain <= 0:
                break
        else:
            buf.append(t[:remain])
            break
    full_text = " ".join(buf)

    chunks = split_into_chunks_fast(full_text)
    if not chunks:
        return None

    col_width = float(TEXTBOX_WIDTH)

    # 先用估计列数，估标题高
    tentative_cols = min(MAX_COLS_PER_ARTICLE, max(1, math.ceil(len(chunks) / MAX_PARAS_PER_COL)))
    tentative_width = tentative_cols * col_width + (tentative_cols - 1) * INNER_BLOCK_GAP
    title_h = estimate_height(len(art["title"]), tentative_width)

    lengths = [(i, len(ch)) for i, ch in enumerate(chunks)]
    cols = decide_cols(len(chunks), title_h, col_width, lengths)
    article_width = cols * col_width + (cols - 1) * INNER_BLOCK_GAP

    # 最终标题高
    title_h = estimate_height(len(art["title"]), article_width)

    # 列内平衡
    cols_obj = distribute_to_cols_lpt_by_lengths(lengths, cols, col_width)

    rel_blocks = []
    y_cursor = 0

    # 标题
    rel_blocks.append({
        "chunk_index": -1,
        "text": art["title"],
        "h": title_h,
        "w": article_width,
        "x_rel": 0.0,
        "y_rel": float(y_cursor),
        "is_title": True,
    })
    y_cursor += title_h + INNER_BLOCK_GAP

    # 正文列
    max_body_h = 0
    for c in range(cols):
        x_rel = c * (col_width + INNER_BLOCK_GAP)
        y_rel = y_cursor
        for idx, h in cols_obj[c]["items"]:
            rel_blocks.append({
                "chunk_index": idx,
                "text": chunks[idx],
                "h": h,
                "w": col_width,
                "x_rel": float(x_rel),
                "y_rel": float(y_rel),
                "is_title": False,
            })
            y_rel += h + INNER_BLOCK_GAP
        max_body_h = max(max_body_h, y_rel - INNER_BLOCK_GAP)

    total_height = snap(max_body_h)
    return {
        "article_index": i,
        "title": art["title"],
        "base_title": art["base_title"],
        "blocks": rel_blocks,
        "total_height": total_height,
        "total_width": snap(article_width),
        "para_count": len(chunks),
    }


def shelf_pack_and_emit(cards_batch, y_start):
    """
    对“一个批次”的卡片做搁板式排布并返回布局 items 与新的 y 起点。
    批与批之间不做全局排序/合并，直接在上一批的下方继续。
    """
    if not cards_batch:
        return [], y_start

    # 本批内宽->窄排序
    cards = sorted(cards_batch, key=lambda c: c["total_width"], reverse=True)

    rows = []
    cur_row = []
    cur_w = 0
    for card in cards:
        w = card["total_width"]
        need = (COLUMN_GAP_X if cur_row else 0) + w
        if cur_w + need <= MAX_ROW_WIDTH:
            cur_row.append(card)
            cur_w += need
        else:
            if cur_row:
                rows.append(cur_row)
            cur_row = [card]
            cur_w = w
    if cur_row:
        rows.append(cur_row)

    items = []
    cur_y = y_start + ROW_GAP_Y
    for row in rows:
        row_height = max((snap(c["total_height"]) for c in row), default=0)
        x_cursor = COLUMN_GAP_X
        for art in row:
            base_x = snap(x_cursor)
            base_y = snap(cur_y)
            for b in art["blocks"]:
                items.append({
                    "article_index": art["article_index"],
                    "article_title": art["title"],
                    "base_title": art["base_title"],
                    "chunk_index": b["chunk_index"],
                    "text": b["text"],
                    "x": snap(base_x + b["x_rel"]),
                    "y": snap(base_y + b["y_rel"]),
                    "w": snap(b["w"]),
                    "h": snap(b["h"]),
                    "is_title": b["is_title"],
                })
            x_cursor += art["total_width"] + COLUMN_GAP_X
        cur_y += row_height + ROW_GAP_Y

    return items, cur_y


def to_stroke(item, gid: int):
    is_title = bool(item.get("is_title", False))
    if is_title:
        style_size = "l"; font_size = 36; font_weight = "700"; grow_dir = "right"
    else:
        style_size = "m"; font_size = 18; font_weight = "400"; grow_dir = "down"

    return {
        "id": f"wh_{item['article_index']}_{item['chunk_index']}_{gid}",
        "tool": "text",
        "points": [[float(item["x"]), float(item["y"])],
                   [float(item["x"])+float(item["w"]), float(item["y"])+float(item["h"])]],
        "style": {"size": style_size, "color": "black", "opacity": 1.0},
        "meta": {
            "author": "wikihow-dataset",
            "source": DATASET_NAME,
            "articleTitle": item["article_title"],
            "baseTitle": item["base_title"],
            "articleIndex": int(item["article_index"]),
            "chunkIndex": int(item["chunk_index"]),
            "isTitle": is_title,
            "text": item["text"],
            "summary": item["text"][:50].strip(),
            "fontFamily": "sans-serif",
            "fontWeight": font_weight,
            "fontSize": font_size,
            "growDir": grow_dir,
            "charLen": len(item["text"]),
            "approxHeight": float(item["h"]),
            "baseWidth": float(item["w"]),
            "baseHeight": float(item["h"]),
        },
    }


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser("Build LineArt JSON from WikiHow cleaned dataset (streaming-like & capped).")
    p.add_argument("--max-merged", type=int, default=int(os.environ.get("WIKIHOW_MAX_MERGED", 2000)),
                   help="最多处理多少篇合并后的文章（默认 2000）")
    p.add_argument("--sample", type=int, default=int(os.environ.get("WIKIHOW_SAMPLE", 0)),
                   help="随机抽样 N 篇（与 --max-merged 互斥，优先取 sample）")
    p.add_argument("--seed", type=int, default=int(os.environ.get("WIKIHOW_SEED", 42)),
                   help="随机抽样随机种子")
    p.add_argument("--max-files", type=int, default=int(os.environ.get("WIKIHOW_MAX_FILES", 0)),
                   help="最多写出多少个 JSON 分片（0 表示不限制）")
    p.add_argument("--max-chars-per-file", type=int,
                   default=int(os.environ.get("WIKIHOW_MAX_CHARS_PER_FILE", DEFAULT_MAX_CHARS_PER_FILE)),
                   help="单个 JSON 分片的近似字数上限")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("WIKIHOW_BATCH_SIZE", 150)),
                   help="构卡+上架的批大小（默认 150）")
    p.add_argument("--progress-every", type=int, default=25,
                   help="每处理多少篇打印一次小进度")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading dataset {DATASET_NAME} ({SPLIT}) ...", flush=True)
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # Pass 1：仅索引（轻量）
    print("Indexing articles (pass 1, no texts) ...", flush=True)
    groups_meta, _ = index_articles(ds)
    total_groups = len(groups_meta)
    print(f"Total merged groups (by normalized title): {total_groups}", flush=True)

    # 选择子集：优先 sample
    if args.sample and args.sample > 0:
        n = min(args.sample, total_groups)
        random.seed(args.seed)
        selected_bases = random.sample(list(groups_meta.keys()), n)
        print(f"[Subset] Using RANDOM SAMPLE of merged articles: {n} (seed={args.seed})", flush=True)
    else:
        n = min(args.max_merged, total_groups)
        selected_bases = [b for b, _ in sorted(
            ((b, meta["first_idx"]) for b, meta in groups_meta.items()),
            key=lambda x: x[1]
        )[:n]]
        print(f"[Subset] Using HEAD of merged articles: {len(selected_bases)}", flush=True)

    # Pass 2：只为选中的分组构建正文
    print("Materializing selected articles (pass 2, texts only for selected) ...", flush=True)
    merged = materialize_selected(ds, set(selected_bases))

    # 写盘控制
    file_index = 0
    cur_strokes = []
    cur_chars = 0
    gid = 0
    max_files = args.max_files
    max_chars_per_file = args.max_chars_per_file

    def flush_file():
        nonlocal file_index, cur_strokes, cur_chars
        if not cur_strokes:
            return True
        payload = {"version": 1, "intent": "import", "strokes": cur_strokes}
        out_path = OUT_DIR / f"wikihow_lineart_{file_index:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(cur_strokes)} strokes to {out_path}", flush=True)
        file_index += 1
        cur_strokes = []
        cur_chars = 0
        if max_files and file_index >= max_files:
            print(f"Reached file limit ({max_files}). Stop.", flush=True)
            return False
        return True

    y_cursor_global = 0
    batch_cards = []
    batch_count = 0
    t_batch_start = time.time()

    print("Building layout in STREAMING batches ...", flush=True)
    total = len(merged)
    for i, art in enumerate(merged, 1):
        # 细粒度心跳
        if i % 5 == 1:
            print(f"[loop] at article {i}/{total}", flush=True)

        # 单篇计时
        t0 = time.time()
        try:
            card = build_article_card(i - 1, art)
        except Exception as e:
            print(f"[WARN] build_article_card failed at idx={i-1}: {e}", flush=True)
            card = None
        dt = time.time() - t0
        if dt > 1.5:
            print(f"[SLOW] build_article_card #{i} took {dt:.2f}s", flush=True)

        if card:
            batch_cards.append(card)

        # 小进度
        if args.progress_every and i % args.progress_every == 0:
            print(f"  processed articles: {i}/{total} (+{len(batch_cards)} in batch)", flush=True)

        # 到批大小就上架+写盘
        if len(batch_cards) >= args.batch_size:
            batch_count += 1
            # 批内上架
            items, y_cursor_global = shelf_pack_and_emit(batch_cards, y_cursor_global)
            # 转 strokes 并流式写盘
            for it in items:
                st = to_stroke(it, gid); gid += 1
                L = len(st["meta"]["text"])
                if cur_strokes and cur_chars + L > max_chars_per_file:
                    if not flush_file():
                        return
                cur_strokes.append(st)
                cur_chars += L
            print(f"  [Batch {batch_count}] cards={len(batch_cards)}, items={len(items)}, "
                  f"elapsed={time.time() - t_batch_start:.1f}s, y={y_cursor_global}", flush=True)
            batch_cards = []
            t_batch_start = time.time()

    # 最后一批
    if batch_cards:
        batch_count += 1
        items, y_cursor_global = shelf_pack_and_emit(batch_cards, y_cursor_global)
        for it in items:
            st = to_stroke(it, gid); gid += 1
            L = len(st["meta"]["text"])
            if cur_strokes and cur_chars + L > max_chars_per_file:
                if not flush_file():
                    return
            cur_strokes.append(st)
            cur_chars += L
        print(f"  [Batch {batch_count}] cards={len(batch_cards)}, items={len(items)}, final-batch", flush=True)

    # 收尾
    flush_file()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
