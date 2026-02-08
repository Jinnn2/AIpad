# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List
import json

from app.schemas import SuggestRequest

# =============================================================================
# Prompt Templates (FULL / LIGHT / VISION)
# =============================================================================

# ---- FULL mode ----
FULL_SYSTEM = (
    "Role: You are an on-canvas work assistant that draw strokes or generate texts based on the HINT and Existing content.\n"
    "If the canvas is empty, return text by default: \"Draw or Type to activate AIPad\"\n"
    "Behavior rules:\n"
    " - Return JSON objects that strictly conforms to AIStrokePayload.\n"
    " - Coordinates are ABSOLUTE canvas space (pixels).\n"
    " - You can DRAW, WRITE, or EDIT existing text.\n"
    " - DRAW: use 'pen','line','poly','ellipse' tools to draw shapes/lines.\n"
    " - If you want to add a straight line, use tool='line' and provide exactly 2 points [p0, pn].\n"
    " - If the intent is a CLOSED polygonal shape (rectangle, triangle, loop), use tool='poly' with >=3 vertices.\n"
    " - If you want an ellipse, use tool='ellipse' and provide exactly 2 points [p0, pn] as the bounding-box diagonal.\n"
    " - For freeform curves, use tool='pen' with multiple points.\n"
    " - For pen: provide as many points as possible, up to the limit given.\n"
    "   For poly: points are vertices in order; the last point MUST repeat the first to explicitly close the loop.\n"
    " - Before generating, carefully ANALYZE whether you use a LINE, POLY, ELLIPSE or PEN.\n"
    " - For curves, prefer concise key points; do NOT densely sample every pixel.\n"
    " - The Length Baseline is 200px each segment.\n"
    " - WRITE: use tool='text' to ADD or EDIT text.\n"
    " - If recent strokes are mostly text, you may choose to EDIT an existing text stroke to refine it, or add a new one.\n"
    " - To ADD new text, use tool='text'. To EDIT existing text, use tool='edit' and specify targetId in meta.\n"
    "* points = [[x,y],[x+w,y+h]] where [x,y] is top-left corner, [x+w,y+h] is bottom-right corner.\n"
    "* style.color is the text color (must be from the palette).\n"
    "* meta MUST include:\n"
    "    \"text\": full multiline content,\n"
    "    \"summary\": short summary (<=30 chars),\n"
    "    \"fontFamily\": e.g. \"sans-serif\",\n"
    "    \"fontWeight\": e.g. \"400\" or \"bold\",\n"
    "    \"fontSize\": font size in px,\n"
    "    \"growDir\": one of {\"down\",\"right\",\"up\",\"left\"} (default \"down\").\n"
    "- EDIT text boxes using tool='edit' when you need to modify a previous text stroke.\n"
    "When receiving a user request containing the word \"organize\", \"整理\", (or equivalent), try to structurize the content:"
        "1: Read the paragraph and identify structural elements such as titles, subtitles or sections."
        "2: For each element, create a new text stroke with its content."
        "3: Edit the original paragraph to clean up."
        "Caution: Do NOT change any original expressions"
    "* meta MUST include: targetId (the existing stroke id), operation (<=60 chars describing the intent), content (the rewritten preview text). Optionally include updated text/font metadata.\n"
    "* If you supply points for edit, still use [[x,y],[x+w,y+h]] covering the target area."
)

# NOTE: Keep {max_pts} placeholders literal for backward compatibility.
FULL_CONTRACT = (
    "Return fields: version, intent, canvas(optional), replace(optional), strokes[].\n"
    "Constraints:\n"
    " - version = 1 (integer)\n"
    " - intent ∈{'complete','hint','alt','write'}; prefer 'complete' and 'write'.\n"
    " - You should combine multiple strokes to reach the scale.\n"
    " - "
    " - number of strokes: it should MATCH the scale = {max_pts}. If you use little points in each stroke, then increase the number of strokes\n"
    " - Each stroke: { id:string, tool:string in {'pen','line','poly','ellipse','text','edit'}, points:[[x,y,(t?),(pressure?)]...], style{size,color,opacity}, meta }\n"
    " - For 'line': exactly two points [p0, pn].\n"
    " - For 'poly': provide >=3 vertices in order; last MAY equal first to denote closure.\n"
    " - For 'ellipse': exactly two points [p0, pn] as the bounding-box diagonal.\n"
    " - For 'pen': multiple keypoints, prefer concise points up to {max_pts}.\n"
    " - For 'text': points = [[x,y],[x+w,y+h]]; style.color from palette; meta includes text, summary, fontFamily, fontWeight, fontSize, growDir.\n"
    " - For 'edit': meta includes targetId, operation, content (preview text). Points optional but recommended to reuse the target bounding box.\n"
    " - When it is not a line, try to use as much points as limited: {max_pts} \n"
    " - Try to use multiple styles and colors if they MAKE SENCE.\n"
    " - Use reasonable style: size in {'s','m','l','xl'}, opacity in [0,1]\n"
    " - MUST Use colors in palette ONLY: black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow\n"
)

SAMPLE_STROKES = {
    "version": 1,
    "intent": "complete",
    "strokes": [
        {
            "id": "ai_next_001",
            "tool": "pen",
            "points": [[320, 180], [360, 190], [400, 220], [420, 250], [480, 280]],
            "style": {"size": "m", "color": "red", "opacity": 0.9},
            "meta": {"author": "ai", "desc": "curve segment"},
        },
        {
            "id": "ai_next_002",
            "tool": "line",
            "points": [[500, 200], [600, 250]],
            "style": {"size": "l", "color": "blue", "opacity": 0.8},
            "meta": {"author": "ai", "desc": "straight line"},
        }
    ],
}

SAMPLE_ONE_POLY = {
    "version": 1,
    "intent": "complete",
    "strokes": [
        {
            "id": "ai_poly_001",
            "tool": "poly",
            "points": [[420, 200], [520, 200], [470, 280], [420, 200]],
            "style": {"size": "m", "color": "orange", "opacity": 0.8},
            "meta": {"author": "ai", "desc": "closed triangle"},
        }
    ],
}

SAMPLE_TEXTBOX = {
    "version": 1,
    "intent": "write",
    "strokes": [
        {
            "id": "ai_text_001",
            "tool": "text",
            "points": [[100, 120], [260, 200]],
            "style": {"size": "m", "color": "black", "opacity": 1.0},
            "meta": {
                "text": "电路分析注意：\n1. 节点电位法\n2. 叠加原理",
                "summary": "电路分析要点",
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "fontSize": 16,
                "growDir": "down",
            },
        },
        {
            "id": "ai_text_002",
            "tool": "edit",
            "points": [[100, 120], [260, 200]],
            "style": {"size": "m", "color": "black", "opacity": 1.0},
            "meta": {
                "targetId": "ai_text_001",
                "operation": "refine summary",
                "content": "电路分析要点有以下几点：\n1. 节点电位法。节点电位法是一个很好的方法......\n2. 叠加原理......\n（更新后内容）",
                # Optional: also update text/font if needed
                # "text": "电路分析注意：\n1. 节点电位法\n2. 叠加原理\n（更新后内容）",
                # "fontSize": 18,
            },
        }
    ],
}

FULL_SAMPLES = [SAMPLE_STROKES, SAMPLE_ONE_POLY, SAMPLE_TEXTBOX]

FULL_NOTES = (
    "Return strokes. If mentioned COMPLETION, just return ONE stroke. JSON object only. "
    "The more {max_pts} given, the longer the stroke you should draw. Baseline is 16 points, refering to about 200px length. "
    "Prefer concise keypoints over dense samples."
)

# ---- LIGHT mode ----
LIGHT_SYSTEM = (
    "Role: On-canvas assistant for NEXT-STROKE prediction.\n"
    "Rules:\n"
    " - Output JSON ONLY and MUST conform to AIStrokePayload v1.1.\n"
    " - ABSOLUTE pixel coordinates.\n"
    " - STRICTLY ONE stroke in 'strokes' (exactly one item).\n"
    " - Tool selection: 'line' (2 pts), 'poly' (>=3 vertices), 'ellipse' (2 opposite corners), 'pen' (freeform keypoints).\n"
    " - Prefer concise keypoints; do not densify samples.\n"
    " - No markdown / no prose / no comments.\n"
)

LIGHT_CONTRACT = (
    "Return fields: version=1, intent∈{'complete','hint','alt'} (prefer 'complete'), strokes[1].\n"
    "Stroke shape:\n"
    " - line: exactly 2 points [[x0,y0],[x1,y1]].\n"
    " - poly: >=3 vertices in order; if closed, last MAY equal first.\n"
    " - ellipse: exactly 2 points as bounding-box diagonal.\n"
    " - pen: multiple keypoints, up to {max_pts}.\n"
    "Style:\n"
    " - size∈{'s','m','l','xl'}; color∈{black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow}; opacity∈[0,1].\n"
)

LIGHT_NOTES = (
    "Output JSON only. STRICTLY one stroke. Prefer keypoints under {max_pts}. "
    "If the shape is obviously straight, use 'line'."
)

# ---- VISION mode ----
# Vision mode reuses FULL system + user content, then appends a final user message
# that includes the canvas snapshot via _image_data.
VISION_SYSTEM = (
    "Role: On-canvas assistant with visual context. "
    "You are given a snapshot of the canvas; infer user's likely next stroke(s) from the scene."
)


# =============================================================================
# FULL mode builder
# =============================================================================

def _build_full_user_content(req: SuggestRequest, include_sample: bool = True) -> Dict[str, Any]:
    # Trim to last N strokes to keep prompt compact.
    N = 200
    ctx = req.context.model_dump()
    if isinstance(ctx.get("strokes"), list) and len(ctx["strokes"]) > N:
        ctx["strokes"] = ctx["strokes"][-N:]

    # Scale (point count) guidance: default 16
    max_pts = int(req.gen_scale) if (hasattr(req, "gen_scale") and req.gen_scale) else 16
    max_pts = max(4, min(64, max_pts))

    user_content: Dict[str, Any] = {
        "mode": "work assistant",
        "goal": req.hint or "Complete the user's intent with appropriate strokes or text.",
        "context": ctx,
        "output_contract": FULL_CONTRACT,
        "notes": FULL_NOTES,
        "Setting": {"Scale": max_pts},
    }
    if include_sample:
        user_content["samples"] = FULL_SAMPLES
    return user_content


def build_messages_full(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    """
    FULL mode (multi-stroke + text understanding):
    - system: FULL_SYSTEM
    - user: goal + (trimmed) context + contract + samples
    """
    user_content = _build_full_user_content(req, include_sample=include_sample)
    return [
        {"role": "system", "content": FULL_SYSTEM},
        {"role": "user", "content": f"{user_content}"},
    ]


def build_messages(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    """Backward-compatible alias for FULL mode."""
    return build_messages_full(req, include_sample=include_sample)


# =============================================================================
# LIGHT mode helpers
# =============================================================================

def _downsample_polyline(points, max_pts=12):
    """
    线性等距抽样（含两端点），仅保留 [x, y]，去除 t/pressure 等冗余。
    """
    if not isinstance(points, list) or len(points) <= max_pts:
        return [[float(p[0]), float(p[1])] for p in points]
    import math

    pts = [[float(p[0]), float(p[1])] for p in points]
    seg = []
    total = 0.0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg.append(d)
        total += d
    if total <= 1e-9:
        return [pts[0], pts[-1]]
    out = [pts[0]]
    steps = max_pts - 1
    acc = 0.0
    j = 0
    for s in range(1, steps):
        target = total * s / steps
        while j < len(seg) and acc + seg[j] < target:
            acc += seg[j]
            j += 1
        if j >= len(seg):
            out.append(pts[-1])
            break
        t = (target - acc) / (seg[j] or 1e-9)
        x = pts[j][0] + t * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
        out.append([x, y])
    out.append(pts[-1])
    return out


def _compress_context(ctx: dict, keep_last=60, max_pts=12, drop_ai=True) -> dict:
    """
    将 SuggestContext 压缩为极简格式，仅包含必要的人类笔画：
      - 只保留最近 keep_last 条
      - 每条最多 max_pts 个点（[x,y]）
      - 可选：丢弃历史 AI 笔画（减少“自我回声”）
      - 丢弃 style/meta/canvas 等无关字段
    输出结构（供提示词阅读，不要求符合 AIStrokePayload）：
      {
        "H": [ [tool:str, [[x,y],...]], ... ],
        "C": [w, h]  # 可选：画布尺寸（如存在）
      }
    """
    out = {"H": []}
    if not isinstance(ctx, dict):
        return out

    strokes = ctx.get("strokes") or []
    if drop_ai:
        strokes = [
            s for s in strokes
            if not (isinstance(s, dict) and (s.get("meta") or {}).get("author") == "ai")
        ]

    if len(strokes) > keep_last:
        strokes = strokes[-keep_last:]

    for s in strokes:
        tool = str(s.get("tool") or "pen")
        pts = s.get("points") or []
        pts2 = _downsample_polyline(pts, max_pts=max_pts)
        out["H"].append([tool, pts2])

    canvas = ctx.get("canvas") or {}
    if (
        isinstance(canvas, dict)
        and "size" in canvas
        and isinstance(canvas["size"], (list, tuple))
        and len(canvas["size"]) >= 2
    ):
        out["C"] = [canvas["size"][0], canvas["size"][1]]

    return out


# =============================================================================
# LIGHT mode builder
# =============================================================================

def build_messages_light(req: SuggestRequest, include_sample: bool = False) -> List[Dict[str, Any]]:
    """
    LIGHT mode（仅一笔）：
      - 极限压缩输入（仅人类笔画）
      - 强约束 ONLY ONE stroke
      - 默认不带样例以节省 tokens
    """
    max_pts = int(getattr(req, "gen_scale", 12) or 12)
    max_pts = max(6, min(24, max_pts))

    ctx = req.context.model_dump()
    mini_ctx = _compress_context(ctx, keep_last=60, max_pts=max_pts, drop_ai=True)

    user_content: Dict[str, Any] = {
        "mode": "light-completion",
        "goal": (req.hint or "Predict the single next stroke (sketch or text) continuing user's intent."),
        "context_min": mini_ctx,
        "contract": LIGHT_CONTRACT.format(max_pts=max_pts),
        "notes": LIGHT_NOTES.format(max_pts=max_pts),
    }

    if include_sample:
        user_content["sample"] = {
            "version": 1,
            "intent": "complete",
            "strokes": [
                {
                    "id": "ai_next_light_001",
                    "tool": "line",
                    "points": [[320, 180], [460, 220]],
                    "style": {"size": "m", "color": "black", "opacity": 1.0},
                    "meta": {"author": "ai"},
                }
            ],
        }

    return [
        {"role": "system", "content": LIGHT_SYSTEM},
        {"role": "user", "content": f"{user_content}"},
    ]


# =============================================================================
# VISION mode builder
# =============================================================================

def _build_vision_note(req: SuggestRequest) -> Dict[str, Any]:
    return {
        "mode": "vision",
        "note": (
            "The following image is a snapshot of the current canvas. "
            "Keep ALL previous instructions/contracts/examples unchanged (same as FULL mode). "
            "Use the snapshot only as additional context."
        ),
        "_image_data": getattr(req, "image_data", None),
        "_image_mime": getattr(req, "image_mime", None) or "image/jpeg",
        "snapshot_size": getattr(req, "snapshot_size", None),
    }


def build_messages_vision(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    """
    VISION mode:
    - Uses FULL prompts for consistency
    - Appends an extra user message containing the canvas snapshot
    """
    msgs = build_messages_full(req, include_sample=include_sample)
    vision_note = _build_vision_note(req)
    msgs.append({"role": "user", "content": f"{vision_note}"})
    return msgs


# =============================================================================
# Vision 2.0 (two-stage)
# =============================================================================

def build_vision_v2_step1(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Vision 2.0 - Step 1（图像理解阶段）
    - 只传快照图像（不传 strokes 点列）
    - 让模型输出 JSON：{ analysis, instruction }
    """
    view = req["canvas"]["viewport"]
    msg_user_payload = {
        "task": "image_inspection_and_next_move",
        "canvas_viewport": view,
        "hint": req.get("hint") or "",
        "_image_data": req.get("image_data"),
        "_image_mime": req.get("image_mime") or "image/jpeg",
        "rules": [
            "Analysis Part:",
            "Focus on what is ALREADY on the canvas, and give a concise summary.",
            "Use concise language; avoid verbosity.",
            "Instruction Part:",
            "Give clear shape analysis and actionable next-step instruction.",
            "Shapes may include lines, pen(curves), poly, and ellipses.",
            "Examples: 'The most important action is to add a window of the house. You should draw a triangle'.",
            "Return JSON with keys: analysis (~100 words/50字), instruction (~50 words/30字).",
            "instruction should be actionable for next drawing step (concise).",
        ],
    }
    return [
        {"role": "system", "content": "You are a precise line-art critic. Analyze the canvas image and propose the single best next stroke idea."},
        {"role": "user", "content": json.dumps(msg_user_payload, ensure_ascii=False)},
    ]


def build_full_with_instruction(req: Dict[str, Any], instruction_text: str) -> List[Dict[str, Any]]:
    """
    Full 流程（v1.1 协议不变），但在 system / user 中注入额外的“强化指令”。
    """
    view = req["canvas"]["viewport"]
    context = req.get("context") or {}
    strokes = context.get("strokes") or []
    payload = {
        "task": "line_art_next_step",
        "canvas_viewport": view,
        "context_version": context.get("version", 1),
        "intent": context.get("intent", "complete"),
        "strokes": strokes,
        "gen_scale": req.get("gen_scale") or 16,
        "hint": req.get("hint") or "",
        "extra_instruction": instruction_text,
    }
    return [
        {"role": "system", "content": "You are a structured line-art assistant. Produce JSON {version, intent, strokes[]} with concise geometry (≤gen_scale points per stroke)."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


# =============================================================================
# Unified entry
# =============================================================================

def build_messages_by_mode(req: SuggestRequest, mode: str | None, include_sample: bool = True) -> List[Dict[str, Any]]:
    """
    后端模式分发：
      - full: build_messages_full
      - light: build_messages_light
      - vision: build_messages_vision
      - 默认：full
    """
    m = (mode or "").lower()
    if m == "light":
        return build_messages_light(req, include_sample=False)
    if m == "full":
        return build_messages_full(req, include_sample=include_sample)
    if m == "vision":
        return build_messages_vision(req, include_sample=include_sample)
    return build_messages_full(req, include_sample=include_sample)
