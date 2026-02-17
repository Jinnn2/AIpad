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
    "Role: On-canvas note assistant. Return JSON ONLY that conforms to AIStrokePayload.\n"
    "If canvas is empty, return a text stroke: \"Draw or Type to activate AIPad\".\n"
    "In most cases, you should give further explanation blocks in TEXT, or correct minor mistakes with EDIT, or draw a explanatory figure with several strokes. More output strokes are allowed.\n"
    "General:\n"
    " - Coordinates are ABSOLUTE canvas pixels.\n"
    " - You can DRAW, WRITE text, or EDIT existing text.\n"
    " - Use concise keypoints. The position of the keypoints is the most important. Texts should not overlap with each other."
    " - If planner_next_step is present in user content, treat it as prioritized guidance.\n"
    " - If block_outline is present, keep semantic continuity with its block structure.\n"
    "DRAW tools:\n"
    " - line: exactly 2 points [p0, pn].\n"
    " - poly: >=3 vertices; repeat first point to close.\n"
    " - ellipse: exactly 2 points as bounding-box diagonal.\n"
    " - pen: freeform keypoints (concise, not dense).\n"
    "WRITE (tool='text'):\n"
    " - points = [[x,y],[x+w,y+h]] (top-left to bottom-right).\n"
    " - style.color must be from palette.\n"
    " - meta MUST include: text, summary(<=30), fontFamily, fontWeight, fontSize, growDir.\n"
    "EDIT (tool='edit'):\n"
    " - meta MUST include: targetId(MOST IMPORTANT), operation(<=60 chars), text (updated content).\n"
    " - points optional; if provided, use target bbox [[x,y],[x+w,y+h]].\n"
)

# NOTE: Keep {max_pts} placeholders literal for backward compatibility.
FULL_CONTRACT = (
    "Return JSON with fields: version, intent, canvas(optional), replace(optional), strokes[].\n"
    "Rules:\n"
    " - version = 1 (integer)\n"
    " - intent in {'complete','hint','alt','write'}\n"
    " - Each stroke: {id, tool in {'pen','line','poly','ellipse','text','edit'}, points, style{size,color,opacity}, meta}\n"
    " - line: 2 points; poly: >=3 vertices (closed); ellipse: 2 bbox points; pen: keypoints <= {max_pts}\n"
" - text: meta includes text/summary/fontFamily/fontWeight/fontSize/growDir. Text should be long enough to be useful. (10*{max_pts} is the best)\n"
    " - edit: meta includes targetId/operation/text\n"
    " - Try to match scale target {max_pts} using stroke length/count.\n"
    " - style.size in {'s','m','l','xl'}; opacity in [0,1]\n"
    " - Colors ONLY: black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow\n"
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
            "id": "ai_edit_001",
            "tool": "edit",
            "points": [[100, 120], [320, 200]],
            "style": {"size": "m", "color": "blue", "opacity": 1.0},
            "meta": {
                "targetId": "ai_text_001",
                "operation": "refine summary",
                "text": "Circuit analysis key points:\n1. Node-voltage method\n2. Superposition principle",
                "summary": "Circuit analysis points",
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "fontSize": 16,
                "growDir": "down",
            },
        }
    ],
}

FULL_SAMPLES = [SAMPLE_STROKES, SAMPLE_ONE_POLY, SAMPLE_TEXTBOX]

FULL_NOTES = (
    "Return JSON strokes only. If asked for COMPLETION, return ONE stroke. "
    "Use concise keypoints; scale target = {max_pts}."
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
    planner_next_step = (getattr(req, "planner_next_step", None) or "").strip()
    if planner_next_step:
        user_content["planner_next_step"] = planner_next_step
    block_outline = getattr(req, "block_outline", None)
    if isinstance(block_outline, list) and block_outline:
        user_content["block_outline"] = block_outline[:8]
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
