# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import SuggestRequest


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
    " - meta.role is recommended: one of {'body','subtitle','title'} for text hierarchy.\n"
    " - growDir must be one of {'right-down','down','right','up','left'}; default is 'right-down'.\n"
    " - For growDir='right-down': wrap text inside current box first, then auto-fit proportionally toward lower-right (shrink/expand as needed while keeping base width/height ratio).\n"
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
" - text: meta includes text/summary/fontFamily/fontWeight/fontSize/growDir and recommended role in {'body','subtitle','title'} (growDir one of {'right-down','down','right','up','left'}, default 'right-down'; for right-down, wrap in-box first then proportionally auto-fit to lower-right, including shrink/expand as needed). Text should be long enough to be useful. (10*{max_pts} is the best)\n"
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
                "text": "鐢佃矾鍒嗘瀽娉ㄦ剰锛歕n1. 鑺傜偣鐢典綅娉昞n2. 鍙犲姞鍘熺悊",
                "summary": "鐢佃矾鍒嗘瀽瑕佺偣",
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "fontSize": 16,
                "role": "subtitle",
                "growDir": "right-down",
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
                "role": "subtitle",
                "growDir": "right-down",
            },
        }
    ],
}

FULL_SAMPLES = [SAMPLE_STROKES, SAMPLE_ONE_POLY, SAMPLE_TEXTBOX]

FULL_NOTES = (
    "Return JSON strokes only. If asked for COMPLETION, return ONE stroke. "
    "Use concise keypoints; scale target = {max_pts}."
)


def _build_full_user_content(req: SuggestRequest, include_sample: bool = True) -> Dict[str, Any]:
    N = 200
    ctx = req.context.model_dump()
    if isinstance(ctx.get("strokes"), list) and len(ctx["strokes"]) > N:
        ctx["strokes"] = ctx["strokes"][-N:]

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
    user_content = _build_full_user_content(req, include_sample=include_sample)
    return [
        {"role": "system", "content": FULL_SYSTEM},
        {"role": "user", "content": f"{user_content}"},
    ]


def build_messages(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    return build_messages_full(req, include_sample=include_sample)
