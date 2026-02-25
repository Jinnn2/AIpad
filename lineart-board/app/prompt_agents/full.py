# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.schemas import SuggestRequest


def _join_prompt_lines(lines: List[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


FULL_SYSTEM_COMMON_PREFIX = [
    "Role: On-canvas note assistant. Return JSON ONLY that conforms to AIStrokePayload.",
    "If canvas is empty, return a text stroke: \"Draw or Type to activate AIPad\".",
    "In most cases, explain with TEXT, fix with EDIT, or add explanatory figures with several strokes.",
    "General:",
    " - Coordinates are ABSOLUTE canvas pixels.",
    " - You can DRAW, WRITE text, or EDIT existing text.",
    " - Use concise keypoints. Positioning matters most. Avoid text overlap.",
    " - If planner_next_step is present, treat it as prioritized guidance.",
    " - If block_outline is present, keep semantic continuity with its structure.",
    " - If prefer_explanatory_drawing=true, prefer concise explanatory diagrams/sketches (short labels allowed) instead of long prose-only additions.",
]

FULL_SYSTEM_MAINTAIN_LINES = [
    " - Auto Maintain is ON: you may manage semantic block placement when confident.",
    " - If a distinct new topic/subtopic starts, you may start a new semantic block.",
    " - If a current block is crowded/heavily overlapping, consider starting a new block for better organization.",
    " - Start a NEW semantic block by emitting a title/subtitle TEXT stroke with meta.graph.blockIntent='create' (optional meta.graph.blockLabel). Use sparingly (at most one per response).",
    " - If a stroke clearly belongs to an EXISTING block, you may set meta.graph.targetBlockId to an exact blockId from block_outline/context.",
    " - To create a new block and attach later strokes in the SAME response, share meta.graph.proposalKey across the anchor title/subtitle and later related strokes. Put the anchor first.",
]

FULL_SYSTEM_NO_MAINTAIN_LINES = [
    " - Auto Maintain is OFF: do not emit semantic graph control fields (meta.graph.blockIntent / targetBlockId / proposalKey).",
]

FULL_SYSTEM_DRAW_LINES = [
    "DRAW tools:",
    " - Prefer multiple clean strokes for meaningful diagrams rather than one dense stroke.",
    " - line: exactly 2 points [p0, pn].",
    " - poly: >=3 vertices; repeat first point to close.",
    " - ellipse: exactly 2 points as bounding-box diagonal.",
    " - pen: freeform keypoints (concise, not dense).",
]

FULL_SYSTEM_WRITE_COMMON_LINES = [
    "WRITE (tool='text'):",
    " - points = [[x,y],[x+w,y+h]] (top-left to bottom-right).",
    " - style.color must be from palette.",
    " - meta MUST include: text, summary(<=30), fontFamily, fontWeight, fontSize, growDir.",
    " - meta.role is recommended: one of {'body','subtitle','title'} for text hierarchy.",
    " - meta.text may use a markdown subset (headings/lists/bold/inline-code only); avoid tables, HTML, and fenced code blocks.",
    " - growDir must be one of {'right-down','down','right','up','left'}; default is 'right-down'.",
    " - For growDir='right-down': wrap text inside current box first, then auto-fit proportionally toward lower-right (shrink/expand as needed while keeping base width/height ratio).",
]

FULL_SYSTEM_WRITE_MAINTAIN_LINES = [
    " - Optional : meta.graph = {blockIntent:'create', blockLabel?:string} on a heading-like text to request new block creation.",
    " - Optional : any stroke may include meta.graph.targetBlockId (exact existing blockId) to attach to a known block when certain.",
    " - Optional : related strokes in the same response may share meta.graph.proposalKey with the anchor create-text.",
]

FULL_SYSTEM_EDIT_LINES = [
    "EDIT (tool='edit'):",
    " - Prefer concise targeted edits to existing strokes over broad replacements.",
    " - meta MUST include: targetId (MOST IMPORTANT), operation, text (updated content).",
    " - points optional; if provided, use target bbox [[x,y],[x+w,y+h]].",
]


def _build_full_system_prompt(*, auto_maintain_enabled: bool) -> str:
    lines: List[str] = []
    lines.extend(FULL_SYSTEM_COMMON_PREFIX)
    lines.extend(FULL_SYSTEM_MAINTAIN_LINES if auto_maintain_enabled else FULL_SYSTEM_NO_MAINTAIN_LINES)
    lines.extend(FULL_SYSTEM_DRAW_LINES)
    lines.extend(FULL_SYSTEM_WRITE_COMMON_LINES)
    if auto_maintain_enabled:
        lines.extend(FULL_SYSTEM_WRITE_MAINTAIN_LINES)
    lines.extend(FULL_SYSTEM_EDIT_LINES)
    return _join_prompt_lines(lines)


# NOTE: Keep {max_pts} placeholders literal for backward compatibility.
FULL_CONTRACT_COMMON_LINES = [
    "Return JSON with fields: version, intent, canvas(optional), replace(optional), strokes[].",
    "Rules:",
    " - version = 1 (integer)",
    " - intent in {'complete','hint','alt','write'}",
    " - Each stroke: {id, tool in {'pen','line','poly','ellipse','text','edit'}, points, style{size,color,opacity}, meta}",
    " - line: 2 points; poly: >=3 vertices (closed); ellipse: 2 bbox points; pen: keypoints <= {max_pts}",
    " - text: meta includes text/summary/fontFamily/fontWeight/fontSize/growDir and recommended role in {'body','subtitle','title'} (growDir one of {'right-down','down','right','up','left'}, default 'right-down'; for right-down, wrap in-box first then proportionally auto-fit to lower-right, including shrink/expand as needed). Text should be long enough to be useful. (~10*{max_pts} chars is a good target)",
    " - markdown in meta.text is allowed only for headings/lists/bold/inline-code; avoid tables/HTML/fenced code blocks",
    " - edit: meta includes targetId/operation/text",
    " - Try to match scale target {max_pts} using stroke length/count.",
    " - style.size in {'s','m','l','xl'}; opacity in [0,1]",
    " - Colors ONLY: black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow",
]

FULL_CONTRACT_MAINTAIN_LINES = [
    " - text.meta.graph may optionally include {blockIntent:'create', blockLabel?:string} on ONE heading-like anchor text to start a new semantic block",
    " - Any stroke.meta.graph may optionally include {targetBlockId:string} to attach it to an existing block (must match an exact blockId from context)",
    " - Related strokes in the same response may share meta.graph.proposalKey:string with the anchor create-text (anchor should appear first)",
]

FULL_CONTRACT_NO_MAINTAIN_LINES = [
    " - Do not emit meta.graph block control fields (blockIntent/targetBlockId/proposalKey)",
]


def _build_full_contract(*, auto_maintain_enabled: bool) -> str:
    lines = list(FULL_CONTRACT_COMMON_LINES)
    lines.extend(FULL_CONTRACT_MAINTAIN_LINES if auto_maintain_enabled else FULL_CONTRACT_NO_MAINTAIN_LINES)
    return _join_prompt_lines(lines)


# Backward-compatible exports (default = non-maintain version).
FULL_SYSTEM = _build_full_system_prompt(auto_maintain_enabled=False)
FULL_CONTRACT = _build_full_contract(auto_maintain_enabled=False)

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
                "text": "Key points of circuit analysis:\n1. Node-voltage method\n2. Superposition principle",
                "summary": "Key points of circuit analysis",
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


def _as_xy_points(points: object) -> List[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    out: List[tuple[float, float]] = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return out


def _dedupe_consecutive_points(points: List[tuple[int, int]]) -> List[tuple[int, int]]:
    if not points:
        return []
    out = [points[0]]
    for pt in points[1:]:
        if pt != out[-1]:
            out.append(pt)
    return out


def _is_closed_like(points: List[tuple[int, int]]) -> bool:
    if len(points) < 4:
        return False
    x0, y0 = points[0]
    x1, y1 = points[-1]
    return abs(x0 - x1) <= 2 and abs(y0 - y1) <= 2


def _point_area_score(prev_pt: tuple[int, int], cur_pt: tuple[int, int], next_pt: tuple[int, int]) -> float:
    # Triangle area proxy (twice area) is a cheap curvature/importance signal.
    return abs(
        (cur_pt[0] - prev_pt[0]) * (next_pt[1] - prev_pt[1])
        - (cur_pt[1] - prev_pt[1]) * (next_pt[0] - prev_pt[0])
    )


def _compress_pen_points(points: object, *, cap: int = 10) -> List[List[int]]:
    raw = _as_xy_points(points)
    if not raw:
        return []
    quantized = _dedupe_consecutive_points([(int(round(x)), int(round(y))) for x, y in raw])
    n = len(quantized)
    if n <= cap:
        return [[x, y] for x, y in quantized]

    key_idx = {0, n - 1}

    xs = [p[0] for p in quantized]
    ys = [p[1] for p in quantized]
    key_idx.update(
        {
            xs.index(min(xs)),
            xs.index(max(xs)),
            ys.index(min(ys)),
            ys.index(max(ys)),
        }
    )

    if _is_closed_like(quantized):
        for k in (1, 2, 3):
            key_idx.add(round((n - 1) * k / 4))

    if len(key_idx) < cap and n >= 3:
        ranked = sorted(
            (
                (_point_area_score(quantized[i - 1], quantized[i], quantized[i + 1]), i)
                for i in range(1, n - 1)
                if i not in key_idx
            ),
            reverse=True,
        )
        for _score, idx in ranked:
            key_idx.add(idx)
            if len(key_idx) >= cap:
                break

    if len(key_idx) < cap:
        for k in range(cap):
            key_idx.add(round((n - 1) * k / max(1, cap - 1)))
            if len(key_idx) >= cap:
                break

    if len(key_idx) < cap:
        for idx in range(n):
            key_idx.add(idx)
            if len(key_idx) >= cap:
                break

    ordered = sorted(key_idx)
    if len(ordered) > cap:
        must_keep = {0, n - 1}
        keep = set(i for i in ordered if i in must_keep)
        for idx in ordered:
            if idx in keep:
                continue
            keep.add(idx)
            if len(keep) >= cap:
                break
        ordered = sorted(keep)
        if len(ordered) > cap:
            ordered = ordered[:cap]
            if 0 not in ordered:
                ordered[0] = 0
            if (n - 1) not in ordered:
                ordered[-1] = n - 1
            ordered = sorted(set(ordered))
            # Fill again if dedupe made it short.
            if len(ordered) < cap:
                for idx in range(n):
                    if idx in ordered:
                        continue
                    ordered.append(idx)
                    if len(ordered) >= cap:
                        break
                ordered = sorted(ordered)

    return [[quantized[i][0], quantized[i][1]] for i in ordered]


def _compress_full_context_for_prompt(ctx: Dict[str, Any], *, pen_cap: int) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return ctx
    strokes = ctx.get("strokes")
    if not isinstance(strokes, list):
        return ctx

    next_ctx = dict(ctx)
    next_strokes: List[Any] = []
    for stroke in strokes:
        if not isinstance(stroke, dict):
            next_strokes.append(stroke)
            continue
        tool = str(stroke.get("tool") or "").lower()
        if tool != "pen":
            next_strokes.append(stroke)
            continue
        compact_points = _compress_pen_points(stroke.get("points"), cap=pen_cap)
        if not compact_points:
            next_strokes.append(stroke)
            continue
        new_stroke = dict(stroke)
        new_stroke["points"] = compact_points
        meta = stroke.get("meta")
        if isinstance(meta, dict):
            meta2 = dict(meta)
            meta2["promptSimplified"] = True
            new_stroke["meta"] = meta2
        next_strokes.append(new_stroke)
    next_ctx["strokes"] = next_strokes
    return next_ctx


def _build_full_user_content(req: SuggestRequest, include_sample: bool = True) -> Dict[str, Any]:
    N = 200
    ctx = req.context.model_dump()
    if isinstance(ctx.get("strokes"), list) and len(ctx["strokes"]) > N:
        ctx["strokes"] = ctx["strokes"][-N:]

    max_pts = int(req.gen_scale) if (hasattr(req, "gen_scale") and req.gen_scale) else 16
    max_pts = max(4, min(64, max_pts))
    pen_context_cap = max(8, min(10, max_pts))
    ctx = _compress_full_context_for_prompt(ctx, pen_cap=pen_context_cap)
    auto_maintain_enabled = bool(getattr(req, "auto_maintain_enabled", False))

    user_content: Dict[str, Any] = {
        "mode": "work assistant",
        "goal": req.hint or "Complete the user's intent with appropriate strokes or text.",
        "context": ctx,
        "output_contract": _build_full_contract(auto_maintain_enabled=auto_maintain_enabled),
        "notes": FULL_NOTES,
        "Setting": {"Scale": max_pts},
    }
    planner_next_step = (getattr(req, "planner_next_step", None) or "").strip()
    if planner_next_step:
        user_content["planner_next_step"] = planner_next_step
    block_outline = getattr(req, "block_outline", None)
    if isinstance(block_outline, list) and block_outline:
        user_content["block_outline"] = block_outline[:8]
    prefer_explanatory_drawing = getattr(req, "prefer_explanatory_drawing", None)
    if isinstance(prefer_explanatory_drawing, bool):
        user_content["prefer_explanatory_drawing"] = prefer_explanatory_drawing
    user_content["auto_maintain_enabled"] = auto_maintain_enabled
    if include_sample:
        user_content["samples"] = FULL_SAMPLES
    return user_content


def build_messages_full(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    auto_maintain_enabled = bool(getattr(req, "auto_maintain_enabled", False))
    user_content = _build_full_user_content(req, include_sample=include_sample)
    return [
        {"role": "system", "content": _build_full_system_prompt(auto_maintain_enabled=auto_maintain_enabled)},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False, separators=(",", ":"), default=str)},
    ]


def build_messages(req: SuggestRequest, include_sample: bool = True) -> List[Dict[str, Any]]:
    return build_messages_full(req, include_sample=include_sample)
