# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List


def build_vision_v2_step1(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Vision 2.0 - Step 1锛堝浘鍍忕悊瑙ｉ樁娈碉級
    - 鍙紶蹇収鍥惧儚锛堜笉浼?strokes 鐐瑰垪锛?
    - 璁╂ā鍨嬭緭鍑?JSON锛歿 analysis, instruction }
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
            "Return JSON with keys: analysis (~100 words/50瀛?, instruction (~50 words/30瀛?.",
            "instruction should be actionable for next drawing step (concise).",
        ],
    }
    return [
        {"role": "system", "content": "You are a precise line-art critic. Analyze the canvas image and propose the single best next stroke idea."},
        {"role": "user", "content": json.dumps(msg_user_payload, ensure_ascii=False)},
    ]


def build_full_with_instruction(req: Dict[str, Any], instruction_text: str) -> List[Dict[str, Any]]:
    """
    Full 娴佺▼锛坴1.1 鍗忚涓嶅彉锛夛紝浣嗗湪 system / user 涓敞鍏ラ澶栫殑鈥滃己鍖栨寚浠も€濄€?
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
        {"role": "system", "content": "You are a structured line-art assistant. Produce JSON {version, intent, strokes[]} with concise geometry (鈮en_scale points per stroke)."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
