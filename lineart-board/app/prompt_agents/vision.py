# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import SuggestRequest

from .full import build_messages_full


VISION_SYSTEM = (
    "Role: On-canvas assistant with visual context. "
    "You are given a snapshot of the canvas; infer user's likely next stroke(s) from the scene."
)


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
    msgs = build_messages_full(req, include_sample=include_sample)
    vision_note = _build_vision_note(req)
    msgs.append({"role": "user", "content": f"{vision_note}"})
    return msgs
