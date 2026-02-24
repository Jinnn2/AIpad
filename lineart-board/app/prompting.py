# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import SuggestRequest
from app.prompt_agents import (
    FULL_CONTRACT,
    FULL_NOTES,
    FULL_SAMPLES,
    FULL_SYSTEM,
    LIGHT_CONTRACT,
    LIGHT_NOTES,
    LIGHT_SYSTEM,
    VISION_SYSTEM,
    SAMPLE_ONE_POLY,
    SAMPLE_STROKES,
    SAMPLE_TEXTBOX,
    build_full_with_instruction,
    build_messages,
    build_messages_full,
    build_messages_light,
    build_messages_vision,
    build_vision_v2_step1,
)


def build_messages_by_mode(req: SuggestRequest, mode: str | None, include_sample: bool = True) -> List[Dict[str, Any]]:
    m = (mode or "").lower()
    if m == "light":
        return build_messages_light(req, include_sample=False)
    if m == "full":
        return build_messages_full(req, include_sample=include_sample)
    if m == "vision":
        return build_messages_vision(req, include_sample=include_sample)
    return build_messages_full(req, include_sample=include_sample)


__all__ = [
    "FULL_CONTRACT",
    "FULL_NOTES",
    "FULL_SAMPLES",
    "FULL_SYSTEM",
    "LIGHT_CONTRACT",
    "LIGHT_NOTES",
    "LIGHT_SYSTEM",
    "VISION_SYSTEM",
    "SAMPLE_ONE_POLY",
    "SAMPLE_STROKES",
    "SAMPLE_TEXTBOX",
    "build_messages",
    "build_messages_full",
    "build_messages_light",
    "build_messages_vision",
    "build_vision_v2_step1",
    "build_full_with_instruction",
    "build_messages_by_mode",
]
