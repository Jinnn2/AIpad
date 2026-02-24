from __future__ import annotations

import json
import os
from typing import Dict, List

from semantic_graph import PlanBackend

from app.llm_client import call_chat_completions


DEFAULT_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
DEFAULT_PLAN_MODEL = os.getenv("GRAPH_PLAN_MODEL") or DEFAULT_LLM_MODEL


class LLMPlanBackend(PlanBackend):
    def __init__(self, model: str = DEFAULT_PLAN_MODEL, max_tokens: int = 1200) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: List[Dict[str, str]]) -> str:
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        if isinstance(parsed, str):
            return parsed
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps({"action": "NOOP", "targetBlockIds": [], "comment": "invalid planner output"}, ensure_ascii=False)
