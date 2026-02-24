from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from semantic_graph import BlockSummarizer, Fragment, FragmentType

from app.llm_client import call_chat_completions


DEFAULT_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
DEFAULT_SUMMARY_MODEL = os.getenv("GRAPH_SUMMARY_MODEL") or DEFAULT_LLM_MODEL


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip()


def _fragment_export(fragment: Fragment) -> Dict[str, object]:
    return {
        "id": fragment.fragment_id,
        "type": fragment.fragment_type.value,
        "text": fragment.text,
        "bbox": fragment.bbox,
        "payload": fragment.payload,
    }


class LLMBlockSummarizer(BlockSummarizer):
    def __init__(self, model: str = DEFAULT_SUMMARY_MODEL, max_tokens: int = 400, *, roster_limit: int = 12) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.roster_limit = roster_limit
        self._block_provider = lambda: []
        self._canvas_size: tuple[float, float] = (1920.0, 1080.0)

    def set_block_provider(self, provider) -> None:
        self._block_provider = provider

    def set_canvas_size(self, size: tuple[float, float]) -> None:
        self._canvas_size = size

    def propose_block(self, fragments: List[Fragment]) -> Dict[str, object]:
        roster = self._build_roster("")
        payload = {
            "task": "propose",
            "fragments": [_fragment_export(f) for f in fragments],
            "others": roster,
            "canvas": {"size": [self._canvas_size[0], self._canvas_size[1]]},
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-graph curator for a collaborative canvas.\n"
                    "You are the knowledge-graph curator for this collaborative canvas.\n"
                    "Create a new block label/summary from the provided fragments.\n"
                    "Also infer relationships from this new block to existing blocks in `others` when justified.\n"
                    "Always return a JSON object {\"label\": str, \"summary\": str, \"relationships\"?: [{\"type\": str, \"target\": str, \"score\": float?}]}.\n"
                    "Keep the label concise (<= 40 characters) and write a summary that captures the block's purpose for future context.\n"
                    "Only reference IDs that appear in `others`. Use relationship types such as refines, comment_on, subtopic or flow_next. Skip uncertain relationships.\n"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        label = _normalize_text(parsed.get("label") if isinstance(parsed, dict) else None) or "Untitled Block"
        summary = _normalize_text(parsed.get("summary") if isinstance(parsed, dict) else None) or label
        relationships: List[Dict[str, object]] = []
        if isinstance(parsed, dict):
            relationships = self._sanitize_relationships(parsed.get("relationships"), "")
        return {"label": label, "summary": summary, "relationships": relationships}

    def refine_summary(self, block, fragments: List[Fragment]) -> Dict[str, object]:
        roster = self._build_roster(block.block_id)
        payload = {
            "task": "refresh",
            "block": {
                "id": block.block_id,
                "label": block.label,
                "summary": block.summary or "",
            },
            "fragments": [self._summarize_fragment(fragment) for fragment in fragments],
            "others": roster,
            "canvas": {"size": [self._canvas_size[0], self._canvas_size[1]]},
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are maintaining the structured knowledge blocks on this canvas.\n"
                    "1. Rewrite the block summary so it covers all current fragments (aim for 120 characters or fewer).\n"
                    "2. Identify relationships between this block and other blocks (semantic, functional, or visual flow).\n"
                    "3. If you believe this block should be merged into another block (because they describe the same concept or the other block already subsumes it), add a merge directive. Use merge objects like {\"source\": \"current_block_id\", \"target\": \"block_xyz\"} or simply {\"target\": \"block_xyz\"}. Only merge when you are confident.\n"
                    "Return JSON {\"summary\": str, \"relationships\": [{\"type\": str, \"target\": str, \"score\": float? ...}], \"merge\"?: {... or [...]}}. Use relationship types such as refines, comment_on, or flow_next. Skip any relationship you cannot justify.\n"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        summary = _normalize_text(parsed.get("summary") if isinstance(parsed, dict) else None) or block.summary or ""
        relationships = []
        if isinstance(parsed, dict):
            relationships = self._sanitize_relationships(parsed.get("relationships"), block.block_id)
        return {"summary": summary[:220], "relationships": relationships}

    def _build_roster(self, current_block_id: str) -> List[Dict[str, object]]:
        roster: List[Dict[str, object]] = []
        provider = self._block_provider or (lambda: [])
        for other in provider():
            if other.block_id == current_block_id:
                continue
            info: Dict[str, object] = {
                "id": other.block_id,
                "label": other.label,
            }
            if other.summary:
                info["summary"] = other.summary[:160]
            if other.position:
                x0, y0, x1, y1 = other.position
                info["bbox"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
            roster.append(info)
            if len(roster) >= self.roster_limit:
                break
        return roster

    def _summarize_fragment(self, fragment: Fragment) -> Dict[str, object]:
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        if fragment.fragment_type == FragmentType.TEXT:
            text = (fragment.text or "").strip()
            return {
                "id": fragment.fragment_id,
                "type": "text",
                "text": text[:320],
            }
        desc: Dict[str, object] = {
            "id": fragment.fragment_id,
            "type": "stroke",
        }
        if fragment.bbox:
            x0, y0, x1, y1 = fragment.bbox
            desc["bbox"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        if isinstance(payload, dict):
            tool = payload.get("tool")
            if tool:
                desc["tool"] = tool
            meta = payload.get("meta")
            if isinstance(meta, dict):
                summary_meta = {k: v for k, v in meta.items() if k in {"desc", "summary", "note"}}
                if summary_meta:
                    desc["meta"] = summary_meta
        return desc

    def _sanitize_relationships(self, relationships, current_block_id: str) -> List[Dict[str, object]]:
        if not isinstance(relationships, list):
            return []
        cleaned: List[Dict[str, object]] = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            rel_type = _normalize_text(rel.get("type"))
            target = _normalize_text(rel.get("target"))
            if not rel_type or not target or target == current_block_id:
                continue
            item: Dict[str, object] = {
                "type": rel_type,
                "target": target,
            }
            score = rel.get("score")
            try:
                if score is not None:
                    item["score"] = float(score)
            except Exception:
                pass
            for key, value in rel.items():
                if key in {"type", "target", "score"}:
                    continue
                item[key] = value
            cleaned.append(item)
            if len(cleaned) >= self.roster_limit:
                break
        return cleaned
