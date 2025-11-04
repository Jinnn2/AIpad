from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))

from semantic_graph import (
    BlockManager,
    BlockSummarizer,
    ContextExecutor,
    ConversationOrchestrator,
    FocusContext,
    Fragment,
    FragmentType,
    GraphState,
    OrchestratorContext,
    PlanBackend,
    TextEmbedder,
    VisionBackend,
    VisionGrouper,
    VisionPayload,
    VisionResult,
)
from semantic_graph.models import GroupNotFoundError

from app.embedding_client import embed_text
from app.llm_client import call_chat_completions
from app.cluster_logging import ClusterLogger
from app import prompting

DEFAULT_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
DEFAULT_PLAN_MODEL = os.getenv("GRAPH_PLAN_MODEL") or DEFAULT_LLM_MODEL
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


class OpenAIEmbedder(TextEmbedder):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    def embed(self, text: str) -> Sequence[float]:
        return embed_text(text or "", model=self.model)


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

    def propose_block(self, fragments: List[Fragment]) -> tuple[str, str]:
        payload = {
            "task": "propose",
            "fragments": [_fragment_export(f) for f in fragments],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-graph curator for a collaborative canvas.\n"
                    "You are the knowledge-graph curator for this collaborative canvas.\n"
                    "Always return a JSON object {\"label\": str, \"summary\": str}.\n"
                    "Keep the label concise (<= 40 characters) and write a summary that captures the block's purpose for future context.\n")
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        label = _normalize_text(parsed.get("label") if isinstance(parsed, dict) else None) or "Untitled Block"
        summary = _normalize_text(parsed.get("summary") if isinstance(parsed, dict) else None) or label
        return label, summary

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
                    "Return JSON {\"summary\": str, \"relationships\": [{\"type\": str, \"target\": str, \"score\": float? ...}]}. Use relationship types such as refines, comment_on, or flow_next. Skip any relationship you cannot justify.\n"
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


class LLMPlanBackend(PlanBackend):
    def __init__(self, model: str = DEFAULT_PLAN_MODEL, max_tokens: int = 240) -> None:
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




class NoopVisionBackend(VisionBackend):
    """Placeholder vision backend."""

    def analyze(self, payload: VisionPayload) -> List[VisionResult]:
        return []


@dataclass
class GraphIngestResult:
    new_fragments: List[str]
    promoted_blocks: List[str]


class GraphRuntime:
    """
    Runtime wrapper that wires the semantic_graph package to the FastAPI session layer.
    """

    def __init__(
        self,
        *,
        canvas_size: tuple[float, float] | None = None,
        embed_model: Optional[str] = None,
        summary_model: Optional[str] = None,
        plan_model: Optional[str] = None,
        prompt_model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        width, height = canvas_size or (1920.0, 1080.0)
        self.state = GraphState()
        self.embedder = OpenAIEmbedder(model=embed_model)
        self.summarizer = LLMBlockSummarizer(model=summary_model or DEFAULT_SUMMARY_MODEL)
        self.cluster_logger = ClusterLogger(session_id=session_id)
        self.block_manager = BlockManager(
            state=self.state,
            embedder=self.embedder,
            summarizer=self.summarizer,
            canvas_size=(float(width), float(height)),
            cluster_logger=self.cluster_logger,
        )
        self.summarizer.set_block_provider(lambda: self.block_manager.state.blocks.values())
        self.summarizer.set_canvas_size(self.block_manager.canvas_size)

        self.plan_backend = LLMPlanBackend(model=plan_model or DEFAULT_PLAN_MODEL)
        self.orchestrator = ConversationOrchestrator(
            self.block_manager, embedder=self.embedder, plan_backend=self.plan_backend
        )
        self.context_executor = ContextExecutor(
            self.block_manager,
            llm_full_backend=self._call_full_backend,
            build_full_messages=prompting.build_messages,
            build_light_messages=prompting.build_messages_light,
        )
        self.vision_backend = NoopVisionBackend()
        self.vision = VisionGrouper(self.block_manager, backend=self.vision_backend)
        self.context = OrchestratorContext()
        self._seen_fragment_ids: Set[str] = set()

    def _call_full_backend(self, messages: List[Dict[str, str]], *, mode: Optional[str] = None) -> Dict[str, object]:
        parsed, dbg = call_chat_completions(messages, model=DEFAULT_LLM_MODEL, max_tokens=900)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except Exception as exc:
                raise RuntimeError(f"LLM returned non-JSON response: {parsed!r}") from exc
        raise RuntimeError(f"Unexpected LLM response: {type(parsed)!r}")

    def ingest_strokes(self, strokes: Iterable[Dict[str, object]]) -> GraphIngestResult:
        new_fragments: List[str] = []
        promoted_blocks: List[str] = []
        for stroke in strokes or []:
            fragment = self._stroke_to_fragment(stroke)
            if fragment is None:
                continue
            if fragment.fragment_id in self._seen_fragment_ids:
                continue
            if fragment.fragment_id in self.state.fragments:
                self._seen_fragment_ids.add(fragment.fragment_id)
                continue
            self._seen_fragment_ids.add(fragment.fragment_id)
            assignment = self.block_manager.register_fragment(fragment)
            new_fragments.append(fragment.fragment_id)
            if self.cluster_logger:
                try:
                    self.cluster_logger.log(
                        "ingest_assignment",
                        {
                            "fragment_id": fragment.fragment_id,
                            "status": assignment.status,
                            "block_id": assignment.block_id,
                            "group_id": assignment.group_id,
                            "promoted_block_id": assignment.promoted_block_id,
                            "text": _normalize_text(fragment.text)[:80],
                            "stroke_tool": str(stroke.get("tool")) if isinstance(stroke, dict) else None,
                        },
                    )
                except Exception:
                    pass
            if assignment.promoted_block_id:
                promoted_blocks.append(assignment.promoted_block_id)
        return GraphIngestResult(new_fragments=new_fragments, promoted_blocks=promoted_blocks)

    def snapshot(self) -> Dict[str, object]:
        blocks = []
        for block in self.state.blocks.values():
            blocks.append(
                {
                    "blockId": block.block_id,
                    "label": block.label,
                    "summary": block.summary,
                    "position": block.position,
                    "contents": list(block.contents),
                    "relationships": [
                        {
                            "target": rel.target_block_id,
                            "type": rel.rel_type.value,
                            "score": rel.score,
                        }
                    for rel in block.relationships
                    ],
                    "updatedAt": block.updated_at.isoformat(),
                }
            )
        groups = []
        for group in self.state.groups.values():
            groups.append(
                {
                    "groupId": group.group_id,
                    "size": len(group.members),
                    "state": group.state.value,
                    "needLLMReview": group.need_llm_review,
                    "members": list(group.members),
                    "touchCount": self.block_manager.get_group_touch_count(group.group_id),
                    "updatedAt": group.updated_at.isoformat(),
                }
            )
        fragments = []
        for fragment in self.state.fragments.values():
            graph_meta = (fragment.payload or {}).get("graph") if isinstance(fragment.payload, dict) else None
            block_label = None
            block_id = None
            if isinstance(graph_meta, dict):
                block_label = graph_meta.get("blockLabel")
                block_id = graph_meta.get("blockId")
            fragments.append(
                {
                    "id": fragment.fragment_id,
                    "type": fragment.fragment_type.value,
                    "bbox": fragment.bbox,
                    "text": fragment.text,
                    "timestamp": fragment.timestamp.isoformat() if fragment.timestamp else None,
                    "blockId": block_id,
                    "blockLabel": block_label,
                }
            )
        return {
            "blocks": blocks,
            "fragments": fragments,
            "groups": groups,
        }

    def promote_group_now(self, group_id: str) -> Optional[Block]:
        try:
            block = self.block_manager.mark_group_stable(group_id)
        except GroupNotFoundError:
            return None
        return block

    def run_conversation(
        self,
        user_input: str,
        *,
        focus_block_id: Optional[str] = None,
        focus_fragment_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        plan = self.orchestrator.generate_plan(
            user_input,
            focus_block_id=focus_block_id,
            focus_fragment_id=focus_fragment_id,
        )
        focus_context = FocusContext(
            main_block_id=self.orchestrator.context.main_block_id,
            active_block_ids=list(self.orchestrator.context.active_block_ids),
        )
        exec_mode = (mode or "").lower()
        if not exec_mode and plan.action:
            candidate = plan.action.lower()
            if candidate in {"full", "light"}:
                exec_mode = candidate
        response = self.context_executor.execute(
            plan,
            user_hint=user_input,
            mode=exec_mode or None,
            context=focus_context,
        )
        return {
            "plan": {
                "action": plan.action,
                "targetBlockIds": plan.target_block_ids,
                "comment": plan.comment,
            },
            "payload": response,
        }

    # ----------------------------- helpers ----------------------------- #

    def _stroke_to_fragment(self, stroke: Dict[str, object]) -> Optional[Fragment]:
        stroke_id = str(stroke.get("id") or "").strip()
        if not stroke_id:
            return None
        tool = str(stroke.get("tool") or "").lower()
        if tool in {"eraser", "cursor"}:
            return None
        points = stroke.get("points") or []
        bbox = self._points_to_bbox(points)
        meta = stroke.get("meta") or {}
        if isinstance(meta, dict):
            meta_payload = dict(meta)
        else:
            meta_payload = {}
        timestamp = datetime.utcnow()
        fragment_type = FragmentType.TEXT if tool == "text" else FragmentType.STROKE
        text = ""
        if fragment_type == FragmentType.TEXT:
            raw_text = meta_payload.get("text") if isinstance(meta_payload, dict) else None
            text = _normalize_text(raw_text) or _normalize_text(meta_payload.get("summary"))
        payload = {
            "tool": tool,
            "style": stroke.get("style"),
            "meta": meta_payload,
            "points": stroke.get("points"),
        }
        return Fragment(
            fragment_id=stroke_id,
            fragment_type=fragment_type,
            bbox=bbox,
            text=text or None,
            timestamp=timestamp,
            payload=payload,
        )

    def _points_to_bbox(self, points: object) -> Optional[tuple[float, float, float, float]]:
        if not isinstance(points, Iterable):
            return None
        xs: List[float] = []
        ys: List[float] = []
        for item in points:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                xs.append(float(item[0]))
                ys.append(float(item[1]))
            except (TypeError, ValueError):
                continue
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

