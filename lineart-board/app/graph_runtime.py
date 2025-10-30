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
    ConversationOrchestrator,
    Fragment,
    FragmentType,
    GraphState,
    OrchestratorContext,
    PlanBackend,
    PromptBackend,
    PromptContext,
    PromptExecutor,
    TextEmbedder,
    VisionBackend,
    VisionGrouper,
    VisionPayload,
    VisionResult,
)
from semantic_graph.models import GroupNotFoundError

from app.embedding_client import embed_text
from app.llm_client import call_chat_completions

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
    def __init__(self, model: str = DEFAULT_SUMMARY_MODEL, max_tokens: int = 400) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def propose_block(self, fragments: List[Fragment]) -> tuple[str, str]:
        payload = {
            "task": "propose",
            "fragments": [_fragment_export(f) for f in fragments],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是知识图谱维护助手。根据提供的画布片段生成块标签和摘要。"
                    "始终返回 JSON 对象 {\"label\": str, \"summary\": str}。"
                    "label 应精炼、<=40 字符，summary 需兼顾上下文用途。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        label = _normalize_text(parsed.get("label") if isinstance(parsed, dict) else None) or "Untitled Block"
        summary = _normalize_text(parsed.get("summary") if isinstance(parsed, dict) else None) or label
        return label, summary

    def refine_summary(self, block, fragments: List[Fragment]) -> str:
        payload = {
            "task": "refine",
            "block": {"id": block.block_id, "label": block.label, "summary": block.summary},
            "fragments": [_fragment_export(f) for f in fragments],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "基于现有块信息生成更新后的摘要，保持 label 稳定。"
                    "输出 JSON {\"summary\": str}，摘要需≤220字符并覆盖新增要点。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        summary = _normalize_text(parsed.get("summary") if isinstance(parsed, dict) else None)
        return summary or block.summary


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


class LLMPromptBackend(PromptBackend):
    def __init__(self, model: str = DEFAULT_LLM_MODEL, max_tokens: int = 800) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        if isinstance(parsed, dict):
            return parsed
        return {"assistant_reply": str(parsed), "block_annotation": {}}


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
    ) -> None:
        width, height = canvas_size or (1920.0, 1080.0)
        self.state = GraphState()
        self.embedder = OpenAIEmbedder(model=embed_model)
        self.summarizer = LLMBlockSummarizer(model=summary_model or DEFAULT_SUMMARY_MODEL)
        self.block_manager = BlockManager(
            state=self.state,
            embedder=self.embedder,
            summarizer=self.summarizer,
            canvas_size=(float(width), float(height)),
        )
        self.plan_backend = LLMPlanBackend(model=plan_model or DEFAULT_PLAN_MODEL)
        self.orchestrator = ConversationOrchestrator(
            self.block_manager, embedder=self.embedder, plan_backend=self.plan_backend
        )
        self.prompt_backend = LLMPromptBackend(model=prompt_model or DEFAULT_LLM_MODEL)
        self.executor = PromptExecutor(self.block_manager, backend=self.prompt_backend)
        self.vision_backend = NoopVisionBackend()
        self.vision = VisionGrouper(self.block_manager, backend=self.vision_backend)
        self.context = OrchestratorContext()
        self._seen_fragment_ids: Set[str] = set()

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
        ctx = PromptContext(
            main_block_id=self.orchestrator.context.main_block_id,
            active_block_ids=list(self.orchestrator.context.active_block_ids),
        )
        assistant = self.executor.execute(plan, user_input, ctx, mode=mode)
        return {
            "plan": {
                "action": plan.action,
                "targetBlockIds": plan.target_block_ids,
                "comment": plan.comment,
            },
            "assistant_reply": assistant.assistant_reply,
            "block_annotation": assistant.block_annotation,
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
