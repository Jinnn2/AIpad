from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .block_manager import BlockManager, TextEmbedder
from .models import Block, BlockRelationshipType, ExecutionPlan
from .similarity import cosine_distance


@dataclass
class OrchestratorContext:
    main_block_id: Optional[str] = None
    active_block_ids: List[str] = field(default_factory=list)


@runtime_checkable
class PlanBackend(Protocol):
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Return a JSON string describing the execution plan."""


class ConversationOrchestrator:
    def __init__(
        self,
        block_manager: BlockManager,
        embedder: TextEmbedder,
        plan_backend: Optional[PlanBackend] = None,
        *,
        similarity_threshold: float = 0.35,
    ) -> None:
        self.block_manager = block_manager
        self.embedder = embedder
        self.plan_backend = plan_backend
        self.similarity_threshold = similarity_threshold
        self.context = OrchestratorContext()

    def generate_plan(
        self,
        user_input: str,
        *,
        focus_block_id: Optional[str] = None,
        focus_fragment_id: Optional[str] = None,
    ) -> ExecutionPlan:
        main_block = self._resolve_focus_block(focus_block_id, focus_fragment_id)
        input_embedding = list(self.embedder.embed(user_input))
        best_block_id = self._nearest_block(input_embedding)
        if best_block_id and not main_block:
            main_block = best_block_id

        summaries = self._collect_block_summaries(main_block)
        prompt = self._build_prompt(main_block, summaries, user_input)
        if self.plan_backend:
            response_text = self.plan_backend.complete(prompt)
            plan = self._parse_plan(response_text)
        else:
            plan = ExecutionPlan(action="NOOP", target_block_ids=[], comment="plan backend unavailable")

        self._update_context(main_block, plan)
        return plan

    def _resolve_focus_block(
        self,
        focus_block_id: Optional[str],
        focus_fragment_id: Optional[str],
    ) -> Optional[str]:
        if focus_block_id and focus_block_id in self.block_manager.state.blocks:
            return focus_block_id
        if focus_fragment_id:
            block_id = self.block_manager.get_block_id_for_fragment(focus_fragment_id)
            if block_id:
                return block_id
        return None

    def _nearest_block(self, embedding: List[float]) -> Optional[str]:
        best_id = None
        best_distance = float("inf")
        for block in self.block_manager.state.list_blocks():
            block_embedding = self._ensure_block_embedding(block)
            if not block_embedding:
                continue
            distance = cosine_distance(embedding, block_embedding)
            if distance < best_distance:
                best_distance = distance
                best_id = block.block_id
        if best_id is None or best_distance > self.similarity_threshold:
            return None
        return best_id

    def _ensure_block_embedding(self, block: Block) -> Optional[List[float]]:
        if block.embedding is not None:
            return list(block.embedding)
        text = block.summary or block.label
        if not text:
            return None
        embedding = list(self.embedder.embed(text))
        block.embedding = embedding
        return embedding

    def _collect_block_summaries(self, main_block_id: Optional[str]) -> Dict[str, Dict[str, str]]:
        summaries: Dict[str, Dict[str, str]] = {}
        if not main_block_id:
            return summaries
        block = self.block_manager.state.blocks.get(main_block_id)
        if not block:
            return summaries
        summaries[block.block_id] = {
            "label": block.label,
            "summary": block.summary,
        }
        for rel in block.relationships:
            if rel.rel_type not in {
                BlockRelationshipType.REFINES,
                BlockRelationshipType.COMMENT_ON,
                BlockRelationshipType.FLOW_NEXT,
            }:
                continue
            related = self.block_manager.state.blocks.get(rel.target_block_id)
            if not related:
                continue
            summaries[related.block_id] = {
                "label": related.label,
                "summary": related.summary,
                "relationship": rel.rel_type.value,
            }
        return summaries

    def _build_prompt(
        self,
        main_block_id: Optional[str],
        summaries: Dict[str, Dict[str, str]],
        user_input: str,
    ) -> List[Dict[str, str]]:
        context_lines = []
        if main_block_id:
            main_info = summaries.get(main_block_id) or {}
            context_lines.append(f"当前焦点块: {main_info.get('label', main_block_id)}")
        if summaries:
            context_lines.append("相关块概览:")
            for block_id, info in summaries.items():
                label = info.get("label", block_id)
                summary = info.get("summary", "")
                rel = info.get("relationship")
                if rel:
                    context_lines.append(f"- [{label}] ({rel}) {summary}")
                else:
                    context_lines.append(f"- [{label}] {summary}")
        context_lines.append(f"用户输入: {user_input}")
        user_prompt = "\n".join(context_lines) + "\n请输出 JSON: {\"action\":...,\"targetBlockIds\":[],\"comment\":\"\"}"
        return [
            {
                "role": "system",
                "content": (
                    "你是交互式白板编排器。阅读上下文，决定系统的下一步行为。"
                    "必须返回 JSON，字段 action/targetBlockIds/comment。"
                    "action 取 CONTINUE/SWITCH/OPEN_RELATED/CLOSE/NOOP 之一。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    def _parse_plan(self, text: str) -> ExecutionPlan:
        text = text.strip()
        candidate = text
        if "```" in text:
            chunks = text.split("```")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk.startswith("{") and chunk.endswith("}"):
                    candidate = chunk
                    break
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return ExecutionPlan(action="NOOP", target_block_ids=[], comment="failed to parse plan")
        action = str(parsed.get("action") or "NOOP")
        targets = [str(i) for i in (parsed.get("targetBlockIds") or [])]
        comment = parsed.get("comment")
        return ExecutionPlan(action=action, target_block_ids=targets, comment=comment)

    def _update_context(self, main_block_id: Optional[str], plan: ExecutionPlan) -> None:
        if plan.action == "SWITCH":
            self.context.active_block_ids = plan.target_block_ids
        elif plan.action == "OPEN_RELATED":
            current = set(self.context.active_block_ids)
            current.update(plan.target_block_ids)
            self.context.active_block_ids = list(current)
        elif plan.action == "CLOSE":
            remaining = [bid for bid in self.context.active_block_ids if bid not in plan.target_block_ids]
            self.context.active_block_ids = remaining
        if plan.target_block_ids:
            self.context.main_block_id = plan.target_block_ids[0]
        elif main_block_id:
            self.context.main_block_id = main_block_id

