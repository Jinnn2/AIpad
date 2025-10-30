from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .block_manager import BlockManager
from .models import AssistantReply, ExecutionPlan, FragmentType


@runtime_checkable
class PromptBackend(Protocol):
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        ...


@dataclass
class PromptContext:
    main_block_id: Optional[str]
    active_block_ids: List[str]


class PromptExecutor:
    def __init__(
        self,
        block_manager: BlockManager,
        backend: PromptBackend,
        *,
        max_blocks: int = 5,
    ) -> None:
        self.block_manager = block_manager
        self.backend = backend
        self.max_blocks = max_blocks

    def execute(
        self,
        plan: ExecutionPlan,
        user_input: str,
        context: PromptContext,
        *,
        mode: Optional[str] = None,
    ) -> AssistantReply:
        block_ids = self._resolve_blocks(plan, context)
        block_context = self._format_blocks(block_ids)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是画布协作助手，根据提供的块上下文与用户输入生成新的回复。"
                    "确保遵守块之间的关系，保持结构化引用。"
                ),
            },
            {
                "role": "user",
                "content": f"{block_context}\n用户输入: {user_input}\n给出你的回复，并返回 JSON {{\"assistant_reply\":\"...\",\"block_annotation\":{{...}}}}。",
            },
        ]

        raw_response = self.backend.complete(messages, mode=mode or plan.action)
        assistant_reply = self._extract_assistant_reply(raw_response)
        block_annotation = self._extract_block_annotation(raw_response)
        self._apply_annotations(block_annotation, default_block_id=context.main_block_id)
        return AssistantReply(assistant_reply=assistant_reply, block_annotation=block_annotation or {})

    def _resolve_blocks(self, plan: ExecutionPlan, context: PromptContext) -> List[str]:
        if plan.target_block_ids:
            return plan.target_block_ids[: self.max_blocks]
        if context.active_block_ids:
            return context.active_block_ids[: self.max_blocks]
        if context.main_block_id:
            return [context.main_block_id]
        return []

    def _format_blocks(self, block_ids: List[str]) -> str:
        lines: List[str] = []
        for idx, block_id in enumerate(block_ids, start=1):
            block = self.block_manager.state.blocks.get(block_id)
            if not block:
                continue
            lines.append(f"块 {idx}: {block.label} ({block_id})")
            if block.summary:
                lines.append(f"摘要: {block.summary}")
            texts = []
            other = 0
            for fragment_id in block.contents:
                fragment = self.block_manager.state.fragments.get(fragment_id)
                if not fragment:
                    continue
                if fragment.fragment_type == FragmentType.TEXT and fragment.text:
                    texts.append(f"- 文本 {fragment.fragment_id}: {fragment.text}")
                else:
                    other += 1
            if texts:
                lines.extend(texts)
            if other:
                lines.append(f"- 其他笔画: {other} 条")
            if block.relationships:
                for rel in block.relationships:
                    lines.append(
                        f"- 关系 {rel.rel_type.value} -> {rel.target_block_id} (分数: {rel.score:.2f})"
                    )
        return "\n".join(lines)

    def _extract_assistant_reply(self, response: Dict[str, object]) -> str:
        if not isinstance(response, dict):
            return ""
        reply = response.get("assistant_reply") or response.get("reply") or ""
        return str(reply)

    def _extract_block_annotation(self, response: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not isinstance(response, dict):
            return None
        annotation = response.get("block_annotation") or response.get("annotation")
        if isinstance(annotation, dict):
            return annotation
        if isinstance(annotation, list) and annotation:
            merged: Dict[str, object] = {}
            for item in annotation:
                if isinstance(item, dict):
                    merged.update(item)
            return merged if merged else None
        return None

    def _apply_annotations(self, annotation: Optional[Dict[str, object]], *, default_block_id: Optional[str]) -> None:
        if not annotation:
            return
        block_id = annotation.get("block_id") or default_block_id
        if not block_id:
            return
        self.block_manager.register_block_annotation(block_id, annotation)

