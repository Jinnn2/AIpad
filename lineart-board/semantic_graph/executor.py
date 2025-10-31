from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, runtime_checkable

from app.schemas import AIStrokePayload, AIStrokeV11, CanvasInfo, SuggestRequest

from .block_manager import BlockManager
from .models import ExecutionPlan, Fragment, FragmentType


@runtime_checkable
class LLMFullBackend(Protocol):
    def __call__(self, messages: List[Dict[str, str]], *, mode: Optional[str] = None) -> Dict[str, object]:
        ...


@runtime_checkable
class MessageBuilder(Protocol):
    def __call__(self, request: SuggestRequest, include_sample: bool = True) -> List[Dict[str, str]]:
        ...


@dataclass
class FocusContext:
    main_block_id: Optional[str]
    active_block_ids: List[str]


class ContextExecutor:
    def __init__(
        self,
        block_manager: BlockManager,
        llm_full_backend: LLMFullBackend,
        *,
        build_full_messages: MessageBuilder,
        build_light_messages: MessageBuilder,
        max_blocks: int = 5,
        max_related_per_block: int = 2,
    ) -> None:
        self.block_manager = block_manager
        self.llm_full_backend = llm_full_backend
        self.build_full_messages = build_full_messages
        self.build_light_messages = build_light_messages
        self.max_blocks = max_blocks
        self.max_related_per_block = max_related_per_block

    def execute(
        self,
        plan: ExecutionPlan,
        user_hint: str,
        *,
        mode: Optional[str] = None,
        context: Optional[FocusContext] = None,
    ) -> Dict[str, object]:
        """Build a local canvas context and ask the FULL-mode backend for new strokes."""
        selected_blocks = self._select_blocks(plan, context)
        strokes = self._collect_strokes(selected_blocks)

        if not strokes:
            return {"version": 1, "intent": "complete", "strokes": []}

        canvas_width, canvas_height = self.block_manager.canvas_size or (1920.0, 1080.0)
        canvas_info = CanvasInfo(width=int(canvas_width), height=int(canvas_height))
        payload = AIStrokePayload(
            version=1,
            intent="complete",
            canvas=canvas_info,
            strokes=[AIStrokeV11.model_validate(stroke) for stroke in strokes],
        )

        request_mode = (mode or "full").lower()
        if request_mode not in {"full", "light"}:
            request_mode = "full"

        req = SuggestRequest(
            mode=request_mode,
            hint=user_hint,
            context=payload,
        )

        if request_mode == "light":
            messages = self.build_light_messages(req, include_sample=False)
        else:
            messages = self.build_full_messages(req, include_sample=True)

        response = self.llm_full_backend(messages, mode=request_mode)
        return response

    def _select_blocks(self, plan: ExecutionPlan, context: Optional[FocusContext]) -> List[str]:
        seeds: List[str] = []
        seen: Set[str] = set()

        for block_id in plan.target_block_ids or []:
            if block_id not in seen:
                seeds.append(block_id)
                seen.add(block_id)

        if not seeds and context:
            for block_id in context.active_block_ids or []:
                if block_id not in seen:
                    seeds.append(block_id)
                    seen.add(block_id)
            if not seeds and context.main_block_id and context.main_block_id not in seen:
                seeds.append(context.main_block_id)
                seen.add(context.main_block_id)

        if not seeds:
            for block in sorted(
                self.block_manager.state.blocks.values(),
                key=lambda b: b.updated_at,
                reverse=True,
            ):
                if block.block_id not in seen:
                    seeds.append(block.block_id)
                    seen.add(block.block_id)
                if len(seeds) >= self.max_blocks:
                    break

        return self._expand_blocks(seeds)

    def _expand_blocks(self, seed_block_ids: Sequence[str]) -> List[str]:
        selected: List[str] = []
        seen: Set[str] = set()

        for block_id in seed_block_ids:
            block = self.block_manager.state.blocks.get(block_id)
            if not block:
                continue
            if block_id not in seen:
                selected.append(block_id)
                seen.add(block_id)
            if len(selected) >= self.max_blocks:
                break

            related = sorted(block.relationships, key=lambda rel: rel.score, reverse=True)
            related_added = 0
            for rel in related:
                if related_added >= self.max_related_per_block or len(selected) >= self.max_blocks:
                    break
                target_id = rel.target_block_id
                if target_id in seen:
                    continue
                if target_id not in self.block_manager.state.blocks:
                    continue
                selected.append(target_id)
                seen.add(target_id)
                related_added += 1

            if len(selected) >= self.max_blocks:
                break

        return selected[: self.max_blocks]

    def _collect_strokes(self, block_ids: Sequence[str]) -> List[Dict[str, object]]:
        strokes: List[Dict[str, object]] = []
        seen_fragments: Set[str] = set()

        for block_id in block_ids:
            block = self.block_manager.state.blocks.get(block_id)
            if not block:
                continue
            for fragment_id in block.contents:
                if fragment_id in seen_fragments:
                    continue
                fragment = self.block_manager.state.fragments.get(fragment_id)
                if not fragment:
                    continue
                stroke = self._fragment_to_stroke(fragment)
                if stroke:
                    strokes.append(stroke)
                    seen_fragments.add(fragment_id)

        return strokes

    def _fragment_to_stroke(self, fragment: Fragment) -> Optional[Dict[str, object]]:
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        tool = payload.get("tool")
        style = payload.get("style") or None
        meta = payload.get("meta") or None
        points = payload.get("points")

        if not tool:
            tool = "text" if fragment.fragment_type == FragmentType.TEXT else "pen"

        if not points and fragment.bbox:
            x0, y0, x1, y1 = fragment.bbox
            points = [[x0, y0], [x1, y1]]

        if not points:
            return None

        norm_points: List[List[float]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            base = [float(pt[0]), float(pt[1])]
            extras: List[float] = []
            for extra in pt[2:]:
                try:
                    extras.append(float(extra))
                except Exception:
                    pass
            norm_points.append(base + extras)
        if not norm_points:
            return None

        stroke: Dict[str, object] = {
            "id": fragment.fragment_id,
            "tool": tool,
            "points": norm_points,
        }
        if isinstance(style, dict):
            stroke["style"] = dict(style)
        elif style is not None:
            stroke["style"] = style
        if isinstance(meta, dict):
            stroke["meta"] = dict(meta)
        elif meta is not None:
            stroke["meta"] = meta
        return stroke
