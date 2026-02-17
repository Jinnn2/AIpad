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
        max_groups: int = 4,
        max_related_per_block: int = 2,
    ) -> None:
        self.block_manager = block_manager
        self.llm_full_backend = llm_full_backend
        self.build_full_messages = build_full_messages
        self.build_light_messages = build_light_messages
        self.max_blocks = max_blocks
        self.max_groups = max_groups
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
        selected_blocks, selected_groups = self._select_context_entities(plan, context)
        block_outline = self._build_block_outline(selected_blocks, selected_groups)
        strokes = self._collect_strokes(selected_blocks, selected_groups)

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

        planner_next_step = self._normalize_planner_hint(plan.next_step_hint)
        composed_hint = self._compose_hint(user_hint, planner_next_step)
        req = SuggestRequest(
            mode=request_mode,
            hint=composed_hint,
            context=payload,
            planner_next_step=planner_next_step,
            block_outline=block_outline,
        )

        if request_mode == "light":
            messages = self.build_light_messages(req, include_sample=False)
        else:
            messages = self.build_full_messages(req, include_sample=True)

        response = self.llm_full_backend(messages, mode=request_mode)
        return response

    @staticmethod
    def _normalize_planner_hint(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        compact = " ".join(str(text).split()).strip()
        if not compact:
            return None
        return compact[:240]

    @staticmethod
    def _compose_hint(user_hint: str, planner_next_step: Optional[str]) -> str:
        base = (user_hint or "").strip()
        if not planner_next_step:
            return base
        if not base:
            return f"Planner next-step: {planner_next_step}"
        return f"{base}\n\nPlanner next-step: {planner_next_step}"

    def _collect_seed_ids(self, plan: ExecutionPlan, context: Optional[FocusContext]) -> List[str]:
        seeds: List[str] = []
        seen: Set[str] = set()

        for context_id in plan.target_block_ids or []:
            if context_id not in seen:
                seeds.append(context_id)
                seen.add(context_id)

        if not seeds and context:
            for context_id in context.active_block_ids or []:
                if context_id not in seen:
                    seeds.append(context_id)
                    seen.add(context_id)
            if not seeds and context.main_block_id and context.main_block_id not in seen:
                seeds.append(context.main_block_id)
                seen.add(context.main_block_id)

        return seeds

    def _select_context_entities(self, plan: ExecutionPlan, context: Optional[FocusContext]) -> tuple[List[str], List[str]]:
        seeds = self._collect_seed_ids(plan, context)
        seed_blocks: List[str] = []
        seed_groups: List[str] = []
        for context_id in seeds:
            if context_id in self.block_manager.state.blocks:
                seed_blocks.append(context_id)
                continue
            if context_id in self.block_manager.state.groups:
                seed_groups.append(context_id)

        selected_blocks = self._expand_blocks(seed_blocks)
        selected_groups = self._select_groups(seed_groups)

        if not selected_blocks and not selected_groups:
            for block in sorted(
                self.block_manager.state.blocks.values(),
                key=lambda b: b.updated_at,
                reverse=True,
            ):
                seed_blocks.append(block.block_id)
                if len(seed_blocks) >= self.max_blocks:
                    break
            selected_blocks = self._expand_blocks(seed_blocks)

        if not selected_groups and not selected_blocks:
            for group in sorted(
                self.block_manager.state.groups.values(),
                key=lambda g: g.updated_at,
                reverse=True,
            ):
                seed_groups.append(group.group_id)
                if len(seed_groups) >= self.max_groups:
                    break
            selected_groups = self._select_groups(seed_groups)

        return selected_blocks, selected_groups

    def _select_groups(self, seed_group_ids: Sequence[str]) -> List[str]:
        selected: List[str] = []
        seen: Set[str] = set()
        for group_id in seed_group_ids:
            if group_id in seen:
                continue
            if group_id not in self.block_manager.state.groups:
                continue
            selected.append(group_id)
            seen.add(group_id)
            if len(selected) >= self.max_groups:
                break
        return selected

    def _select_blocks(self, plan: ExecutionPlan, context: Optional[FocusContext]) -> List[str]:
        """Backward-compatible block-only selection helper."""
        seeds = self._collect_seed_ids(plan, context)
        block_seeds: List[str] = []
        seen: Set[str] = set()
        for context_id in seeds:
            if context_id not in self.block_manager.state.blocks:
                continue
            if context_id in seen:
                continue
            block_seeds.append(context_id)
            seen.add(context_id)
        if not block_seeds:
            for block in sorted(
                self.block_manager.state.blocks.values(),
                key=lambda b: b.updated_at,
                reverse=True,
            ):
                if block.block_id in seen:
                    continue
                block_seeds.append(block.block_id)
                seen.add(block.block_id)
                if len(block_seeds) >= self.max_blocks:
                    break
        return self._expand_blocks(block_seeds)

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

    def _collect_strokes(self, block_ids: Sequence[str], group_ids: Sequence[str] = ()) -> List[Dict[str, object]]:
        strokes: List[Dict[str, object]] = []
        seen_fragments: Set[str] = set()

        for block_id in block_ids:
            block = self.block_manager.state.blocks.get(block_id)
            if not block:
                continue
            for fragment_id in sorted(block.contents):
                if fragment_id in seen_fragments:
                    continue
                fragment = self.block_manager.state.fragments.get(fragment_id)
                if not fragment:
                    continue
                stroke = self._fragment_to_stroke(fragment)
                if stroke:
                    strokes.append(stroke)
                    seen_fragments.add(fragment_id)

        for group_id in group_ids:
            group = self.block_manager.state.groups.get(group_id)
            if not group:
                continue
            for fragment_id in sorted(group.members):
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

    def _build_block_outline(self, block_ids: Sequence[str], group_ids: Sequence[str] = ()) -> List[Dict[str, object]]:
        outline: List[Dict[str, object]] = []
        rank = 1
        for block_id in block_ids:
            block = self.block_manager.state.blocks.get(block_id)
            if not block:
                continue

            fragment_ids = sorted(block.contents)
            item: Dict[str, object] = {
                "rank": rank,
                "entityType": "block",
                "blockId": block.block_id,
                "label": block.label,
                "summary": block.summary,
                "fragmentCount": len(fragment_ids),
                "fragmentIds": fragment_ids[:12],
            }

            text_snippets = self._collect_text_snippets(fragment_ids)
            if text_snippets:
                item["textSnippets"] = text_snippets

            relationships: List[Dict[str, object]] = []
            for rel in sorted(block.relationships, key=lambda rel: rel.score, reverse=True)[: self.max_related_per_block]:
                relationships.append(
                    {
                        "targetBlockId": rel.target_block_id,
                        "type": rel.rel_type.value,
                        "score": round(float(rel.score), 4),
                    }
                )
            if relationships:
                item["relationships"] = relationships

            outline.append(item)
            rank += 1

        for group_id in group_ids:
            group = self.block_manager.state.groups.get(group_id)
            if not group:
                continue
            fragment_ids = sorted(group.members)
            fragments: List[Dict[str, object]] = []
            for fragment_id in fragment_ids:
                fragment = self.block_manager.state.fragments.get(fragment_id)
                compact = self._compact_fragment_for_group(fragment)
                if compact:
                    fragments.append(compact)
            if fragments:
                # For group entries, keep only core fragment data.
                outline.append({"fragments": fragments})
            rank += 1
        return outline

    def _collect_text_snippets(self, fragment_ids: Sequence[str], limit: int = 3) -> List[Dict[str, str]]:
        text_snippets: List[Dict[str, str]] = []
        for fid in fragment_ids:
            fragment = self.block_manager.state.fragments.get(fid)
            if not fragment:
                continue
            text = " ".join(str(fragment.text or "").split()).strip()
            if not text:
                payload = fragment.payload if isinstance(fragment.payload, dict) else {}
                meta = payload.get("meta") if isinstance(payload, dict) else {}
                if isinstance(meta, dict):
                    text = " ".join(str(meta.get("text") or "").split()).strip()
            if not text:
                continue
            text_snippets.append({"fragmentId": fid, "text": text[:120]})
            if len(text_snippets) >= limit:
                break
        return text_snippets

    def _compact_fragment_for_group(self, fragment: Optional[Fragment]) -> Optional[Dict[str, object]]:
        if fragment is None:
            return None
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        kind = getattr(fragment.fragment_type, "value", str(fragment.fragment_type))
        if kind == "text":
            text = " ".join(str(fragment.text or "").split()).strip()
            if not text:
                meta = payload.get("meta") if isinstance(payload, dict) else {}
                if isinstance(meta, dict):
                    text = " ".join(str(meta.get("text") or "").split()).strip()
            item: Dict[str, object] = {"type": "text", "text": text[:320]}
            if fragment.bbox:
                item["bbox"] = [round(float(v), 2) for v in fragment.bbox]
            return item

        item = {"type": "stroke", "strokeType": str(payload.get("tool") or "stroke")}
        point = self._compact_point(payload.get("points"))
        if point:
            item["point"] = point
        return item

    @staticmethod
    def _compact_point(raw_points: object) -> Optional[List[float]]:
        if isinstance(raw_points, dict):
            try:
                return [round(float(raw_points.get("x")), 2), round(float(raw_points.get("y")), 2)]
            except (TypeError, ValueError):
                return None
        if not isinstance(raw_points, (list, tuple)):
            return None
        if len(raw_points) >= 2 and not isinstance(raw_points[0], (list, tuple, dict)):
            try:
                return [round(float(raw_points[0]), 2), round(float(raw_points[1]), 2)]
            except (TypeError, ValueError):
                return None
        latest: Optional[List[float]] = None
        for point in raw_points:
            if isinstance(point, dict):
                try:
                    latest = [round(float(point.get("x")), 2), round(float(point.get("y")), 2)]
                except (TypeError, ValueError):
                    continue
                continue
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                latest = [round(float(point[0]), 2), round(float(point[1]), 2)]
            except (TypeError, ValueError):
                continue
        return latest

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
