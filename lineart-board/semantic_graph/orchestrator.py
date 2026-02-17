from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, runtime_checkable

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
        normalized_input = self._normalize_user_input(user_input)
        main_block = self._resolve_focus_block(focus_block_id, focus_fragment_id)
        best_block_id: Optional[str] = None
        if normalized_input:
            input_embedding = list(self.embedder.embed(normalized_input))
            best_block_id = self._nearest_block(input_embedding)
        else:
            best_block_id = self._latest_block_id()
        if best_block_id and not main_block:
            main_block = best_block_id

        summaries = self._collect_block_summaries(main_block)
        group_candidates = self._collect_group_candidates()
        if not main_block and summaries:
            main_block = next(iter(summaries.keys()))
        latest_context = self._build_latest_context(main_block)
        prompt = self._build_prompt(
            main_block,
            summaries,
            normalized_input,
            latest_context,
            group_candidates,
        )
        if self.plan_backend:
            response_text = self.plan_backend.complete(prompt)
            plan = self._parse_plan(response_text)
            plan.target_block_ids = self._resolve_targets(plan.target_block_ids, summaries)
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
            pass
        else:
            block = self.block_manager.state.blocks.get(main_block_id)
            if block:
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
        if not summaries:
            recent_blocks = sorted(
                self.block_manager.state.list_blocks(),
                key=lambda b: getattr(b, "updated_at", None) or datetime.min,
                reverse=True,
            )
            for block in recent_blocks[:5]:
                if block.block_id in summaries:
                    continue
                summaries[block.block_id] = {
                    "label": block.label,
                    "summary": block.summary,
                }
        return summaries

    def _collect_group_candidates(self, limit: int = 6) -> List[Dict[str, object]]:
        groups = sorted(
            self.block_manager.state.list_groups(),
            key=lambda g: getattr(g, "updated_at", None) or datetime.min,
            reverse=True,
        )
        candidates: List[Dict[str, object]] = []
        for group in groups[: max(0, limit)]:
            text_preview = self._group_text_preview(group.members)
            item: Dict[str, object] = {
                "groupId": group.group_id,
                "size": len(group.members),
                "updatedAt": group.updated_at.isoformat(),
            }
            if text_preview:
                item["textPreview"] = text_preview
            candidates.append(item)
        return candidates

    def _group_text_preview(self, fragment_ids: Iterable[str], *, max_chars: int = 100) -> str:
        snippets: List[str] = []
        for fragment_id in fragment_ids:
            fragment = self.block_manager.state.fragments.get(fragment_id)
            if not fragment:
                continue
            payload = fragment.payload if isinstance(fragment.payload, dict) else {}
            meta = payload.get("meta") if isinstance(payload, dict) else {}
            text = ""
            if isinstance(meta, dict):
                text = str(meta.get("text") or "").strip()
            if not text:
                text = str(fragment.text or "").strip()
            if not text:
                continue
            compact = " ".join(text.split()).strip()
            if compact:
                snippets.append(compact)
            if len(" ".join(snippets)) >= max_chars:
                break
            if len(snippets) >= 2:
                break
        preview = " | ".join(snippets).strip()
        return preview[:max_chars]

    def _build_prompt(
        self,
        main_block_id: Optional[str],
        summaries: Dict[str, Dict[str, str]],
        user_input: str,
        latest_context: Optional[Dict[str, object]],
        group_candidates: List[Dict[str, object]],
    ) -> List[Dict[str, str]]:
        context_lines = []
        if latest_context:
            context_lines.append("LATEST_CONTEXT:")
            context_lines.append(json.dumps(latest_context, ensure_ascii=False))
        if main_block_id:
            main_info = summaries.get(main_block_id) or {}
            context_lines.append(f"FOCUSED: {main_info.get('label', main_block_id)}")
        if summaries:
            context_lines.append("RELATED BLOCKS:")
            for block_id, info in summaries.items():
                label = info.get("label", block_id)
                summary = info.get("summary", "")
                rel = info.get("relationship")
                if rel:
                    context_lines.append(f"- [{label}] ({rel}) {summary}")
                else:
                    context_lines.append(f"- [{label}] {summary}")
        if group_candidates:
            context_lines.append("RELATED GROUPS:")
            for item in group_candidates:
                group_id = str(item.get("groupId") or "").strip()
                if not group_id:
                    continue
                size = int(item.get("size") or 0)
                updated_at = str(item.get("updatedAt") or "")
                preview = str(item.get("textPreview") or "").strip()
                line = f"- [{group_id}] size={size} updatedAt={updated_at}"
                if preview:
                    line += f" text={preview}"
                context_lines.append(line)
        if user_input:
            context_lines.append(f"USERS INPUT: {user_input}")
        else:
            context_lines.append("USERS INPUT: (none provided)")
            context_lines.append(
                "TASK: Infer the user's intent from the latest operations and related text, "
                "then decide which context is needed."
            )
        user_prompt = (
            "\n".join(context_lines)
            + "\nPlease return JSON: "
            + "{\"action\":...,\"targetBlockIds\":[],\"comment\":\"\",\"nextStepHint\":\"\"}"
        )
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]

    def _build_latest_context(self, focused_block_id: Optional[str]) -> Optional[Dict[str, object]]:
        latest_block = None
        latest_group = None

        blocks = self.block_manager.state.list_blocks()
        groups = self.block_manager.state.list_groups()

        if blocks:
            latest_block = max(blocks, key=lambda block: getattr(block, "updated_at", None) or datetime.min)
        if groups:
            latest_group = max(groups, key=lambda group: getattr(group, "updated_at", None) or datetime.min)

        if not latest_block and not latest_group:
            return None

        if latest_group and latest_block:
            choose_group = (latest_group.updated_at or datetime.min) >= (latest_block.updated_at or datetime.min)
        else:
            choose_group = bool(latest_group)

        if choose_group and latest_group:
            members = []
            for fragment_id in latest_group.members:
                fragment = self.block_manager.state.fragments.get(fragment_id)
                if fragment:
                    members.append(fragment)
            members.sort(key=lambda frag: frag.timestamp or datetime.min, reverse=True)
            compact_members = []
            for fragment in members:
                compact = self._compact_fragment(fragment)
                if compact:
                    compact_members.append(compact)
            return {
                "kind": "group",
                "groupId": latest_group.group_id,
                "updatedAt": latest_group.updated_at.isoformat(),
                "fragmentCount": len(compact_members),
                "fragments": compact_members,
            }

        if not latest_block:
            return None
        if focused_block_id and latest_block.block_id == focused_block_id:
            # If semantic focus already points to the same block, skip latest context.
            return None

        latest_fragment = self._latest_fragment_for_ids(latest_block.contents)
        payload: Dict[str, object] = {
            "kind": "block",
            "blockId": latest_block.block_id,
            "label": latest_block.label,
            "summary": latest_block.summary,
            "updatedAt": latest_block.updated_at.isoformat(),
        }
        compact_latest = self._compact_fragment(latest_fragment) if latest_fragment else None
        if compact_latest:
            payload["latestFragment"] = compact_latest
        return payload

    def _latest_fragment_for_ids(self, fragment_ids: Iterable[str]):
        latest = None
        latest_ts = datetime.min
        for fragment_id in fragment_ids:
            fragment = self.block_manager.state.fragments.get(fragment_id)
            if not fragment:
                continue
            ts = fragment.timestamp or datetime.min
            if ts >= latest_ts:
                latest = fragment
                latest_ts = ts
        return latest

    def _compact_fragment(self, fragment) -> Optional[Dict[str, object]]:
        if fragment is None:
            return None

        kind = getattr(fragment.fragment_type, "value", str(fragment.fragment_type))
        if kind == "text":
            payload = fragment.payload if isinstance(fragment.payload, dict) else {}
            meta = payload.get("meta") if isinstance(payload, dict) else {}
            text = ""
            if isinstance(meta, dict):
                text = str(meta.get("text") or "").strip()
            if not text:
                text = str(fragment.text or "").strip()
            result: Dict[str, object] = {
                "type": "text",
                "text": text[:320],
            }
            if fragment.bbox:
                result["bbox"] = [round(float(v), 2) for v in fragment.bbox]
            return result

        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        tool = str(payload.get("tool") or "stroke")
        point = self._compact_point(payload.get("points"))
        result = {
            "type": "stroke",
            "strokeType": tool,
        }
        if point:
            result["point"] = point
        return result

    @staticmethod
    def _compact_point(raw_points: object) -> Optional[List[float]]:
        if isinstance(raw_points, dict):
            try:
                x = float(raw_points.get("x"))
                y = float(raw_points.get("y"))
                return [round(x, 2), round(y, 2)]
            except (TypeError, ValueError):
                return None

        if not isinstance(raw_points, (list, tuple)):
            return None

        # Support both [x, y] and [[x, y], ...]. For sequences, keep the last
        # valid point as a compact "latest location" signal.
        if len(raw_points) >= 2 and not isinstance(raw_points[0], (list, tuple, dict)):
            try:
                x = float(raw_points[0])
                y = float(raw_points[1])
                return [round(x, 2), round(y, 2)]
            except (TypeError, ValueError):
                return None

        latest: Optional[List[float]] = None
        for candidate in raw_points:
            if isinstance(candidate, dict):
                try:
                    x = float(candidate.get("x"))
                    y = float(candidate.get("y"))
                except (TypeError, ValueError):
                    continue
                latest = [round(x, 2), round(y, 2)]
                continue
            if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
                continue
            try:
                x = float(candidate[0])
                y = float(candidate[1])
            except (TypeError, ValueError):
                continue
            latest = [round(x, 2), round(y, 2)]
        return latest

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
        targets: List[str] = []
        raw_targets = parsed.get("targetBlockIds")
        if raw_targets is None:
            raw_targets = parsed.get("targetIds")
        if isinstance(raw_targets, list):
            for item in raw_targets:
                if isinstance(item, str):
                    targets.append(item)
                    continue
                if isinstance(item, dict):
                    candidate = (
                        item.get("id")
                        or item.get("targetId")
                        or item.get("blockId")
                        or item.get("groupId")
                    )
                    if candidate:
                        targets.append(str(candidate))
        raw_target_groups = parsed.get("targetGroupIds")
        if isinstance(raw_target_groups, list):
            for item in raw_target_groups:
                if isinstance(item, str):
                    targets.append(item)
        comment = parsed.get("comment")

        raw_next_hint = (
            parsed.get("nextStepHint")
            or parsed.get("next_step_hint")
            or parsed.get("nextContentHint")
            or parsed.get("next_content_hint")
        )
        next_step_hint: Optional[str] = None
        if raw_next_hint is not None:
            next_text = str(raw_next_hint).strip()
            if next_text:
                next_step_hint = next_text[:240]

        return ExecutionPlan(
            action=action,
            target_block_ids=targets,
            comment=comment,
            next_step_hint=next_step_hint,
        )

    def _resolve_targets(
        self,
        targets: List[str],
        summaries: Dict[str, Dict[str, str]],
    ) -> List[str]:
        if not targets:
            return []
        label_to_id: Dict[str, str] = {}
        for block_id, info in summaries.items():
            label = info.get("label")
            if label:
                clean = label.strip()
                if clean:
                    label_to_id[clean] = block_id
                    label_to_id[clean.lower()] = block_id
        block_ids = set(self.block_manager.state.blocks.keys())
        group_ids = set(self.block_manager.state.groups.keys())
        resolved: List[str] = []
        seen: Set[str] = set()
        for target in targets:
            token = self._normalize_user_input(str(target))
            token = token.strip("[](){}\"'`").rstrip(",.;")
            if not token:
                continue
            candidates = [token]
            if ":" in token:
                _, suffix = token.split(":", 1)
                suffix = suffix.strip()
                if suffix:
                    candidates.append(suffix)
            for candidate in candidates:
                if candidate in block_ids or candidate in group_ids:
                    if candidate not in seen:
                        resolved.append(candidate)
                        seen.add(candidate)
                    break
            else:
                lookup = label_to_id.get(token)
                if lookup and lookup not in seen:
                    resolved.append(lookup)
                    seen.add(lookup)
        return resolved

    @staticmethod
    def _normalize_user_input(user_input: str) -> str:
        return " ".join(str(user_input or "").split()).strip()

    def _latest_block_id(self) -> Optional[str]:
        blocks = self.block_manager.state.list_blocks()
        if not blocks:
            return None
        latest = max(blocks, key=lambda block: getattr(block, "updated_at", None) or datetime.min)
        return latest.block_id

    def _update_context(self, main_block_id: Optional[str], plan: ExecutionPlan) -> None:
        """
        Update orchestrator context according to the plan.

        Semantics after this patch:
        - SWITCH:
            * Replace the active set with target_block_ids.
            * Move main focus to the first target.
        - OPEN_RELATED:
            * Add target_block_ids into the active set (union).
            * Keep current main focus (do NOT force-switch focus).
        - CLOSE:
            * Remove target_block_ids from the active set.
            * Keep current main focus (even if the closed block
              was the main focus; higher-level logic may resolve that).
        - CONTINUE / NOOP:
            * No change to active set, no change to main focus.
        """

        action = (plan.action or "").upper()
        targets = list(plan.target_block_ids or [])

        if action == "SWITCH":
            # Hard context switch: we are now talking about these block(s).
            self.context.active_block_ids = list(targets)
            # Switch main focus to the first specified target, if any.
            if targets:
                self.context.main_block_id = targets[0]

        elif action in {"OPEN", "OPEN_RELATED"}:
            # Soft-open additional related blocks, but do NOT steal focus.
            current = set(self.context.active_block_ids)
            current.update(targets)
            self.context.active_block_ids = list(current)
            # main_block_id is intentionally NOT changed here.

        elif action == "CLOSE":
            # Just remove these blocks from the active set.
            remaining = [
                bid for bid in self.context.active_block_ids
                if bid not in targets
            ]
            self.context.active_block_ids = remaining
            # Do NOT force main_block_id to a closed block.
            # We leave main_block_id as-is.

        # CONTINUE / NOOP:
        #   No modification to active_block_ids or main_block_id.

        # Fallback: if we still don't have a main_block_id set at all,
        #           try to ensure we keep at least *some* focus.
        if not getattr(self.context, "main_block_id", None):
            if action == "SWITCH" and targets:
                # already handled in SWITCH branch, but keep it safe
                self.context.main_block_id = targets[0]
            elif main_block_id:
                self.context.main_block_id = main_block_id

system_prompt = (
    "You are an interactive whiteboard orchestrator. "
    "Read the latest user message and decide what should be included. "
    "Always return JSON of the form {\"action\": ..., \"targetBlockIds\": [...], "
    "\"comment\": \"...\", \"nextStepHint\": \"...\"}. targetBlockIds may contain block IDs and group IDs.\n\n"
    "Allowed actions:\n"
    "- CONTINUE: The content provided is what you need to complete the task. Put what you need into targetBlockIds\n"
    "- NOOP: Nothing should happen; acknowledge but take no action.\n"
    "- SWITCH: Move the focus to the listed context IDs (block/group). After switching, orchestration runs again.\n"
    "- OPEN_RELATED: Other context IDs are related and needed to add to the context. "
    "Add the listed IDs to the active set (do not steal focus). You can ONLY use existing IDs.\n"
    "- CLOSE: Remove the listed IDs from the active set.\n\n"
    "Rules:\n"
    "1. Return valid JSON only; no markdown or code fences.\n"
    "2. Include a short human-readable explanation in `comment`.\n"
    "3. Add `nextStepHint` as one concise sentence or paragraph for the next generation focus.\n"
    "4. Only reference IDs that appear in RELATED BLOCKS / RELATED GROUPS / LATEST_CONTEXT.\n"
    "5. If LATEST_CONTEXT is present, treat it as the freshest change signal.\n"
    "6. If uncertain, choose NOOP.\n\n"
    "Your answer must contain only the JSON object, with no extra text."
)
