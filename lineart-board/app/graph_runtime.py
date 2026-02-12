from __future__ import annotations

import base64
import hashlib
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
from semantic_graph.vision import _bbox_overlap_ratio

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


def _env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    value = default
    if raw is not None and str(raw).strip() != "":
        try:
            value = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    value = default
    if raw is not None and str(raw).strip() != "":
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


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


VISION_SYSTEM_PROMPT = (
    "You are a diagram-understanding assistant for a collaborative canvas. "
    "Given a group of stroke fragments plus the nearby blocks, decide whether the strokes "
    "should be merged into an existing block or promoted as a brand new diagram block. "
    "Always return JSON: {\"results\": [{\"decision\": \"merge_block\"|\"new_block\", "
    "\"target_block_id\": str?, \"label\": str?, \"summary\": str?, "
    "\"relationships\": [{\"type\": str, \"target\": str, \"score\": float?}]?}]}. "
    "Only reference block IDs provided in the context. If unsure, prefer \"new_block\" with "
    "a cautious summary."
)


class LLMVisionBackend(VisionBackend):
    def __init__(self, model: Optional[str] = None, *, max_tokens: int = 600) -> None:
        self.model = model or os.getenv("VISION_MODEL") or DEFAULT_LLM_MODEL
        self.max_tokens = max_tokens

    def analyze(self, payload: VisionPayload) -> List[VisionResult]:
        context = self._build_context(payload)
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        try:
            parsed, _ = call_chat_completions(messages, model=self.model, max_tokens=self.max_tokens)
        except Exception:
            return []
        results_raw = []
        if isinstance(parsed, dict):
            if isinstance(parsed.get("results"), list):
                results_raw = parsed["results"]
            else:
                results_raw = parsed.get("items") or []
        elif isinstance(parsed, list):
            results_raw = parsed
        results: List[VisionResult] = []
        for item in results_raw or []:
            try:
                results.append(
                    VisionResult(
                        kind=str(item.get("kind") or ""),
                        decision=item.get("decision") or item.get("kind"),
                        label=item.get("label"),
                        stroke_fragment_ids=list(item.get("stroke_fragment_ids") or payload.stroke_fragment_ids),
                        target_fragment_id=item.get("target_fragment_id"),
                        target_block_id=item.get("target_block_id"),
                        confidence=_safe_float(item.get("confidence")),
                        summary=item.get("summary"),
                        relationships=item.get("relationships"),
                        extra={k: v for k, v in item.items() if k not in {"kind", "decision", "label", "stroke_fragment_ids", "target_fragment_id", "target_block_id", "confidence", "summary", "relationships"}},
                    )
                )
            except Exception:
                continue
        return results

    def _build_context(self, payload: VisionPayload) -> Dict[str, object]:
        meta = payload.metadata[0] if payload.metadata else {}
        return {
            "group": {
                "id": meta.get("group_id"),
                "reason": meta.get("reason"),
                "bbox": meta.get("bbox"),
                "count": meta.get("count"),
            },
            "strokes": payload.fragments,
            "candidateBlocks": payload.candidate_blocks,
        }


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class GraphIngestResult:
    new_fragments: List[str]
    promoted_blocks: List[str]


@dataclass
class CanvasSnapshot:
    image_bytes: bytes
    mime: str
    width: int
    height: int
    bbox: Optional[tuple[float, float, float, float]]
    updated_at: datetime


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
        vision_stroke_threshold = _env_int("GRAPH_VISION_STROKE_THRESHOLD", 6, minimum=1)
        vision_spatial_threshold = _env_float("GRAPH_VISION_SPATIAL_THRESHOLD", 280.0, minimum=0.0)
        vision_auto_promote_confidence = _env_float(
            "GRAPH_VISION_AUTO_PROMOTE_CONFIDENCE",
            0.92,
            minimum=0.0,
            maximum=1.0,
        )
        block_group_distance_threshold = _env_float("GRAPH_BLOCK_GROUP_DISTANCE_THRESHOLD", 0.45, minimum=0.0)
        block_block_distance_threshold = _env_float("GRAPH_BLOCK_BLOCK_DISTANCE_THRESHOLD", 0.40, minimum=0.0)
        block_auto_promote_group_size = _env_int("GRAPH_BLOCK_AUTO_PROMOTE_GROUP_SIZE", 7, minimum=1)

        self.state = GraphState()
        self.embedder = OpenAIEmbedder(model=embed_model)
        self.summarizer = LLMBlockSummarizer(model=summary_model or DEFAULT_SUMMARY_MODEL)
        self.cluster_logger = ClusterLogger(session_id=session_id)
        self.block_manager = BlockManager(
            state=self.state,
            embedder=self.embedder,
            summarizer=self.summarizer,
            group_distance_threshold=block_group_distance_threshold,
            block_distance_threshold=block_block_distance_threshold,
            canvas_size=(float(width), float(height)),
            auto_promote_group_size=block_auto_promote_group_size,
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
        self.vision_backend = LLMVisionBackend(model=os.getenv("VISION_MODEL"))
        self.vision = VisionGrouper(
            self.block_manager,
            backend=self.vision_backend,
            stroke_threshold=vision_stroke_threshold,
            auto_promote_confidence=vision_auto_promote_confidence,
            spatial_threshold=vision_spatial_threshold,
        )
        self.context = OrchestratorContext()
        self._seen_fragment_ids: Set[str] = set()
        self._fragment_signatures: Dict[str, str] = {}
        self._latest_canvas_snapshot: Optional[CanvasSnapshot] = None

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
                if (
                    isinstance(stroke, dict)
                    and fragment.fragment_id not in self._fragment_signatures
                ):
                    self._fragment_signatures[fragment.fragment_id] = self._stroke_signature(stroke)
                continue
            self._seen_fragment_ids.add(fragment.fragment_id)
            assignment = self.block_manager.register_fragment(fragment)
            if isinstance(stroke, dict):
                self._fragment_signatures[fragment.fragment_id] = self._stroke_signature(stroke)
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
            if fragment.fragment_type == FragmentType.STROKE:
                ready_payloads = self.vision.ingest_fragment(fragment, reason="auto")
                if ready_payloads:
                    self._process_vision_batches(ready_payloads, reason="auto")
        return GraphIngestResult(new_fragments=new_fragments, promoted_blocks=promoted_blocks)

    def sync_strokes_snapshot(self, strokes: Iterable[Dict[str, object]]) -> GraphIngestResult:
        """
        Reconcile graph state against the latest full-canvas stroke snapshot.
        - Remove fragments that disappeared from the snapshot.
        - Re-ingest fragments whose content changed under the same stroke id.
        - Keep existing call flow unchanged for callers.
        """
        incoming_by_id: Dict[str, Dict[str, object]] = {}
        ordered_ids: List[str] = []
        for stroke in strokes or []:
            if not isinstance(stroke, dict):
                continue
            stroke_id = str(stroke.get("id") or "").strip()
            if not stroke_id:
                continue
            if stroke_id not in incoming_by_id:
                ordered_ids.append(stroke_id)
            incoming_by_id[stroke_id] = stroke

        incoming_ids = set(incoming_by_id.keys())
        existing_ids = set(self.state.fragments.keys())
        removed_ids = existing_ids - incoming_ids

        updated_ids: Set[str] = set()
        common_ids = incoming_ids & existing_ids
        for fid in common_ids:
            previous = self._fragment_signatures.get(fid)
            if not previous:
                continue
            current = self._stroke_signature(incoming_by_id[fid])
            if current != previous:
                updated_ids.add(fid)

        to_remove = removed_ids | updated_ids
        if to_remove:
            self._remove_fragments(to_remove)

        to_ingest_ids = (incoming_ids - existing_ids) | updated_ids
        ingest_batch = [incoming_by_id[fid] for fid in ordered_ids if fid in to_ingest_ids]
        result = (
            self.ingest_strokes(ingest_batch)
            if ingest_batch
            else GraphIngestResult(new_fragments=[], promoted_blocks=[])
        )

        stale_signature_ids = set(self._fragment_signatures.keys()) - incoming_ids
        for fid in stale_signature_ids:
            self._fragment_signatures.pop(fid, None)

        for fid in incoming_ids:
            self._fragment_signatures[fid] = self._stroke_signature(incoming_by_id[fid])

        self._seen_fragment_ids.intersection_update(set(self.state.fragments.keys()))
        return result

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

    def update_canvas_snapshot(self, snapshot: Dict[str, object]) -> None:
        data_b64 = snapshot.get("data")
        if not data_b64:
            return
        try:
            image_bytes = base64.b64decode(data_b64)
        except Exception as exc:
            print("[graph] decode snapshot failed:", exc)
            return
        bbox_raw = snapshot.get("bbox")
        bbox: Optional[tuple[float, float, float, float]] = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except Exception:
                bbox = None
        mime = str(snapshot.get("mime") or "image/jpeg")
        width = int(snapshot.get("width") or 0)
        height = int(snapshot.get("height") or 0)
        self._latest_canvas_snapshot = CanvasSnapshot(
            image_bytes=image_bytes,
            mime=mime,
            width=width,
            height=height,
            bbox=bbox,
            updated_at=datetime.utcnow(),
        )

    def _candidate_blocks_for_bbox(self, bbox: Optional[Tuple[float, float, float, float]]) -> List[Dict[str, object]]:
        scored = []
        for block in self.block_manager.state.list_blocks():
            block_bbox = getattr(block, "position", None)
            overlap = 0.0
            if bbox and block_bbox:
                overlap = _bbox_overlap_ratio(bbox, block_bbox)
            scored.append((overlap, getattr(block, "updated_at", datetime.min), block))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result: List[Dict[str, object]] = []
        for overlap, _, block in scored[:6]:
            result.append(
                {
                    "blockId": block.block_id,
                    "label": block.label,
                    "summary": block.summary,
                    "bbox": block.position,
                    "overlap": overlap,
                }
            )
        return result

    def run_conversation(
        self,
        user_input: str,
        *,
        focus_block_id: Optional[str] = None,
        focus_fragment_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        pending_payloads = self.vision.flush_groups(
            reason="ask_ai",
            ready_only=True,
            min_size=self.vision.stroke_threshold,
            stale_seconds=8,
        )
        if pending_payloads:
            self._process_vision_batches(pending_payloads, reason="ask_ai")
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
        action_upper = (plan.action or "").upper()
        if (
            action_upper in {"NOOP", "CONTINUE"}
            and not focus_context.main_block_id
            and not focus_context.active_block_ids
        ):
            return {
                "plan": {
                    "action": plan.action,
                    "targetBlockIds": plan.target_block_ids,
                    "comment": plan.comment,
                    "nextStepHint": plan.next_step_hint,
                },
                "payload": {
                    "version": 1,
                    "intent": "noop",
                    "strokes": [],
                },
            }
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
                "nextStepHint": plan.next_step_hint,
            },
            "payload": response,
        }

    # ----------------------------- helpers ----------------------------- #

    def _process_vision_batches(self, payloads: Sequence[VisionPayload], *, reason: str) -> None:
        for payload in payloads:
            self._enrich_vision_payload(payload)
            try:
                results = self.vision.process(payload)
                if self.cluster_logger:
                    self.cluster_logger.log(
                        "vision_process",
                        {
                            "reason": reason,
                            "stroke_ids": payload.stroke_fragment_ids,
                            "results": [r.__dict__ for r in results],
                        },
                    )
            except Exception as exc:
                print(f"[vision] failed to process payload ({reason}): {exc}")

    def _enrich_vision_payload(self, payload: VisionPayload) -> None:
        fragments: List[Dict[str, object]] = []
        for fid in payload.stroke_fragment_ids:
            fragment = self.block_manager.state.fragments.get(fid)
            if not fragment:
                continue
            frag_payload = fragment.payload if isinstance(fragment.payload, dict) else {}
            fragments.append(
                {
                    "id": fragment.fragment_id,
                    "bbox": fragment.bbox,
                    "timestamp": fragment.timestamp.isoformat() if fragment.timestamp else None,
                    "tool": frag_payload.get("tool"),
                    "style": frag_payload.get("style"),
                    "points": frag_payload.get("points"),
                }
            )
        payload.fragments = fragments
        bbox = None
        if payload.metadata:
            data = payload.metadata[0]
            bbox = tuple(data.get("bbox") or []) if isinstance(data.get("bbox"), (list, tuple)) and len(data.get("bbox")) == 4 else None
        payload.candidate_blocks = self._candidate_blocks_for_bbox(bbox)
        if self._latest_canvas_snapshot:
            payload.image_bytes = self._latest_canvas_snapshot.image_bytes
            payload.image_mime = self._latest_canvas_snapshot.mime
            payload.metadata.append(
                {
                    "snapshot": {
                        "bbox": self._latest_canvas_snapshot.bbox,
                        "width": self._latest_canvas_snapshot.width,
                        "height": self._latest_canvas_snapshot.height,
                        "capturedAt": self._latest_canvas_snapshot.updated_at.isoformat(),
                    }
                }
            )

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
        if fragment_type == FragmentType.TEXT:
            is_heading = self._is_heading_meta(meta_payload, stroke.get("style"))
            graph_meta = dict(payload.get("graph") or {})
            graph_meta["isHeading"] = is_heading
            payload["graph"] = graph_meta
        return Fragment(
            fragment_id=stroke_id,
            fragment_type=fragment_type,
            bbox=bbox,
            text=text or None,
            timestamp=timestamp,
            payload=payload,
        )

    def _is_heading_meta(self, meta: Dict[str, object], style: object) -> bool:
        font_size = meta.get("fontSize") if isinstance(meta, dict) else None
        font_weight = meta.get("fontWeight") if isinstance(meta, dict) else None
        if font_size is None and isinstance(style, dict):
            font_size = style.get("fontSize")
        if font_weight is None and isinstance(style, dict):
            font_weight = style.get("fontWeight")
        try:
            size_val = float(font_size)
        except (TypeError, ValueError):
            size_val = None
        weight_val = 0
        if isinstance(font_weight, str):
            ft = font_weight.strip().lower()
            if ft.isdigit():
                weight_val = int(ft)
            elif ft in {"bold", "heavy"}:
                weight_val = 700
        elif isinstance(font_weight, (int, float)):
            weight_val = int(font_weight)
        role = (meta.get("role") if isinstance(meta, dict) else None) or ""
        role = str(role).lower()
        if role in {"title", "heading", "header"}:
            return True
        if size_val is not None and size_val >= 28:
            return True
        if weight_val >= 600:
            return True
        return False

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

    def _stroke_signature(self, stroke: Dict[str, object]) -> str:
        payload = {
            "tool": stroke.get("tool"),
            "points": stroke.get("points"),
            "style": stroke.get("style"),
            "meta": stroke.get("meta"),
        }
        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            body = repr(payload)
        return hashlib.sha1(body.encode("utf-8", "replace")).hexdigest()

    def _remove_fragments(self, fragment_ids: Set[str]) -> None:
        if not fragment_ids:
            return
        now = datetime.utcnow()
        for fragment_id in fragment_ids:
            fragment = self.state.fragments.pop(fragment_id, None)
            self._seen_fragment_ids.discard(fragment_id)
            self._fragment_signatures.pop(fragment_id, None)
            self.block_manager.remove_unlabeled_strokes([fragment_id])

            group_id = self.block_manager.get_group_id_for_fragment(fragment_id)
            candidate_group_ids: Set[str] = set()
            if group_id:
                candidate_group_ids.add(group_id)
            for gid, group in self.state.groups.items():
                if fragment_id in group.members:
                    candidate_group_ids.add(gid)
            for gid in candidate_group_ids:
                group = self.state.groups.get(gid)
                if not group:
                    continue
                group.members.discard(fragment_id)
                group.updated_at = now
                if not group.members:
                    self.state.remove_group(gid)
                    self.block_manager._group_touch_counts.pop(gid, None)  # pylint: disable=protected-access

            block_id = self.block_manager.get_block_id_for_fragment(fragment_id)
            candidate_block_ids: Set[str] = set()
            if block_id:
                candidate_block_ids.add(block_id)
            for bid, block in self.state.blocks.items():
                if fragment_id in block.contents:
                    candidate_block_ids.add(bid)
            for bid in candidate_block_ids:
                block = self.state.blocks.get(bid)
                if not block:
                    continue
                block.contents.discard(fragment_id)
                if fragment:
                    block.character_count = max(
                        0,
                        block.character_count - self.block_manager._fragment_text_length(fragment),  # pylint: disable=protected-access
                    )
                block.position = self.block_manager._refresh_block_bbox(block)  # pylint: disable=protected-access
                block.updated_at = now
                if not block.contents:
                    self.state.remove_block(bid)
                    self.block_manager._block_incoming_counts.pop(bid, None)  # pylint: disable=protected-access
                    self.vision._diagram_blocks.pop(bid, None)  # pylint: disable=protected-access
                    if self.orchestrator.context.main_block_id == bid:
                        self.orchestrator.context.main_block_id = None
                    if bid in self.orchestrator.context.active_block_ids:
                        self.orchestrator.context.active_block_ids = [
                            item for item in self.orchestrator.context.active_block_ids if item != bid
                        ]

            self.block_manager.orphan_fragment(fragment_id)

        for gid in list(self.vision._pending_groups.keys()):  # pylint: disable=protected-access
            pending = self.vision._pending_groups.get(gid)  # pylint: disable=protected-access
            if not pending:
                continue
            pending.stroke_ids = [fid for fid in pending.stroke_ids if fid not in fragment_ids]
            if not pending.stroke_ids:
                self.vision._pending_groups.pop(gid, None)  # pylint: disable=protected-access

