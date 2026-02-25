from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
from semantic_graph.markdown_text import markdown_to_semantic_text
from semantic_graph.vision import _bbox_overlap_ratio

from app.embedding_client import embed_text
from app.llm_client import call_chat_completions
from app.cluster_logging import ClusterLogger
from app.agents import (
    LLMBlockSummarizer,
    LLMPlanBackend,
    LLMVisionBackend,
    normalize_vision_image_mode as _normalize_vision_image_mode_agent,
)
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on", "y"}:
        return True
    if token in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _normalize_group_promote_mode(value: Optional[str]) -> str:
    token = str(value or "").strip().lower()
    if token in {"heuristic", "hybrid", "llm"}:
        return token
    return "heuristic"


def _normalize_vision_image_mode(value: Optional[str]) -> str:
    return _normalize_vision_image_mode_agent(value)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 1.0
    dims = min(len(a), len(b))
    if dims <= 0:
        return 1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for idx in range(dims):
        va = float(a[idx])
        vb = float(b[idx])
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 1.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def _merge_bbox_local(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


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


@dataclass
class GroupPromotionDecision:
    group_id: str
    allow: bool
    score: float
    reasons: List[str]
    metrics: Dict[str, object]
    hard_reject: bool = False
    source: str = "heuristic"


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
        block_summary_refresh_ratio = _env_float(
            "GRAPH_BLOCK_SUMMARY_REFRESH_RATIO",
            0.8,
            minimum=0.0,
        )
        block_summary_refresh_interval_seconds = _env_int(
            "GRAPH_BLOCK_SUMMARY_REFRESH_INTERVAL_SECONDS",
            1800,
            minimum=1,
        )
        self.agent_group_promote_enabled = _env_bool("GRAPH_AGENT_GROUP_PROMOTE_ENABLED", True)
        self.agent_group_promote_min_members = _env_int("GRAPH_AGENT_GROUP_PROMOTE_MIN_MEMBERS", 4, minimum=1)
        self.agent_group_promote_min_text_members = _env_int(
            "GRAPH_AGENT_GROUP_PROMOTE_MIN_TEXT_MEMBERS",
            2,
            minimum=0,
        )
        self.agent_group_promote_min_text_chars = _env_int(
            "GRAPH_AGENT_GROUP_PROMOTE_MIN_TEXT_CHARS",
            20,
            minimum=0,
        )
        self.agent_group_promote_min_age_seconds = _env_int(
            "GRAPH_AGENT_GROUP_PROMOTE_MIN_AGE_SECONDS",
            2,
            minimum=0,
        )
        self.agent_group_promote_max_diag_ratio = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_MAX_DIAG_RATIO",
            0.7,
            minimum=0.0,
        )
        self.agent_group_promote_max_semantic_distance = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_MAX_SEMANTIC_DISTANCE",
            0.34,
            minimum=0.0,
            maximum=2.0,
        )
        self.agent_group_promote_min_distance_to_block = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_MIN_DISTANCE_TO_BLOCK",
            0.12,
            minimum=0.0,
            maximum=2.0,
        )
        self.agent_group_promote_max_overlap_to_block = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_MAX_OVERLAP_TO_BLOCK",
            0.55,
            minimum=0.0,
            maximum=1.0,
        )
        self.agent_group_promote_min_score = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_MIN_SCORE",
            5.0,
            minimum=0.0,
        )
        self.agent_group_promote_mode = _normalize_group_promote_mode(
            os.getenv("GRAPH_AGENT_GROUP_PROMOTE_MODE", "heuristic")
        )
        self.vision_image_mode = _normalize_vision_image_mode(
            os.getenv("GRAPH_VISION_IMAGE_MODE", "auto")
        )
        self.agent_group_promote_review_margin = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_REVIEW_MARGIN",
            1.0,
            minimum=0.0,
        )
        self.agent_group_promote_review_on_hard_reject = _env_bool(
            "GRAPH_AGENT_GROUP_PROMOTE_REVIEW_ON_HARD_REJECT",
            False,
        )
        self.agent_group_promote_review_model = (
            os.getenv("GRAPH_AGENT_GROUP_PROMOTE_MODEL")
            or os.getenv("GRAPH_PLAN_MODEL")
            or DEFAULT_PLAN_MODEL
        )
        self.agent_group_promote_review_max_tokens = _env_int(
            "GRAPH_AGENT_GROUP_PROMOTE_MAX_TOKENS",
            420,
            minimum=120,
        )
        self.agent_group_promote_review_temperature = _env_float(
            "GRAPH_AGENT_GROUP_PROMOTE_TEMPERATURE",
            0.0,
            minimum=0.0,
            maximum=1.0,
        )

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
            summary_refresh_ratio=block_summary_refresh_ratio,
            summary_refresh_interval=timedelta(seconds=block_summary_refresh_interval_seconds),
            canvas_size=(float(width), float(height)),
            auto_promote_group_size=block_auto_promote_group_size,
            cluster_logger=self.cluster_logger,
            allow_ai_block_intent_create=True,
            allow_ai_block_target_assign=True,
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
        self.vision_backend = LLMVisionBackend(
            model=os.getenv("VISION_MODEL"),
            image_mode=self.vision_image_mode,
        )
        self.vision = VisionGrouper(
            self.block_manager,
            backend=self.vision_backend,
            stroke_threshold=vision_stroke_threshold,
            auto_promote_confidence=vision_auto_promote_confidence,
            spatial_threshold=vision_spatial_threshold,
            manual_pending_promotion=True,
        )
        self.context = OrchestratorContext()
        self._seen_fragment_ids: Set[str] = set()
        self._fragment_signatures: Dict[str, str] = {}
        self._latest_canvas_snapshot: Optional[CanvasSnapshot] = None

    def _call_full_backend(self, messages: List[Dict[str, str]], *, mode: Optional[str] = None) -> Dict[str, object]:
        parsed, dbg = call_chat_completions(messages, model=DEFAULT_LLM_MODEL, max_tokens=9000)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except Exception as exc:
                raise RuntimeError(f"LLM returned non-JSON response: {parsed!r}") from exc
        raise RuntimeError(f"Unexpected LLM response: {type(parsed)!r}")

    def set_group_promotion_mode(self, mode: Optional[str]) -> None:
        normalized = _normalize_group_promote_mode(mode)
        if normalized == self.agent_group_promote_mode:
            return
        self.agent_group_promote_mode = normalized
        if self.cluster_logger:
            try:
                self.cluster_logger.log("group_promotion_mode_updated", {"mode": normalized})
            except Exception:
                pass

    def set_vision_image_mode(self, mode: Optional[str]) -> None:
        normalized = _normalize_vision_image_mode(mode)
        if normalized == self.vision_image_mode:
            return
        self.vision_image_mode = normalized
        if hasattr(self.vision_backend, "set_image_mode"):
            try:
                self.vision_backend.set_image_mode(normalized)  # type: ignore[attr-defined]
            except Exception:
                pass
        if self.cluster_logger:
            try:
                self.cluster_logger.log("vision_image_mode_updated", {"mode": normalized})
            except Exception:
                pass

    def ingest_strokes(self, strokes: Iterable[Dict[str, object]]) -> GraphIngestResult:
        new_fragments: List[str] = []
        promoted_blocks: List[str] = []
        proposal_key_to_block: Dict[str, str] = {}
        for stroke in strokes or []:
            raw_stroke = stroke if isinstance(stroke, dict) else {}
            proposal_key = self._stroke_graph_proposal_key(raw_stroke)
            stroke_for_ingest = raw_stroke

            mapped_block_id = proposal_key_to_block.get(proposal_key) if proposal_key else None
            if mapped_block_id:
                rebound = self._stroke_with_proposal_target(raw_stroke, mapped_block_id)
                if rebound is not None:
                    stroke_for_ingest = rebound
                    if self.cluster_logger:
                        try:
                            self.cluster_logger.log(
                                "proposal_key_target_applied",
                                {
                                    "proposalKey": proposal_key,
                                    "targetBlockId": mapped_block_id,
                                    "strokeId": str(raw_stroke.get("id") or ""),
                                    "tool": str(raw_stroke.get("tool") or ""),
                                },
                            )
                        except Exception:
                            pass

            fragment = self._stroke_to_fragment(stroke_for_ingest)
            if fragment is None:
                continue
            if fragment.fragment_id in self._seen_fragment_ids:
                continue
            if fragment.fragment_id in self.state.fragments:
                self._seen_fragment_ids.add(fragment.fragment_id)
                if (
                    isinstance(raw_stroke, dict)
                    and fragment.fragment_id not in self._fragment_signatures
                ):
                    self._fragment_signatures[fragment.fragment_id] = self._stroke_signature(raw_stroke)
                continue
            self._seen_fragment_ids.add(fragment.fragment_id)
            assignment = self.block_manager.register_fragment(fragment)

            if proposal_key and assignment.block_id:
                create_intent = self._stroke_graph_is_block_create(raw_stroke)
                explicit_target = self._stroke_graph_target_block_id(raw_stroke)
                if create_intent or explicit_target:
                    proposal_key_to_block[proposal_key] = assignment.block_id
                    if self.cluster_logger:
                        try:
                            self.cluster_logger.log(
                                "proposal_key_anchor_resolved",
                                {
                                    "proposalKey": proposal_key,
                                    "blockId": assignment.block_id,
                                    "strokeId": fragment.fragment_id,
                                    "fromCreateIntent": create_intent,
                                    "fromExplicitTarget": bool(explicit_target),
                                },
                            )
                        except Exception:
                            pass

            if isinstance(raw_stroke, dict):
                self._fragment_signatures[fragment.fragment_id] = self._stroke_signature(raw_stroke)
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
                            "stroke_tool": str(raw_stroke.get("tool")) if isinstance(raw_stroke, dict) else None,
                            "proposal_key": proposal_key,
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
            "visionPendingGroups": self.vision.list_pending_groups(),
        }

    def promote_group_now(self, group_id: str) -> Optional[Block]:
        try:
            block = self.block_manager.mark_group_stable(group_id)
        except GroupNotFoundError:
            return None
        return block

    def promote_vision_pending_group_now(self, group_id: str) -> bool:
        payload = self.vision.pop_pending_group_payload(group_id, reason="manual_promote")
        if payload is None:
            return False
        self._enrich_vision_payload(payload)
        try:
            results = self.vision.process(payload, force_promote=True)
            if self.cluster_logger:
                self.cluster_logger.log(
                    "vision_manual_promote",
                    {
                        "group_id": group_id,
                        "stroke_ids": list(payload.stroke_fragment_ids),
                        "results": [r.__dict__ for r in results],
                    },
                )
        except Exception as exc:
            print(f"[vision] manual promote failed ({group_id}): {exc}")
            return False
        return True

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
        prefer_explanatory_drawing: Optional[bool] = None,
    ) -> Dict[str, object]:
        if not getattr(self.vision, "manual_pending_promotion", False):
            pending_payloads = self.vision.flush_groups(
                reason="ask_ai",
                ready_only=True,
                min_size=self.vision.stroke_threshold,
                stale_seconds=8,
            )
            if pending_payloads:
                self._process_vision_batches(pending_payloads, reason="ask_ai")

        # Re-run planner after SWITCH so context can stabilize on the new focus.
        # Hard cap avoids planner ping-pong loops.
        max_planner_passes = 3
        plan = None
        loop_focus_block_id = focus_block_id
        loop_focus_fragment_id = focus_fragment_id
        for attempt in range(max_planner_passes):
            plan = self.orchestrator.generate_plan(
                user_input,
                focus_block_id=loop_focus_block_id,
                focus_fragment_id=loop_focus_fragment_id,
            )
            if (plan.action or "").upper() != "SWITCH":
                break
            # From the second pass onward, rely on orchestrator-updated context.
            loop_focus_fragment_id = None
            current_focus = self.orchestrator.context.main_block_id
            if current_focus and current_focus in self.state.blocks:
                loop_focus_block_id = current_focus
            else:
                loop_focus_block_id = None
            if self.cluster_logger:
                try:
                    self.cluster_logger.log(
                        "planner_switch_retry",
                        {
                            "attempt": attempt + 1,
                            "max_attempts": max_planner_passes,
                            "target_ids": list(plan.target_block_ids or []),
                            "next_focus_block_id": loop_focus_block_id,
                        },
                    )
                except Exception:
                    pass

        if plan is None:
            plan = self.orchestrator.generate_plan(
                user_input,
                focus_block_id=focus_block_id,
                focus_fragment_id=focus_fragment_id,
            )
        self._maybe_promote_plan_groups(plan, user_input=user_input)
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
        if action_upper == "NOOP":
            placeholder_payload = self._build_noop_placeholder_payload(plan.next_step_hint)
            return {
                "plan": {
                    "action": plan.action,
                    "targetBlockIds": plan.target_block_ids,
                    "comment": plan.comment,
                    "nextStepHint": plan.next_step_hint,
                },
                "payload": placeholder_payload,
            }
        if (
            action_upper == "CONTINUE"
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
            prefer_explanatory_drawing=prefer_explanatory_drawing,
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

    def _maybe_promote_plan_groups(self, plan, *, user_input: str) -> None:
        if not self.agent_group_promote_enabled or plan is None:
            return

        raw_targets = list(getattr(plan, "target_block_ids", []) or [])
        if not raw_targets:
            return

        candidate_group_ids: List[str] = []
        seen_groups: Set[str] = set()
        for context_id in raw_targets:
            if context_id in self.state.groups and context_id not in seen_groups:
                candidate_group_ids.append(context_id)
                seen_groups.add(context_id)

        if not candidate_group_ids:
            return

        promoted_map: Dict[str, str] = {}
        for group_id in candidate_group_ids:
            decision = self._evaluate_group_for_promotion(group_id, user_input=user_input)
            decision = self._maybe_apply_group_promotion_review(
                group_id=group_id,
                heuristic_decision=decision,
                user_input=user_input,
            )
            if self.cluster_logger:
                try:
                    self.cluster_logger.log(
                        "group_promotion_candidate",
                        {
                            "group_id": group_id,
                            "allow": decision.allow,
                            "score": decision.score,
                            "reasons": decision.reasons,
                            "metrics": decision.metrics,
                            "source": decision.source,
                            "hard_reject": decision.hard_reject,
                        },
                    )
                except Exception:
                    pass

            if not decision.allow:
                continue

            try:
                promoted_block = self.block_manager.promote_group(group_id)
                promoted_map[group_id] = promoted_block.block_id
                if self.cluster_logger:
                    try:
                        self.cluster_logger.log(
                            "group_promotion_approved",
                            {
                                "group_id": group_id,
                                "block_id": promoted_block.block_id,
                                "score": decision.score,
                                "reasons": decision.reasons,
                                "metrics": decision.metrics,
                                "source": decision.source,
                            },
                        )
                    except Exception:
                        pass
            except Exception as exc:
                if self.cluster_logger:
                    try:
                        self.cluster_logger.log(
                            "group_promotion_failed",
                            {
                                "group_id": group_id,
                                "error": str(exc),
                                "score": decision.score,
                                "reasons": decision.reasons,
                                "source": decision.source,
                            },
                        )
                    except Exception:
                        pass

        if not promoted_map:
            return

        updated_targets: List[str] = []
        seen_targets: Set[str] = set()
        for context_id in raw_targets:
            mapped = promoted_map.get(context_id, context_id)
            if mapped in seen_targets:
                continue
            updated_targets.append(mapped)
            seen_targets.add(mapped)
        plan.target_block_ids = updated_targets
        self._remap_orchestrator_context_ids(promoted_map)

    def _evaluate_group_for_promotion(self, group_id: str, *, user_input: str) -> GroupPromotionDecision:
        group = self.state.groups.get(group_id)
        if not group:
            return GroupPromotionDecision(
                group_id=group_id,
                allow=False,
                score=0.0,
                reasons=["group_not_found"],
                metrics={},
                hard_reject=True,
                source="heuristic",
            )

        fragments: List[Fragment] = []
        text_fragments: List[Fragment] = []
        text_vectors: List[List[float]] = []
        text_chars = 0
        for fragment_id in group.members:
            fragment = self.state.fragments.get(fragment_id)
            if not fragment:
                continue
            fragments.append(fragment)
            if fragment.fragment_type == FragmentType.TEXT:
                text_fragments.append(fragment)
                text = self._fragment_text(fragment)
                text_chars += len(text)
                if fragment.feature_vec is not None:
                    text_vectors.append([float(v) for v in fragment.feature_vec])

        member_count = len(fragments)
        text_count = len(text_fragments)
        age_seconds = max(
            0.0,
            (datetime.utcnow() - (group.updated_at or datetime.utcnow())).total_seconds(),
        )
        group_bbox = self._group_bbox(fragments)
        canvas_w, canvas_h = self.block_manager.canvas_size or (1920.0, 1080.0)
        canvas_diag = math.hypot(float(canvas_w or 0.0), float(canvas_h or 0.0))
        diag_ratio: Optional[float] = None
        if group_bbox and canvas_diag > 1e-6:
            gx0, gy0, gx1, gy1 = group_bbox
            group_diag = math.hypot(max(0.0, gx1 - gx0), max(0.0, gy1 - gy0))
            diag_ratio = group_diag / canvas_diag

        group_vec = self._build_group_embedding(group, text_vectors)
        avg_semantic_distance: Optional[float] = None
        if group_vec is not None and text_vectors:
            distances = [_cosine_distance(vec, group_vec) for vec in text_vectors]
            if distances:
                avg_semantic_distance = sum(distances) / max(1, len(distances))

        nearest_block_id: Optional[str] = None
        nearest_block_distance: Optional[float] = None
        max_overlap_to_block = 0.0
        if group_vec is not None:
            for block in self.state.list_blocks():
                block_vec = self._block_embedding_from_contents(block.contents)
                if not block_vec:
                    continue
                distance = _cosine_distance(group_vec, block_vec)
                if nearest_block_distance is None or distance < nearest_block_distance:
                    nearest_block_distance = distance
                    nearest_block_id = block.block_id
        if group_bbox:
            for block in self.state.list_blocks():
                if not block.position:
                    continue
                overlap = _bbox_overlap_ratio(group_bbox, block.position)
                if overlap > max_overlap_to_block:
                    max_overlap_to_block = overlap

        reasons: List[str] = []
        score = 0.0
        hard_reject = False

        if member_count >= self.agent_group_promote_min_members:
            score += 2.0
            reasons.append("member_count_ok")
        else:
            reasons.append("member_count_too_small")
            hard_reject = True

        if text_count >= self.agent_group_promote_min_text_members:
            score += 1.0
            reasons.append("text_members_ok")
        elif self.agent_group_promote_min_text_members > 0:
            reasons.append("text_members_low")

        if text_chars >= self.agent_group_promote_min_text_chars:
            score += 1.0
            reasons.append("text_chars_ok")
        elif self.agent_group_promote_min_text_chars > 0:
            reasons.append("text_chars_low")

        if age_seconds >= self.agent_group_promote_min_age_seconds:
            score += 1.0
            reasons.append("stable_enough")
        else:
            reasons.append("too_fresh")

        if diag_ratio is None or diag_ratio <= self.agent_group_promote_max_diag_ratio:
            score += 1.0
            reasons.append("spatially_compact")
        else:
            reasons.append("spatially_wide")

        if avg_semantic_distance is not None:
            if avg_semantic_distance <= self.agent_group_promote_max_semantic_distance:
                score += 1.0
                reasons.append("semantic_cohesion_ok")
            else:
                reasons.append("semantic_cohesion_low")

        normalized_hint = (user_input or "").strip().lower()
        if normalized_hint:
            for token in ("expand", "organize", "summarize", "continue", "refine", "structure"):
                if token in normalized_hint:
                    score += 0.5
                    reasons.append("hint_supports_structuring")
                    break

        if (
            nearest_block_distance is not None
            and nearest_block_distance <= self.agent_group_promote_min_distance_to_block
            and max_overlap_to_block >= self.agent_group_promote_max_overlap_to_block
        ):
            hard_reject = True
            reasons.append("likely_already_covered_by_block")

        if text_count == 0 and member_count < (self.agent_group_promote_min_members + 2):
            hard_reject = True
            reasons.append("non_text_group_too_small")

        allow = (not hard_reject) and score >= self.agent_group_promote_min_score
        if allow:
            reasons.append("promote")
        else:
            reasons.append("defer")

        metrics: Dict[str, object] = {
            "member_count": member_count,
            "text_count": text_count,
            "text_chars": text_chars,
            "age_seconds": round(age_seconds, 3),
            "diag_ratio": round(diag_ratio, 4) if diag_ratio is not None else None,
            "avg_semantic_distance": round(avg_semantic_distance, 4) if avg_semantic_distance is not None else None,
            "nearest_block_id": nearest_block_id,
            "nearest_block_distance": round(nearest_block_distance, 4) if nearest_block_distance is not None else None,
            "max_overlap_to_block": round(max_overlap_to_block, 4),
            "score_threshold": self.agent_group_promote_min_score,
            "score_delta": round(score - self.agent_group_promote_min_score, 3),
            "hard_reject": hard_reject,
        }
        return GroupPromotionDecision(
            group_id=group_id,
            allow=allow,
            score=round(score, 3),
            reasons=reasons,
            metrics=metrics,
            hard_reject=hard_reject,
            source="heuristic",
        )

    def _maybe_apply_group_promotion_review(
        self,
        *,
        group_id: str,
        heuristic_decision: GroupPromotionDecision,
        user_input: str,
    ) -> GroupPromotionDecision:
        mode = self.agent_group_promote_mode
        if mode == "heuristic":
            return heuristic_decision

        should_review = self._should_use_group_promotion_review(
            mode=mode,
            decision=heuristic_decision,
        )
        if not should_review:
            metrics = dict(heuristic_decision.metrics)
            metrics["llm_review_used"] = False
            metrics["llm_review_skipped"] = True
            metrics["llm_review_mode"] = mode
            return GroupPromotionDecision(
                group_id=heuristic_decision.group_id,
                allow=heuristic_decision.allow,
                score=heuristic_decision.score,
                reasons=list(heuristic_decision.reasons),
                metrics=metrics,
                hard_reject=heuristic_decision.hard_reject,
                source=heuristic_decision.source,
            )

        reviewed = self._review_group_promotion_with_llm(
            group_id=group_id,
            heuristic_decision=heuristic_decision,
            user_input=user_input,
            mode=mode,
        )
        if reviewed is not None:
            return reviewed

        # LLM review failed: keep heuristic to avoid breaking the current flow.
        metrics = dict(heuristic_decision.metrics)
        metrics["llm_review_used"] = True
        metrics["llm_review_fallback"] = "heuristic"
        metrics["llm_review_mode"] = mode
        reasons = list(heuristic_decision.reasons) + ["llm_review_unavailable"]
        return GroupPromotionDecision(
            group_id=heuristic_decision.group_id,
            allow=heuristic_decision.allow,
            score=heuristic_decision.score,
            reasons=reasons,
            metrics=metrics,
            hard_reject=heuristic_decision.hard_reject,
            source="heuristic_fallback",
        )

    def _should_use_group_promotion_review(
        self,
        *,
        mode: str,
        decision: GroupPromotionDecision,
    ) -> bool:
        if mode == "llm":
            return True
        if mode != "hybrid":
            return False
        if decision.hard_reject and not self.agent_group_promote_review_on_hard_reject:
            return False
        threshold = float(self.agent_group_promote_min_score)
        margin = float(self.agent_group_promote_review_margin)
        delta = abs(float(decision.score) - threshold)
        if delta <= margin:
            return True
        if "semantic_cohesion_low" in decision.reasons and delta <= (margin + 0.5):
            return True
        if "likely_already_covered_by_block" in decision.reasons and delta <= (margin + 0.5):
            return True
        return False

    def _review_group_promotion_with_llm(
        self,
        *,
        group_id: str,
        heuristic_decision: GroupPromotionDecision,
        user_input: str,
        mode: str,
    ) -> Optional[GroupPromotionDecision]:
        group = self.state.groups.get(group_id)
        if not group:
            return None

        fragments: List[Fragment] = []
        text_vectors: List[List[float]] = []
        for fragment_id in group.members:
            fragment = self.state.fragments.get(fragment_id)
            if not fragment:
                continue
            fragments.append(fragment)
            if fragment.fragment_type == FragmentType.TEXT and fragment.feature_vec is not None:
                text_vectors.append([float(v) for v in fragment.feature_vec])

        group_bbox = self._group_bbox(fragments)
        group_vec = self._build_group_embedding(group, text_vectors)
        review_payload = self._build_group_promotion_review_payload(
            group_id=group_id,
            group=group,
            fragments=fragments,
            group_bbox=group_bbox,
            group_vec=group_vec,
            user_input=user_input,
            heuristic_decision=heuristic_decision,
        )

        system_prompt = (
            "You decide whether a canvas fragment group should be promoted to a new block. "
            "Return JSON only: {\"allow\": true|false, \"confidence\": 0..1, \"reason\": \"...\"}. "
            "Prefer conservative decisions. If the group is likely already covered by an existing block, choose allow=false."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)},
        ]
        try:
            parsed, _ = call_chat_completions(
                messages,
                model=self.agent_group_promote_review_model,
                temperature=self.agent_group_promote_review_temperature,
                max_tokens=self.agent_group_promote_review_max_tokens,
            )
        except Exception as exc:
            if self.cluster_logger:
                try:
                    self.cluster_logger.log(
                        "group_promotion_review_failed",
                        {"group_id": group_id, "error": str(exc), "mode": mode},
                    )
                except Exception:
                    pass
            return None

        allow = self._parse_llm_allow(parsed)
        if allow is None:
            if self.cluster_logger:
                try:
                    self.cluster_logger.log(
                        "group_promotion_review_failed",
                        {"group_id": group_id, "error": "allow_missing_or_invalid", "mode": mode, "raw": parsed},
                    )
                except Exception:
                    pass
            return None

        confidence = _safe_float(parsed.get("confidence")) if isinstance(parsed, dict) else None
        reason = ""
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or "").strip()[:240]

        reasons = list(heuristic_decision.reasons)
        reasons.append("llm_review_allow" if allow else "llm_review_deny")
        if reason:
            reasons.append(f"llm_reason:{reason[:80]}")
        metrics = dict(heuristic_decision.metrics)
        metrics["llm_review_used"] = True
        metrics["llm_review_mode"] = mode
        metrics["llm_review_allow"] = allow
        metrics["llm_review_confidence"] = round(confidence, 4) if confidence is not None else None

        if self.cluster_logger:
            try:
                self.cluster_logger.log(
                    "group_promotion_review",
                    {
                        "group_id": group_id,
                        "mode": mode,
                        "allow": allow,
                        "confidence": metrics.get("llm_review_confidence"),
                        "heuristic_allow": heuristic_decision.allow,
                        "heuristic_score": heuristic_decision.score,
                        "reason": reason,
                    },
                )
            except Exception:
                pass

        return GroupPromotionDecision(
            group_id=heuristic_decision.group_id,
            allow=allow,
            score=heuristic_decision.score,
            reasons=reasons,
            metrics=metrics,
            hard_reject=(heuristic_decision.hard_reject and not allow),
            source="llm" if mode == "llm" else "hybrid_llm",
        )

    def _build_group_promotion_review_payload(
        self,
        *,
        group_id: str,
        group,
        fragments: Sequence[Fragment],
        group_bbox: Optional[Tuple[float, float, float, float]],
        group_vec: Optional[List[float]],
        user_input: str,
        heuristic_decision: GroupPromotionDecision,
    ) -> Dict[str, object]:
        compact_fragments: List[Dict[str, object]] = []
        ordered = sorted(fragments, key=lambda f: f.timestamp or datetime.min, reverse=True)
        for fragment in ordered[:14]:
            compact = self._compact_fragment_for_promotion_review(fragment)
            if compact:
                compact_fragments.append(compact)

        candidate_blocks = self._review_candidate_blocks(group_bbox, group_vec, limit=5)
        return {
            "task": "decide_group_promotion",
            "group": {
                "groupId": group_id,
                "size": len(group.members),
                "updatedAt": group.updated_at.isoformat() if getattr(group, "updated_at", None) else None,
                "bbox": [round(float(v), 2) for v in group_bbox] if group_bbox else None,
                "fragments": compact_fragments,
            },
            "heuristic": {
                "allow": heuristic_decision.allow,
                "score": heuristic_decision.score,
                "hardReject": heuristic_decision.hard_reject,
                "reasons": list(heuristic_decision.reasons),
                "metrics": dict(heuristic_decision.metrics),
            },
            "userHint": str(user_input or "").strip(),
            "candidateBlocks": candidate_blocks,
        }

    def _review_candidate_blocks(
        self,
        group_bbox: Optional[Tuple[float, float, float, float]],
        group_vec: Optional[Sequence[float]],
        *,
        limit: int = 5,
    ) -> List[Dict[str, object]]:
        scored: List[Tuple[float, Dict[str, object]]] = []
        for block in self.state.list_blocks():
            overlap = 0.0
            if group_bbox and block.position:
                overlap = _bbox_overlap_ratio(group_bbox, block.position)

            semantic_distance: Optional[float] = None
            semantic_score = 0.0
            if group_vec is not None:
                block_vec = self._block_embedding_from_contents(block.contents)
                if block_vec:
                    semantic_distance = _cosine_distance(group_vec, block_vec)
                    semantic_score = max(0.0, 1.0 - semantic_distance)

            combined = (2.0 * overlap) + semantic_score
            payload: Dict[str, object] = {
                "blockId": block.block_id,
                "label": block.label,
                "summary": (block.summary or "")[:180],
                "overlap": round(overlap, 4),
                "semanticDistance": round(semantic_distance, 4) if semantic_distance is not None else None,
            }
            if block.position:
                payload["bbox"] = [round(float(v), 2) for v in block.position]
            scored.append((combined, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[: max(0, int(limit))]]

    def _compact_fragment_for_promotion_review(self, fragment: Fragment) -> Optional[Dict[str, object]]:
        kind = getattr(fragment.fragment_type, "value", str(fragment.fragment_type))
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        if kind == "text":
            text = self._fragment_text(fragment)
            item: Dict[str, object] = {"type": "text", "text": text[:220]}
            if fragment.bbox:
                item["bbox"] = [round(float(v), 2) for v in fragment.bbox]
            return item

        item = {"type": "stroke", "strokeType": str(payload.get("tool") or "stroke")}
        point = self._compact_point_for_review(payload.get("points"))
        if point:
            item["point"] = point
        return item

    @staticmethod
    def _compact_point_for_review(raw_points: object) -> Optional[List[float]]:
        if not isinstance(raw_points, (list, tuple)):
            return None
        latest: Optional[List[float]] = None
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                latest = [round(float(point[0]), 2), round(float(point[1]), 2)]
            except (TypeError, ValueError):
                continue
        return latest

    @staticmethod
    def _parse_llm_allow(parsed: object) -> Optional[bool]:
        if isinstance(parsed, dict):
            raw = parsed.get("allow")
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)):
                return bool(raw)
            if isinstance(raw, str):
                token = raw.strip().lower()
                if token in {"true", "allow", "yes", "promote", "1"}:
                    return True
                if token in {"false", "deny", "no", "defer", "0"}:
                    return False
            decision = parsed.get("decision")
            if isinstance(decision, str):
                token = decision.strip().lower()
                if token in {"allow", "promote", "approve"}:
                    return True
                if token in {"deny", "defer", "reject"}:
                    return False
        return None

    def _build_group_embedding(self, group, text_vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
        if getattr(group, "prototype_vec", None):
            return [float(v) for v in group.prototype_vec]

        if not text_vectors:
            return None
        dims = len(text_vectors[0])
        if dims <= 0:
            return None
        agg = [0.0] * dims
        used = 0
        for vec in text_vectors:
            if len(vec) != dims:
                continue
            used += 1
            for idx in range(dims):
                agg[idx] += float(vec[idx])
        if used <= 0:
            return None
        return [value / used for value in agg]

    def _block_embedding_from_contents(self, fragment_ids: Iterable[str]) -> Optional[List[float]]:
        vectors: List[List[float]] = []
        dims: Optional[int] = None
        for fragment_id in fragment_ids:
            fragment = self.state.fragments.get(fragment_id)
            if not fragment or fragment.feature_vec is None:
                continue
            vec = [float(v) for v in fragment.feature_vec]
            if dims is None:
                dims = len(vec)
            if dims is None or len(vec) != dims:
                continue
            vectors.append(vec)
        if not vectors or dims is None:
            return None
        agg = [0.0] * dims
        for vec in vectors:
            for idx in range(dims):
                agg[idx] += vec[idx]
        return [value / len(vectors) for value in agg]

    def _group_bbox(self, fragments: Sequence[Fragment]) -> Optional[Tuple[float, float, float, float]]:
        merged: Optional[Tuple[float, float, float, float]] = None
        for fragment in fragments:
            if not fragment.bbox:
                continue
            bbox = (
                float(fragment.bbox[0]),
                float(fragment.bbox[1]),
                float(fragment.bbox[2]),
                float(fragment.bbox[3]),
            )
            if merged is None:
                merged = bbox
            else:
                merged = _merge_bbox_local(merged, bbox)
        return merged

    def _fragment_text(self, fragment: Fragment) -> str:
        text = str(fragment.text or "").strip()
        if text:
            return text
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        meta = payload.get("meta") if isinstance(payload, dict) else {}
        if isinstance(meta, dict):
            return markdown_to_semantic_text(str(meta.get("text") or ""))
        return ""

    def _remap_orchestrator_context_ids(self, promoted_map: Dict[str, str]) -> None:
        if not promoted_map:
            return
        context = self.orchestrator.context
        main_id = getattr(context, "main_block_id", None)
        if isinstance(main_id, str) and main_id in promoted_map:
            context.main_block_id = promoted_map[main_id]

        active_ids = list(getattr(context, "active_block_ids", []) or [])
        valid_ids = set(self.state.blocks.keys()) | set(self.state.groups.keys())
        remapped: List[str] = []
        seen: Set[str] = set()
        for context_id in active_ids:
            mapped = promoted_map.get(context_id, context_id)
            if mapped not in valid_ids:
                continue
            if mapped in seen:
                continue
            remapped.append(mapped)
            seen.add(mapped)
        context.active_block_ids = remapped

    def _build_noop_placeholder_payload(self, next_step_hint: Optional[str]) -> Dict[str, object]:
        message = self._normalize_noop_placeholder_text(next_step_hint)
        x0, y0, x1, y1 = self._choose_noop_placeholder_bbox(message)
        canvas_w, canvas_h = self.block_manager.canvas_size or (1920.0, 1080.0)
        stroke_id = f"ai_noop_placeholder_{int(datetime.utcnow().timestamp() * 1000)}"
        return {
            "version": 1,
            "intent": "write",
            "canvas": {"width": int(canvas_w), "height": int(canvas_h)},
            "strokes": [
                {
                    "id": stroke_id,
                    "tool": "text",
                    "points": [
                        [round(float(x0), 3), round(float(y0), 3)],
                        [round(float(x1), 3), round(float(y1), 3)],
                    ],
                    "style": {"size": "m", "color": "grey", "opacity": 1.0},
                    "meta": {
                        "text": message,
                        "summary": "Await user input",
                        "fontFamily": "sans-serif",
                        "fontWeight": "400",
                        "fontSize": 18,
                        "growDir": "right-down",
                        "placeholder": True,
                    },
                }
            ],
        }

    @staticmethod
    def _normalize_noop_placeholder_text(next_step_hint: Optional[str]) -> str:
        raw = " ".join(str(next_step_hint or "").split()).strip()
        if raw:
            return raw[:260]
        return "Await further user input to decide next steps or context changes."

    def _choose_noop_placeholder_bbox(self, text: str) -> Tuple[float, float, float, float]:
        canvas_w, canvas_h = self.block_manager.canvas_size or (1920.0, 1080.0)
        w = float(min(560.0, max(320.0, len(text) * 7.2 + 44.0)))
        max_chars_per_line = max(24, int(w / 9.0))
        line_count = max(1, int(math.ceil(max(1, len(text)) / max_chars_per_line)))
        h = float(min(220.0, max(56.0, line_count * 30.0 + 20.0)))

        x_min = 8.0
        y_min = 8.0
        x_max = max(x_min, float(canvas_w) - w - 8.0)
        y_max = max(y_min, float(canvas_h) - h - 8.0)

        def _clamp_xy(x: float, y: float) -> Tuple[float, float]:
            return (min(max(x, x_min), x_max), min(max(y, y_min), y_max))

        center_x = (float(canvas_w) - w) / 2.0
        center_y = (float(canvas_h) - h) / 2.0
        step_x = max(32.0, min(w * 0.55, float(canvas_w) * 0.18))
        step_y = max(28.0, min(h * 0.70, float(canvas_h) * 0.18))

        # Prefer positions near the user's screen center, while still minimizing overlap.
        # Build a small center-biased lattice (center + 8-neighborhood + outer ring).
        raw_candidates: List[Tuple[float, float]] = [(center_x, center_y)]
        for ring in (1.0, 2.0):
            for dx_mul, dy_mul in (
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ):
                raw_candidates.append(
                    (
                        center_x + dx_mul * step_x * ring,
                        center_y + dy_mul * step_y * ring,
                    )
                )
        # Add a few elongated offsets to help dodge dense central clusters.
        raw_candidates.extend(
            [
                (center_x - 3.0 * step_x, center_y),
                (center_x + 3.0 * step_x, center_y),
                (center_x, center_y - 3.0 * step_y),
                (center_x, center_y + 3.0 * step_y),
            ]
        )

        candidates_xy: List[Tuple[float, float]] = []
        seen_candidates: Set[Tuple[float, float]] = set()
        for raw_x, raw_y in raw_candidates:
            cx, cy = _clamp_xy(raw_x, raw_y)
            key = (round(cx, 3), round(cy, 3))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates_xy.append((cx, cy))

        occupied: List[Tuple[float, float, float, float]] = []
        for fragment in self.state.fragments.values():
            bbox = getattr(fragment, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            try:
                bx0, by0, bx1, by1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                continue
            if not all(math.isfinite(v) for v in (bx0, by0, bx1, by1)):
                continue
            if bx1 <= bx0 or by1 <= by0:
                continue
            occupied.append((bx0, by0, bx1, by1))

        best_bbox: Optional[Tuple[float, float, float, float]] = None
        best_score: Optional[Tuple[float, float, float]] = None
        target_cx = float(canvas_w) / 2.0
        target_cy = float(canvas_h) / 2.0
        for x, y in candidates_xy:
            candidate = (x, y, x + w, y + h)
            overlap_area = 0.0
            for occ in occupied:
                overlap_area += self._bbox_intersection_area(candidate, occ)
            candidate_cx = x + w / 2.0
            candidate_cy = y + h / 2.0
            center_distance = math.hypot(candidate_cx - target_cx, candidate_cy - target_cy)
            # Final tiny tie-break prefers slightly upper positions when overlap and
            # center distance are equivalent, to reduce covering lower canvas content.
            vertical_tie = y
            score = (overlap_area, center_distance, vertical_tie)
            if best_score is None or score < best_score:
                best_score = score
                best_bbox = candidate

        if best_bbox is None:
            x, y = _clamp_xy(center_x, center_y)
            best_bbox = (x, y, x + w, y + h)
        return best_bbox

    @staticmethod
    def _bbox_intersection_area(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

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
            raw_source = _normalize_text(raw_text) or _normalize_text(meta_payload.get("summary"))
            text = markdown_to_semantic_text(raw_source)
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

    @staticmethod
    def _stroke_graph_meta(stroke: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not isinstance(stroke, dict):
            return None
        meta = stroke.get("meta")
        if not isinstance(meta, dict):
            return None
        graph_meta = meta.get("graph")
        if not isinstance(graph_meta, dict):
            return None
        return graph_meta

    def _stroke_graph_proposal_key(self, stroke: Dict[str, object]) -> str:
        graph_meta = self._stroke_graph_meta(stroke)
        if not isinstance(graph_meta, dict):
            return ""
        key = str(graph_meta.get("proposalKey") or "").strip()
        if not key:
            return ""
        compact = " ".join(key.split()).strip()
        return compact[:64]

    def _stroke_graph_target_block_id(self, stroke: Dict[str, object]) -> Optional[str]:
        graph_meta = self._stroke_graph_meta(stroke)
        if not isinstance(graph_meta, dict):
            return None
        candidate = str(graph_meta.get("targetBlockId") or graph_meta.get("assignToBlockId") or "").strip()
        if not candidate:
            return None
        return candidate

    def _stroke_graph_is_block_create(self, stroke: Dict[str, object]) -> bool:
        graph_meta = self._stroke_graph_meta(stroke)
        if not isinstance(graph_meta, dict):
            return False
        intent = str(graph_meta.get("blockIntent") or graph_meta.get("intent") or "").strip().lower()
        return intent in {"create", "create_block", "new", "new_block"}

    def _stroke_with_proposal_target(self, stroke: Dict[str, object], block_id: str) -> Optional[Dict[str, object]]:
        if not isinstance(stroke, dict) or not block_id:
            return None
        meta = stroke.get("meta")
        if not isinstance(meta, dict):
            return None
        graph_meta = meta.get("graph")
        if not isinstance(graph_meta, dict):
            return None

        stroke2 = dict(stroke)
        meta2 = dict(meta)
        graph2 = dict(graph_meta)
        graph2["targetBlockId"] = block_id
        # Once a proposalKey has been resolved in-batch, later same-key strokes should attach
        # instead of accidentally creating more blocks even if the model repeats blockIntent.
        for key in ("blockIntent", "intent"):
            raw_val = str(graph2.get(key) or "").strip().lower()
            if raw_val in {"create", "create_block", "new", "new_block"}:
                graph2.pop(key, None)
        meta2["graph"] = graph2
        stroke2["meta"] = meta2
        return stroke2

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
        if role in {
            "title",
            "heading",
            "header",
            "subtitle",
            "subheading",
            "sub-title",
            "sub_title",
            "标题",
            "副标题",
            "小标题",
        }:
            return True
        text_val = ""
        if isinstance(meta, dict):
            text_val = markdown_to_semantic_text(str(meta.get("text") or ""))
        if isinstance(meta, dict):
            raw_line_count = meta.get("lineCount")
        else:
            raw_line_count = None
        try:
            line_count = int(raw_line_count) if raw_line_count is not None else None
        except (TypeError, ValueError):
            line_count = None
        if line_count is None:
            line_count = text_val.count("\n") + 1 if text_val else 1
        # Conservative heading heuristic:
        # - very large concise text, or
        # - large + heavy concise text.
        concise = line_count <= 2
        short_text = (not text_val) or len(text_val) <= 80
        if size_val is not None and size_val >= 34 and concise and short_text:
            return True
        if size_val is not None and size_val >= 28 and weight_val >= 700 and concise and (not text_val or len(text_val) <= 60):
            return True
        if size_val is not None and size_val >= 22 and weight_val >= 600 and concise and short_text:
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
