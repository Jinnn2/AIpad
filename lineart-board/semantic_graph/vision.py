from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
import math
import uuid

from .block_manager import BlockManager
from .models import FragmentType


@dataclass
class VisionPayload:
    stroke_fragment_ids: List[str]
    image_bytes: bytes
    image_mime: str = "image/png"
    metadata: List[Dict[str, object]] = field(default_factory=list)
    fragments: List[Dict[str, object]] = field(default_factory=list)
    candidate_blocks: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class VisionResult:
    kind: str  # legacy field; kept for backwards compat
    decision: Optional[str] = None  # "merge_block" | "new_block"
    label: Optional[str] = None
    stroke_fragment_ids: List[str] = field(default_factory=list)
    target_fragment_id: Optional[str] = None
    target_block_id: Optional[str] = None
    confidence: Optional[float] = None
    summary: Optional[str] = None
    relationships: Optional[List[Dict[str, object]]] = None
    extra: Dict[str, object] = field(default_factory=dict)


@runtime_checkable
class VisionBackend(Protocol):
    def analyze(self, payload: VisionPayload) -> List[VisionResult]:
        ...


class VisionGrouper:
    def __init__(
        self,
        block_manager: BlockManager,
        backend: VisionBackend,
        *,
        stroke_threshold: int = 4,
        idle_timeout: timedelta = timedelta(seconds=2),
        auto_promote_confidence: float = 0.85,
        spatial_threshold: float = 220.0,
    ) -> None:
        self.block_manager = block_manager
        self.backend = backend
        self.stroke_threshold = stroke_threshold
        self.idle_timeout = idle_timeout
        self.auto_promote_confidence = auto_promote_confidence
        self.spatial_threshold = spatial_threshold
        self._last_activity: Optional[datetime] = None
        self._pending_groups: Dict[str, _PendingGroup] = {}
        self._diagram_blocks: Dict[str, Tuple[float, float, float, float]] = {}

    def register_activity(self, when: Optional[datetime] = None) -> None:
        self._last_activity = when or datetime.utcnow()

    def should_trigger(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        unlabeled = self.block_manager.list_unlabeled_strokes()
        if len(unlabeled) < self.stroke_threshold:
            return False
        latest_ts = self._latest_unlabeled_timestamp(unlabeled)
        if not latest_ts:
            latest_ts = self._last_activity or now
        return (now - latest_ts) >= self.idle_timeout

    def process(self, payload: VisionPayload) -> List[VisionResult]:
        raw_results = self.backend.analyze(payload)
        if not isinstance(raw_results, list):
            return []
        applied: List[VisionResult] = []
        for raw in raw_results:
            if isinstance(raw, VisionResult):
                result = raw
            elif isinstance(raw, dict):
                confidence = raw.get("confidence")
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                    except Exception:
                        confidence = None
                result = VisionResult(
                    kind=str(raw.get("kind") or ""),
                    decision=raw.get("decision"),
                    label=raw.get("label"),
                    stroke_fragment_ids=list(raw.get("stroke_fragment_ids") or []),
                    target_fragment_id=raw.get("target_fragment_id"),
                    target_block_id=raw.get("target_block_id"),
                    confidence=confidence,
                    summary=raw.get("summary"),
                    relationships=raw.get("relationships"),
                    extra={k: v for k, v in raw.items() if k not in {"kind", "label", "stroke_fragment_ids", "target_fragment_id", "confidence"}},
                )
            else:
                continue
            decision = (result.decision or result.kind or "").lower()
            strokes = [fid for fid in result.stroke_fragment_ids if fid in payload.stroke_fragment_ids]
            if not strokes:
                continue
            self.block_manager.remove_unlabeled_strokes(strokes)
            if decision in {"annotation", "merge_block", "merge_to_block"}:
                self._handle_annotation(result, strokes)
            elif decision in {"diagram", "new_block"}:
                self._handle_diagram(result, strokes)
            applied.append(result)
        return applied

    def ingest_fragment(self, fragment, *, reason: str = "auto") -> List[VisionPayload]:
        if fragment.fragment_type != FragmentType.STROKE:
            return []
        bbox = fragment.bbox
        if not bbox:
            return []
        # Diagram shortcut: if overlaps known diagram block, attach immediately.
        for block_id, block_bbox in self._diagram_blocks.items():
            if _bbox_overlap_ratio(bbox, block_bbox) > 0.6:
                self.block_manager.remove_unlabeled_strokes([fragment.fragment_id])
                self.block_manager.attach_fragment_to_block(block_id, fragment.fragment_id)
                return []
        group = self._assign_to_group(fragment.fragment_id, bbox, reason=reason)
        group.updated_at = datetime.utcnow()
        return self._drain_ready_groups()

    def flush_groups(self, *, reason: str) -> List[VisionPayload]:
        ready = list(self._pending_groups.values())
        self._pending_groups.clear()
        payloads = [self._group_to_payload(group, override_reason=reason) for group in ready]
        return payloads

    def register_diagram_block(self, block_id: str, bbox: Optional[Tuple[float, float, float, float]]) -> None:
        if bbox:
            self._diagram_blocks[block_id] = bbox


    # ----------------------------- internal helpers ----------------------------- #

    def _assign_to_group(self, fragment_id: str, bbox: Tuple[float, float, float, float], *, reason: str) -> _PendingGroup:
        now = datetime.utcnow()
        best_group = None
        best_distance = float("inf")
        for group in self._pending_groups.values():
            distance = _bbox_distance(group.bbox, bbox)
            if distance < best_distance:
                best_distance = distance
                best_group = group
        if best_group and best_distance <= self.spatial_threshold:
            best_group.stroke_ids.append(fragment_id)
            best_group.bbox = _merge_bbox(best_group.bbox, bbox)
            if len(best_group.stroke_ids) >= self.stroke_threshold and not best_group.ready_reason:
                best_group.ready_reason = "stroke_threshold"
            return best_group
        group_id = f"vision_{uuid.uuid4().hex[:8]}"
        group = _PendingGroup(
            group_id=group_id,
            stroke_ids=[fragment_id],
            bbox=bbox,
            created_at=now,
            updated_at=now,
            ready_reason=None,
        )
        self._pending_groups[group_id] = group
        for other in self._pending_groups.values():
            if other is group:
                continue
            if not other.ready_reason:
                other.ready_reason = "spatial_split"
        return group

    def _group_to_payload(self, group: _PendingGroup, override_reason: Optional[str] = None) -> VisionPayload:
        metadata = [
            {
                "group_id": group.group_id,
                "reason": override_reason or group.ready_reason or "manual",
                "bbox": list(group.bbox),
                "count": len(group.stroke_ids),
            }
        ]
        return VisionPayload(
            stroke_fragment_ids=list(group.stroke_ids),
            image_bytes=b"",
            image_mime="application/octet-stream",
            metadata=metadata,
        )

    def _drain_ready_groups(self) -> List[VisionPayload]:
        ready_payloads: List[VisionPayload] = []
        for gid in list(self._pending_groups.keys()):
            group = self._pending_groups[gid]
            if group.ready_reason:
                ready_payloads.append(self._group_to_payload(group))
                self._pending_groups.pop(gid, None)
        return ready_payloads

    def _handle_annotation(self, result: VisionResult, strokes: List[str]) -> None:
        block_id = result.target_block_id
        target_fragment_id = result.target_fragment_id
        if not block_id and target_fragment_id:
            block_id = self.block_manager.get_block_id_for_fragment(target_fragment_id)
        if not block_id and target_fragment_id:
            group_id = self.block_manager.get_group_id_for_fragment(target_fragment_id)
            if group_id:
                group = self.block_manager.state.groups.get(group_id)
                if group:
                    group.members.update(strokes)
                    group.need_llm_review = True
                    group.updated_at = datetime.utcnow()
            return
        if not block_id:
            return
        for stroke_id in strokes:
            self.block_manager.attach_fragment_to_block(block_id, stroke_id)
        block = self.block_manager.state.blocks.get(block_id)
        if block and block.contents:
            block.position = self.block_manager._refresh_block_bbox(block)  # type: ignore[attr-defined]

    def _handle_diagram(self, result: VisionResult, strokes: List[str]) -> None:
        confidence = result.confidence or 0.0
        label = result.label or "diagram"
        try:
            group = self.block_manager.create_group_from_fragments(strokes, need_llm_review=True)
        except ValueError:
            return
        if confidence >= self.auto_promote_confidence:
            fragments = [self.block_manager.state.fragments[fid] for fid in group.members]
            for fragment in fragments:
                fragment.fragment_type = FragmentType.STROKE
            try:
                block = self.block_manager.promote_group(group.group_id)
            except Exception:
                return
            if label:
                block.label = label
            annotation = {
                "summary": result.summary or block.summary or f"Diagram: {label}",
                "label": block.label,
                "relationships": result.relationships,
            }
            try:
                self.block_manager.register_block_annotation(block.block_id, annotation)
            except Exception:
                pass
            block.position = self.block_manager._refresh_block_bbox(block)  # type: ignore[attr-defined]
            self.register_diagram_block(block.block_id, block.position)

    def _latest_unlabeled_timestamp(self, fragment_ids: Iterable[str]) -> Optional[datetime]:
        timestamps = []
        for fid in fragment_ids:
            fragment = self.block_manager.state.fragments.get(fid)
            if fragment and fragment.timestamp:
                timestamps.append(fragment.timestamp)
        return max(timestamps) if timestamps else None


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def _merge_bbox(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_overlap_ratio(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = max(area_a + area_b - intersection, 1e-6)
    return intersection / union
@dataclass
class _PendingGroup:
    group_id: str
    stroke_ids: List[str]
    bbox: Tuple[float, float, float, float]
    created_at: datetime
    updated_at: datetime
    ready_reason: Optional[str] = None
