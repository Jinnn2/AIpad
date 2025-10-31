from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Protocol, runtime_checkable

from .block_manager import BlockManager
from .models import FragmentType


@dataclass
class VisionPayload:
    stroke_fragment_ids: List[str]
    image_bytes: bytes
    image_mime: str = "image/png"
    metadata: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class VisionResult:
    kind: str  # "annotation" | "diagram"
    label: Optional[str] = None
    stroke_fragment_ids: List[str] = field(default_factory=list)
    target_fragment_id: Optional[str] = None
    confidence: Optional[float] = None
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
    ) -> None:
        self.block_manager = block_manager
        self.backend = backend
        self.stroke_threshold = stroke_threshold
        self.idle_timeout = idle_timeout
        self.auto_promote_confidence = auto_promote_confidence
        self._last_activity: Optional[datetime] = None

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
                    label=raw.get("label"),
                    stroke_fragment_ids=list(raw.get("stroke_fragment_ids") or []),
                    target_fragment_id=raw.get("target_fragment_id"),
                    confidence=confidence,
                    extra={k: v for k, v in raw.items() if k not in {"kind", "label", "stroke_fragment_ids", "target_fragment_id", "confidence"}},
                )
            else:
                continue
            if result.kind not in {"annotation", "diagram"}:
                continue
            strokes = [fid for fid in result.stroke_fragment_ids if fid in payload.stroke_fragment_ids]
            if not strokes:
                continue
            self.block_manager.remove_unlabeled_strokes(strokes)
            if result.kind == "annotation":
                self._handle_annotation(result, strokes)
            elif result.kind == "diagram":
                self._handle_diagram(result, strokes)
            applied.append(result)
        return applied

    # ----------------------------- internal helpers ----------------------------- #

    def _handle_annotation(self, result: VisionResult, strokes: List[str]) -> None:
        target_fragment_id = result.target_fragment_id
        if not target_fragment_id:
            return
        block_id = self.block_manager.get_block_id_for_fragment(target_fragment_id)
        if not block_id:
            group_id = self.block_manager.get_group_id_for_fragment(target_fragment_id)
            if not group_id:
                return
            group = self.block_manager.state.groups.get(group_id)
            if not group:
                return
            group.members.update(strokes)
            group.need_llm_review = True
            group.updated_at = datetime.utcnow()
            return
        for stroke_id in strokes:
            self.block_manager.attach_fragment_to_block(block_id, stroke_id)

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
            if self.block_manager.summarizer:
                block = self.block_manager.promote_group(group.group_id)
                if label:
                    block.label = label
                if not block.summary:
                    block.summary = f"Vision detected diagram: {label}"

    def _latest_unlabeled_timestamp(self, fragment_ids: Iterable[str]) -> Optional[datetime]:
        timestamps = []
        for fid in fragment_ids:
            fragment = self.block_manager.state.fragments.get(fid)
            if fragment and fragment.timestamp:
                timestamps.append(fragment.timestamp)
        return max(timestamps) if timestamps else None
