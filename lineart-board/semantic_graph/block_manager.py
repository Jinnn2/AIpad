from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .models import (
    BBox,
    Block,
    BlockNotFoundError,
    BlockRelationship,
    BlockRelationshipType,
    Fragment,
    FragmentNotFoundError,
    FragmentType,
    Group,
    GroupNotFoundError,
    GroupState,
)
from .state import GraphState
from .similarity import cosine_distance

FeatureVector = Sequence[float]


@dataclass
class FragmentAssignment:
    fragment_id: str
    status: str  # "block" | "group" | "stroke"
    block_id: Optional[str] = None
    group_id: Optional[str] = None
    promoted_block_id: Optional[str] = None


@runtime_checkable
class TextEmbedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        """Return a high-dimensional embedding for the given text."""


@runtime_checkable
class BlockSummarizer(Protocol):
    def propose_block(self, fragments: List[Fragment]) -> Tuple[str, str]:
        """Return (label, summary) for a new block."""

    def refine_summary(self, block: Block, fragments: List[Fragment]) -> str:
        """Return an updated summary for an existing block."""


def _union_bbox(bboxes: Iterable[BBox]) -> Optional[BBox]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for x0, y0, x1, y1 in bboxes:
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


class BlockManager:
    """
    Maintains the fragment/group/block hierarchy, handles clustering, and keeps block metadata fresh.
    """

    def __init__(
        self,
        *,
        state: Optional[GraphState] = None,
        embedder: Optional[TextEmbedder] = None,
        summarizer: Optional[BlockSummarizer] = None,
        group_distance_threshold: float = 0.4,
        block_distance_threshold: float = 0.35,
        summary_refresh_ratio: float = 0.3,
        summary_refresh_interval: timedelta = timedelta(minutes=10),
        canvas_size: Optional[Tuple[float, float]] = None,
        auto_promote_group_size: int = 5,
        spatial_target_ratio: float = 0.40,
        time_target_ratio: float = 0.08,
        cluster_logger: Optional[Any] = None,
    ) -> None:
        self.state = state or GraphState()
        self.embedder = embedder
        self.summarizer = summarizer
        self.group_distance_threshold = group_distance_threshold
        self.block_distance_threshold = block_distance_threshold
        self.summary_refresh_ratio = summary_refresh_ratio
        self.summary_refresh_interval = summary_refresh_interval
        self.canvas_size = canvas_size or (1.0, 1.0)
        self.auto_promote_group_size = max(1, auto_promote_group_size)
        self.spatial_target_ratio = max(0.0, float(spatial_target_ratio))
        self.time_target_ratio = max(0.0, float(time_target_ratio))

        self._time_anchor: Optional[datetime] = None
        self._fragment_to_group: Dict[str, str] = {}
        self._fragment_to_block: Dict[str, str] = {}
        self._unlabeled_strokes: List[str] = []
        self._block_incoming_counts: Dict[str, int] = defaultdict(int)
        self._group_touch_counts: Dict[str, int] = defaultdict(int)
        self.cluster_logger = cluster_logger

    # ------------------------------- Public API -------------------------------- #

    def register_fragment(self, fragment: Fragment) -> FragmentAssignment:
        """
        Ingest a fresh fragment into the knowledge graph.
        Returns a FragmentAssignment describing where the fragment landed.
        """
        self.state.add_fragment(fragment)
        assignment = FragmentAssignment(fragment_id=fragment.fragment_id, status="stroke")

        if fragment.fragment_type != FragmentType.TEXT:
            self._unlabeled_strokes.append(fragment.fragment_id)
            return assignment

        feature_vec = self._ensure_feature_vector(fragment)
        fragment.feature_vec = feature_vec

        self._log_cluster(
            "fragment_ingest",
            fragment_id=fragment.fragment_id,
            fragment_type=fragment.fragment_type.value,
            text=self._shorten_text(fragment.text),
            bbox=fragment.bbox,
            groups=len(self.state.groups),
            blocks=len(self.state.blocks),
        )

        if not self.state.blocks and not self.state.groups:
            self._log_cluster(
                "fragment_cold_start",
                fragment_id=fragment.fragment_id,
                text=self._shorten_text(fragment.text),
            )
            new_block = self._create_block_from_fragment(fragment)
            assignment.status = "block"
            assignment.block_id = new_block.block_id
            assignment.promoted_block_id = new_block.block_id
            return assignment

        if self._is_heading_fragment(fragment):
            self._log_cluster(
                "heading_promoted",
                fragment_id=fragment.fragment_id,
                text=self._shorten_text(fragment.text),
            )
            new_block = self._create_block_from_fragment(fragment)
            assignment.status = "block"
            assignment.block_id = new_block.block_id
            assignment.promoted_block_id = new_block.block_id
            return assignment

        block_id, block_distance = self._match_block(feature_vec, fragment_id=fragment.fragment_id)
        if block_id:
            self._log_cluster(
                "fragment_matched_block",
                fragment_id=fragment.fragment_id,
                block_id=block_id,
                distance=round(block_distance, 6),
                text=self._shorten_text(fragment.text),
            )
            block = self.attach_fragment_to_block(block_id, fragment.fragment_id)
            self._tag_fragment_with_block(fragment, block)
            assignment.status = "block"
            assignment.block_id = block_id
            return assignment

        group_id, group_distance = self._match_group(feature_vec, fragment_id=fragment.fragment_id)
        if group_id:
            self._log_cluster(
                "fragment_matched_group",
                fragment_id=fragment.fragment_id,
                group_id=group_id,
                distance=round(group_distance, 6),
                text=self._shorten_text(fragment.text),
            )
            self._assign_to_group(fragment.fragment_id, feature_vec, allow_create=False, existing_group_id=group_id)
            assignment.status = "group"
            assignment.group_id = group_id
            self._group_touch_counts[group_id] += 1
            if self._should_promote_group(group_id):
                self._log_cluster(
                    "group_promoted",
                    group_id=group_id,
                    trigger_fragment=fragment.fragment_id,
                )
                promoted_block = self.promote_group(group_id)
                assignment.status = "block"
                assignment.block_id = promoted_block.block_id
                assignment.promoted_block_id = promoted_block.block_id
            return assignment

        # No matching block or group -> start a new pending group
        group_id = self._assign_to_group(fragment.fragment_id, feature_vec, allow_create=True)
        if group_id:
            self._log_cluster(
                "group_created",
                group_id=group_id,
                fragment_id=fragment.fragment_id,
                text=self._shorten_text(fragment.text),
            )
            assignment.status = "group"
            assignment.group_id = group_id
            self._group_touch_counts[group_id] += 1
            if self._should_promote_group(group_id):
                self._log_cluster(
                    "group_promoted",
                    group_id=group_id,
                    trigger_fragment=fragment.fragment_id,
                )
                promoted_block = self.promote_group(group_id)
                assignment.status = "block"
                assignment.block_id = promoted_block.block_id
                assignment.promoted_block_id = promoted_block.block_id
            return assignment

        # Fallback safety: create block if group assignment failed
        self._log_cluster(
            "fragment_fallback_block",
            fragment_id=fragment.fragment_id,
            text=self._shorten_text(fragment.text),
        )
        new_block = self._create_block_from_fragment(fragment)
        assignment.status = "block"
        assignment.block_id = new_block.block_id
        assignment.promoted_block_id = new_block.block_id
        return assignment

    def mark_group_stable(self, group_id: str) -> Block:
        group = self._get_group(group_id)
        group.state = GroupState.STABLE
        group.updated_at = datetime.utcnow()
        return self.promote_group(group_id)

    def promote_group(self, group_id: str) -> Block:
        group = self._get_group(group_id)
        fragments = [self.state.fragments[fid] for fid in group.members]
        if not fragments:
            raise ValueError(f"group {group_id} has no fragments to promote")

        if not self.summarizer:
            raise RuntimeError("BlockSummarizer is required to promote groups")

        label, summary = self.summarizer.propose_block(fragments)
        bbox_candidates = [f.bbox for f in fragments if f.bbox]
        position = _union_bbox(bbox_candidates) if bbox_candidates else None
        block_id = self._generate_block_id()
        block = Block(
            block_id=block_id,
            label=label,
            summary=summary,
            position=position,
            contents=set(group.members),
        )
        block.last_summary_member_count = len(block.contents)
        block.last_summary_ts = datetime.utcnow()
        block.character_count = sum(
            self._fragment_text_length(self.state.fragments.get(fid))
            for fid in group.members
            if self.state.fragments.get(fid)
        )
        self.state.add_block(block)

        for fragment_id in group.members:
            self._fragment_to_block[fragment_id] = block_id
            self._fragment_to_group.pop(fragment_id, None)
            fragment = self.state.fragments.get(fragment_id)
            if fragment:
                self._tag_fragment_with_block(fragment, block)

        group.state = GroupState.RETIRED
        self.state.remove_group(group_id)
        self._group_touch_counts.pop(group_id, None)
        self._refresh_block_embedding(block)
        return block

    def attach_fragment_to_block(self, block_id: str, fragment_id: str) -> Block:
        block = self.state.blocks.get(block_id)
        if not block:
            raise BlockNotFoundError(block_id)
        fragment = self.state.fragments.get(fragment_id)
        if not fragment:
            raise FragmentNotFoundError(fragment_id)
        block.add_contents({fragment_id})
        block.position = self._refresh_block_bbox(block)
        block.character_count += self._fragment_text_length(fragment)
        self._fragment_to_block[fragment_id] = block_id
        self._fragment_to_group.pop(fragment_id, None)
        self._block_incoming_counts[block_id] += 1
        self._maybe_refresh_summary(block_id)
        self._refresh_block_embedding(block)
        return block

    def orphan_fragment(self, fragment_id: str) -> None:
        self._fragment_to_group.pop(fragment_id, None)
        self._fragment_to_block.pop(fragment_id, None)

    def list_unlabeled_strokes(self) -> List[str]:
        return list(self._unlabeled_strokes)

    def pop_unlabeled_strokes(self, count: Optional[int] = None) -> List[str]:
        if count is None or count >= len(self._unlabeled_strokes):
            items, self._unlabeled_strokes = self._unlabeled_strokes, []
            return items
        items = self._unlabeled_strokes[:count]
        self._unlabeled_strokes = self._unlabeled_strokes[count:]
        return items

    def remove_unlabeled_strokes(self, fragment_ids: Iterable[str]) -> None:
        removal = set(fragment_ids)
        if not removal:
            return
        self._unlabeled_strokes = [fid for fid in self._unlabeled_strokes if fid not in removal]

    def connect_blocks(
        self,
        source_block_id: str,
        target_block_id: str,
        rel_type: BlockRelationshipType,
        score: float = 1.0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        relationship = BlockRelationship(
            source_block_id=source_block_id,
            target_block_id=target_block_id,
            rel_type=rel_type,
            score=score,
            metadata=metadata or {},
        )
        self.state.connect_blocks(relationship)

    def create_group_from_fragments(
        self,
        fragment_ids: Iterable[str],
        *,
        need_llm_review: bool = True,
    ) -> Group:
        fragment_set = {fid for fid in fragment_ids if fid in self.state.fragments}
        if not fragment_set:
            raise ValueError("cannot create group with empty fragment set")
        group = Group(
            group_id=self._generate_group_id(),
            members=fragment_set,
            prototype_vec=None,
            need_llm_review=need_llm_review,
        )
        self.state.add_group(group)
        for fid in fragment_set:
            self._fragment_to_group[fid] = group.group_id
        return group

    def ensure_block_summary_fresh(self, block_id: str, force: bool = False) -> None:
        self._maybe_refresh_summary(block_id, force=force)

    # ------------------------------- Internal helpers -------------------------------- #

    def _assign_to_group(
        self,
        fragment_id: str,
        feature_vec: FeatureVector,
        *,
        allow_create: bool = True,
        existing_group_id: Optional[str] = None,
    ) -> Optional[str]:
        if existing_group_id:
            group = self._get_group(existing_group_id)
            group.add_member(fragment_id, feature_vec)
            self._fragment_to_group[fragment_id] = existing_group_id
            self._log_cluster(
                "assign_existing_group",
                group_id=existing_group_id,
                fragment_id=fragment_id,
                members=len(group.members),
            )
            return existing_group_id

        best_group_id = None
        best_distance = float("inf")
        for group in self.state.groups.values():
            if not group.prototype_vec:
                continue
            distance = cosine_distance(feature_vec, group.prototype_vec)
            if distance < best_distance:
                best_distance = distance
                best_group_id = group.group_id

        if best_group_id is not None and best_distance <= self.group_distance_threshold:
            group = self.state.groups[best_group_id]
            group.add_member(fragment_id, feature_vec)
            self._fragment_to_group[fragment_id] = best_group_id
            self._log_cluster(
                "assign_existing_group",
                group_id=best_group_id,
                fragment_id=fragment_id,
                members=len(group.members),
                distance=round(best_distance, 6),
            )
            return best_group_id

        if not allow_create:
            return None

        new_group = Group(
            group_id=self._generate_group_id(),
            members={fragment_id},
            prototype_vec=list(feature_vec),
        )
        self.state.add_group(new_group)
        self._fragment_to_group[fragment_id] = new_group.group_id
        self._group_touch_counts[new_group.group_id] = 0
        self._log_cluster(
            "assign_new_group",
            group_id=new_group.group_id,
            fragment_id=fragment_id,
        )
        return new_group.group_id

    def _ensure_group_embedding(self, group: Group) -> Optional[List[float]]:
        if group.prototype_vec is not None:
            return list(group.prototype_vec)
        embedding = self._compute_block_embedding(group.members)
        if embedding is not None:
            group.prototype_vec = list(embedding)
            return list(embedding)
        return None

    def _absorb_group_into_block(self, block: Block, group: Group) -> None:
        fragment_ids = list(group.members)
        if not fragment_ids:
            return
        block.add_contents(fragment_ids)
        block.position = self._refresh_block_bbox(block)
        char_delta = 0
        for fid in fragment_ids:
            self._fragment_to_block[fid] = block.block_id
            self._fragment_to_group.pop(fid, None)
            fragment = self.state.fragments.get(fid)
            if fragment:
                char_delta += self._fragment_text_length(fragment)
                self._tag_fragment_with_block(fragment, block)
        group.state = GroupState.RETIRED
        self.state.remove_group(group.group_id)
        self._group_touch_counts.pop(group.group_id, None)
        block.character_count += char_delta
        self._log_cluster(
            "group_absorbed_into_block",
            block_id=block.block_id,
            group_id=group.group_id,
            members=len(fragment_ids),
        )

    def _reevaluate_groups_for_block(self, block_id: str) -> bool:
        block = self.state.blocks.get(block_id)
        if not block:
            return False
        block_vec = self._ensure_block_embedding(block)
        if not block_vec:
            return False
        candidates: List[Tuple[float, Group]] = []
        for group in self.state.groups.values():
            if group.state == GroupState.RETIRED or not group.members:
                continue
            group_vec = self._ensure_group_embedding(group)
            if not group_vec:
                continue
            distance = cosine_distance(block_vec, group_vec)
            if distance <= self.block_distance_threshold:
                candidates.append((distance, group))
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0])
        for _, group in candidates:
            block = self.state.blocks.get(block_id)
            if not block:
                break
            self._absorb_group_into_block(block, group)
        block = self.state.blocks.get(block_id)
        if block:
            self._refresh_block_embedding(block)
        return True

    def merge_blocks(self, source_block_id: str, target_block_id: str) -> Block:
        if source_block_id == target_block_id:
            block = self.state.blocks.get(target_block_id)
            if not block:
                raise BlockNotFoundError(target_block_id)
            return block
        source = self.state.blocks.get(source_block_id)
        target = self.state.blocks.get(target_block_id)
        if not source:
            raise BlockNotFoundError(source_block_id)
        if not target:
            raise BlockNotFoundError(target_block_id)
        moved_fragments = list(source.contents)
        if moved_fragments:
            target.add_contents(moved_fragments)
            for fid in moved_fragments:
                self._fragment_to_block[fid] = target_block_id
                fragment = self.state.fragments.get(fid)
                if fragment:
                    target.character_count += self._fragment_text_length(fragment)
                    self._tag_fragment_with_block(fragment, target)
        if source.position:
            if target.position:
                target.position = _union_bbox([target.position, source.position])
            else:
                target.position = source.position
        for relationship in list(source.relationships):
            other_id = relationship.target_block_id
            if other_id in {source_block_id, target_block_id}:
                continue
            try:
                self.connect_blocks(
                    target_block_id,
                    other_id,
                    relationship.rel_type,
                    score=relationship.score,
                    metadata=relationship.metadata,
                )
            except KeyError:
                continue
        self.state.remove_block(source_block_id)
        self._block_incoming_counts.pop(source_block_id, None)
        self._log_cluster(
            "block_merged",
            source_block_id=source_block_id,
            target_block_id=target_block_id,
            fragments=len(moved_fragments),
        )
        self._refresh_block_embedding(target)
        self._maybe_refresh_summary(target_block_id, force=True, allow_group_scan=False)
        return target

    def _handle_merge_instructions(self, current_block_id: str, instructions: object) -> None:
        for source_id, target_id in self._coerce_merge_pairs(current_block_id, instructions):
            if source_id == target_id:
                continue
            try:
                self.merge_blocks(source_id, target_id)
            except (BlockNotFoundError, KeyError) as exc:
                print(f"[graph][merge] failed: {exc}")

    def _coerce_merge_pairs(self, current_block_id: str, instructions: object) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if not instructions:
            return pairs

        def append_pair(src: Optional[str], tgt: Optional[str]) -> None:
            if not src or not tgt or src == tgt:
                return
            pairs.append((str(src).strip(), str(tgt).strip()))

        items = instructions if isinstance(instructions, list) else [instructions]
        for item in items:
            if isinstance(item, str):
                append_pair(current_block_id, item)
                continue
            if isinstance(item, dict):
                source = item.get("source") or item.get("from")
                target = item.get("target") or item.get("into")
                if source and not target:
                    target = current_block_id
                if target and not source:
                    source = current_block_id
                append_pair(source, target)
        return pairs

    def _effective_block_threshold(self, block: Block) -> float:
        base = self.block_distance_threshold
        excess = max(block.character_count - self._BLOCK_CHAR_THRESHOLD, 0)
        if excess <= 0:
            return base
        ratio = excess / self._BLOCK_CHAR_THRESHOLD
        penalty_factor = 1.0 + min(ratio, 3.0) * self._BLOCK_CHAR_PENALTY
        return base / penalty_factor

    # ------------------------------- Logging helpers -------------------------------- #

    def _log_cluster(self, event: str, **payload) -> None:
        logger = getattr(self, "cluster_logger", None)
        if not logger:
            return
        try:
            logger.log(event, payload)
        except Exception:
            pass

    @staticmethod
    def _shorten_text(text: Optional[str], limit: int = 80) -> str:
        if not text:
            return ""
        trimmed = text.strip()
        if len(trimmed) <= limit:
            return trimmed
        return trimmed[:limit] + "…"

    def _match_block(
        self,
        feature_vec: FeatureVector,
        *,
        fragment_id: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        best_block_id = None
        best_distance = float("inf")
        best_threshold = self.block_distance_threshold
        for block in self.state.blocks.values():
            block_embedding = self._ensure_block_embedding(block)
            if not block_embedding:
                continue
            distance = cosine_distance(feature_vec, block_embedding)
            if fragment_id:
                self._log_cluster(
                    "match_block_candidate",
                    fragment_id=fragment_id,
                    candidate_block_id=block.block_id,
                    distance=round(distance, 6),
                )
            if distance < best_distance:
                best_distance = distance
                best_block_id = block.block_id
                best_threshold = self._effective_block_threshold(block)
        if best_block_id is None or best_distance > best_threshold:
            return None, best_distance
        if fragment_id:
            self._log_cluster(
                "match_block_best",
                fragment_id=fragment_id,
                block_id=best_block_id,
                distance=round(best_distance, 6),
                threshold=best_threshold,
            )
        return best_block_id, best_distance

    def _match_group(
        self,
        feature_vec: FeatureVector,
        *,
        fragment_id: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        best_group_id = None
        best_distance = float("inf")
        for group in self.state.groups.values():
            if group.state == GroupState.RETIRED or not group.prototype_vec:
                continue
            distance = cosine_distance(feature_vec, group.prototype_vec)
            if fragment_id:
                self._log_cluster(
                    "match_group_candidate",
                    fragment_id=fragment_id,
                    candidate_group_id=group.group_id,
                    distance=round(distance, 6),
                )
            if distance < best_distance:
                best_distance = distance
                best_group_id = group.group_id
        if best_group_id is None or best_distance > self.group_distance_threshold:
            return None, best_distance
        if fragment_id:
            self._log_cluster(
                "match_group_best",
                fragment_id=fragment_id,
                group_id=best_group_id,
                distance=round(best_distance, 6),
                threshold=self.group_distance_threshold,
            )
        return best_group_id, best_distance

    def _should_promote_group(self, group_id: str) -> bool:
        group = self.state.groups.get(group_id)
        if not group or not self.summarizer:
            return False
        member_count = len(group.members)
        if member_count >= self.auto_promote_group_size:
            return True
        touches = self._group_touch_counts.get(group_id, 0)
        min_members_for_touches = max(3, math.ceil(self.auto_promote_group_size * 0.6))
        return touches >= self.auto_promote_group_size and member_count >= min_members_for_touches

    def _refresh_block_embedding(self, block: Block) -> None:
        embedding = self._compute_block_embedding(block.contents)
        if embedding:
            block.embedding = embedding

    def _compute_block_embedding(self, fragment_ids: Iterable[str]) -> Optional[List[float]]:
        vectors: List[List[float]] = []
        weights: List[float] = []
        for fid in fragment_ids:
            fragment = self.state.fragments.get(fid)
            if fragment and fragment.feature_vec:
                vec = list(fragment.feature_vec)
                weight = self._fragment_embed_weight(fragment)
                if weight <= 0:
                    continue
                vectors.append(vec)
                weights.append(weight)
        if not vectors:
            return None
        dims = len(vectors[0])
        avg = [0.0] * dims
        total_weight = 0.0
        for vec, weight in zip(vectors, weights):
            if len(vec) != dims:
                continue
            total_weight += weight
            for idx in range(dims):
                avg[idx] += vec[idx] * weight
        if total_weight == 0:
            return None
        for idx in range(dims):
            avg[idx] /= total_weight
        return avg

    def _ensure_block_embedding(self, block: Block) -> Optional[List[float]]:
        if block.embedding is not None:
            return list(block.embedding)
        embedding = self._compute_block_embedding(block.contents)
        if embedding is not None:
            block.embedding = embedding
            return list(embedding)
        if block.summary and self.embedder:
            embedding = list(self.embedder.embed(block.summary))
            block.embedding = embedding
            return list(embedding)
        return None

    def _tag_fragment_with_block(self, fragment: Fragment, block: Block) -> None:
        payload = fragment.payload or {}
        graph_meta = dict(payload.get("graph") or {})
        graph_meta["blockId"] = block.block_id
        graph_meta["blockLabel"] = block.label
        payload["graph"] = graph_meta
        payload["label"] = block.label
        fragment.payload = payload

    def _create_block_from_fragment(self, fragment: Fragment) -> Block:
        if not self.summarizer:
            raw_label = (fragment.text or '').strip() or f'Block {fragment.fragment_id[:6]}'
            label = raw_label[:36]
            summary = raw_label[:220]
        else:
            label, summary = self.summarizer.propose_block([fragment])
        bbox = fragment.bbox
        block_id = self._generate_block_id()
        block = Block(
            block_id=block_id,
            label=label or f'Block {block_id[-4:]}',
            summary=summary or label or 'New block',
            position=bbox,
            contents={fragment.fragment_id},
        )
        block.last_summary_member_count = 1
        block.last_summary_ts = datetime.utcnow()
        block.character_count = self._fragment_text_length(fragment)
        self.state.add_block(block)
        self._fragment_to_block[fragment.fragment_id] = block_id
        self._fragment_to_group.pop(fragment.fragment_id, None)
        self._tag_fragment_with_block(fragment, block)
        self._refresh_block_embedding(block)
        return block

    def _ensure_feature_vector(self, fragment: Fragment) -> List[float]:
        if fragment.feature_vec is not None:
            return list(fragment.feature_vec)
        components: List[float] = []
        semantic_norm = 1.0
        if fragment.fragment_type == FragmentType.TEXT:
            if not self.embedder:
                raise RuntimeError("TextEmbedder is required for text fragments")
            text = fragment.text or ""
            emphasis = self._text_emphasis_factor(fragment)
            embedding = list(self.embedder.embed(text))
            if emphasis != 1.0:
                embedding = [value * emphasis for value in embedding]
            semantic_norm = max(math.sqrt(sum(value * value for value in embedding)), 1e-6)
            components.extend(embedding)
            size_feature, weight_feature = self._text_style_features(fragment)
            components.append(size_feature)
            components.append(weight_feature)
        else:
            components.append(0.0)
            components.append(0.0)

        components.extend(self._weighted_bbox(fragment.bbox, semantic_norm))
        components.append(self._weighted_timestamp(fragment.timestamp, semantic_norm))
        components.append(self._type_indicator(fragment.fragment_type))
        return components

    def _normalize_bbox(self, bbox: Optional[BBox]) -> List[float]:
        if not bbox:
            return [0.0, 0.0, 0.0, 0.0]
        width, height = self.canvas_size
        width = max(width or 1.0, 1.0)
        height = max(height or 1.0, 1.0)
        x0, y0, x1, y1 = bbox

        def squash(value: float, scale: float) -> float:
            return math.tanh(value / scale)

        return [
            squash(x0, width),
            squash(y0, height),
            squash(x1, width),
            squash(y1, height),
        ]

    def _normalize_timestamp(self, timestamp: Optional[datetime]) -> float:
        if not timestamp:
            return 0.0
        if not self._time_anchor:
            self._time_anchor = timestamp
        delta = (timestamp - self._time_anchor).total_seconds()
        return max(delta, 0.0) / 3600.0  # hours since start

    # -------------------------- feature weighting helpers -------------------------- #

    _TIME_CLAMP_HOURS = 24.0
    _TYPE_SCALE = 0.05
    _FONT_SIZE_BASE = 18.0
    _FONT_SIZE_MAX = 120.0
    _FONT_WEIGHT_EMPHASIS = 3
    _HEADING_FONT_SIZE = 34.0
    _HEADING_FONT_WEIGHT = 700
    _HEADING_MAX_TEXT_LEN = 80
    _HEADING_MAX_LINES = 2
    _SPATIAL_FALLBACK_SCALE = 0.2
    _TIME_FALLBACK_SCALE = 0.05
    _BLOCK_CHAR_THRESHOLD = 5000
    _BLOCK_CHAR_PENALTY = 0.35

    def _weighted_bbox(self, bbox: Optional[BBox], semantic_norm: float) -> List[float]:
        raw = self._normalize_bbox(bbox)
        return self._scale_aux_vector(
            raw,
            semantic_norm,
            self.spatial_target_ratio,
            self._SPATIAL_FALLBACK_SCALE,
        )

    def _weighted_timestamp(self, ts: Optional[datetime], semantic_norm: float) -> float:
        hours = self._normalize_timestamp(ts)
        clamped = min(hours, self._TIME_CLAMP_HOURS)
        return self._scale_aux_scalar(
            clamped,
            semantic_norm,
            self.time_target_ratio,
            self._TIME_FALLBACK_SCALE,
        )

    def _type_indicator(self, fragment_type: FragmentType) -> float:
        return self._TYPE_SCALE if fragment_type == FragmentType.TEXT else 0.0

    def _scale_aux_vector(
        self,
        values: Sequence[float],
        semantic_norm: float,
        target_ratio: float,
        fallback_scale: float,
    ) -> List[float]:
        raw_norm = math.sqrt(sum(v * v for v in values))
        if raw_norm <= 1e-9 or semantic_norm <= 1e-9:
            scale = fallback_scale
        else:
            scale = (target_ratio * semantic_norm) / raw_norm
        return [v * scale for v in values]

    def _scale_aux_scalar(
        self,
        value: float,
        semantic_norm: float,
        target_ratio: float,
        fallback_scale: float,
    ) -> float:
        raw_norm = abs(value)
        if raw_norm <= 1e-9 or semantic_norm <= 1e-9:
            return value * fallback_scale
        scale = (target_ratio * semantic_norm) / raw_norm
        return value * scale

    def _text_emphasis_factor(self, fragment: Fragment) -> float:
        size, weight_token = self._extract_font_meta(fragment)
        norm_size = min(max(size, 0.0), self._FONT_SIZE_MAX)
        size_bonus = max(norm_size - self._FONT_SIZE_BASE, 0.0) / max(self._FONT_SIZE_BASE, 1.0)
        emphasis = 1.0 + 3 * min(size_bonus, 6.0)
        if weight_token >= 600:
            emphasis *= self._FONT_WEIGHT_EMPHASIS
        return emphasis

    def _text_style_features(self, fragment: Fragment) -> Tuple[float, float]:
        size, weight_token = self._extract_font_meta(fragment)
        size_feature = min(max(size, 0.0), self._FONT_SIZE_MAX) / self._FONT_SIZE_MAX
        weight_feature = 1.0 if weight_token >= 600 else 0.0
        return size_feature, weight_feature

    def _fragment_embed_weight(self, fragment: Fragment) -> float:
        if fragment.fragment_type != FragmentType.TEXT:
            return 1.0
        return self._text_emphasis_factor(fragment)

    def _extract_font_meta(self, fragment: Fragment) -> Tuple[float, int]:
        payload = fragment.payload or {}
        meta = payload.get("meta")
        style = payload.get("style")
        font_size = None
        font_weight = None
        if isinstance(meta, dict):
            font_size = meta.get("fontSize") or meta.get("fontsize")
            font_weight = meta.get("fontWeight") or meta.get("font_weight")
        if font_size is None and isinstance(style, dict):
            font_size = style.get("fontSize")
        if font_weight is None and isinstance(style, dict):
            font_weight = style.get("fontWeight")
        try:
            size_value = float(font_size)
        except (TypeError, ValueError):
            size_value = self._FONT_SIZE_BASE
        weight_value = 400
        if isinstance(font_weight, str):
            token = font_weight.strip().lower()
            if token.isdigit():
                weight_value = int(token)
            elif token == "bold":
                weight_value = 700
        elif isinstance(font_weight, (int, float)):
            weight_value = int(font_weight)
        return size_value, weight_value

    def _fragment_text_length(self, fragment: Optional[Fragment]) -> int:
        if not fragment or fragment.fragment_type != FragmentType.TEXT:
            return 0
        if fragment.text:
            return len(fragment.text)
        payload = fragment.payload if isinstance(fragment.payload, dict) else {}
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if isinstance(meta, dict):
            summary = meta.get("summary")
            if isinstance(summary, str):
                return len(summary)
            text_meta = meta.get("text")
            if isinstance(text_meta, str):
                return len(text_meta)
        return 0

    def _is_heading_fragment(self, fragment: Fragment) -> bool:
        payload = fragment.payload or {}
        graph_meta = payload.get("graph")
        if isinstance(graph_meta, dict):
            flag = graph_meta.get("isHeading")
            if isinstance(flag, bool):
                return flag
        size, weight = self._extract_font_meta(fragment)
        text_val = (fragment.text or "").strip()
        line_count = text_val.count("\n") + 1 if text_val else 1
        concise = line_count <= self._HEADING_MAX_LINES
        short_text = (not text_val) or len(text_val) <= self._HEADING_MAX_TEXT_LEN
        if size >= self._HEADING_FONT_SIZE and concise and short_text:
            return True
        if size >= 28 and weight >= self._HEADING_FONT_WEIGHT and concise and (not text_val or len(text_val) <= 60):
            return True
        return False

    def _generate_group_id(self) -> str:
        return f"group_{uuid.uuid4().hex[:8]}"

    def _generate_block_id(self) -> str:
        return f"block_{uuid.uuid4().hex[:8]}"

    def _get_group(self, group_id: str) -> Group:
        group = self.state.groups.get(group_id)
        if not group:
            raise GroupNotFoundError(group_id)
        return group

    def _refresh_block_bbox(self, block: Block) -> Optional[BBox]:
        bboxes = [self.state.fragments[fid].bbox for fid in block.contents if self.state.fragments[fid].bbox]
        return _union_bbox(bboxes) if bboxes else block.position

    def _maybe_refresh_summary(self, block_id: str, force: bool = False, *, allow_group_scan: bool = True) -> None:
        block = self.state.blocks.get(block_id)
        if not block:
            raise BlockNotFoundError(block_id)
        if not self.summarizer:
            return
        member_count = len(block.contents)
        previous = block.last_summary_member_count or 1
        ratio = (member_count - previous) / previous
        elapsed = datetime.utcnow() - block.last_summary_ts
        if not force:
            refresh_needed = ratio >= self.summary_refresh_ratio or elapsed >= self.summary_refresh_interval
            if not refresh_needed:
                return
        fragments = [self.state.fragments[fid] for fid in block.contents]
        summary_payload = self.summarizer.refine_summary(block, fragments)
        if isinstance(summary_payload, dict):
            summary_text = str(summary_payload.get("summary") or block.summary or "").strip()
            relationships = summary_payload.get("relationships")
            merge_payload = summary_payload.get("merge")
        else:
            summary_text = str(summary_payload or block.summary or "").strip()
            relationships = None
            merge_payload = None

        if summary_text:
            block.summary = summary_text[:220]
        now = datetime.utcnow()
        block.revision += 1
        block.last_summary_member_count = member_count
        block.last_summary_ts = now
        block.updated_at = now

        if relationships is not None:
            try:
                annotation = {"summary": block.summary, "relationships": relationships}
                self.register_block_annotation(block_id, annotation)
            except Exception as exc:
                print(f"[graph][summary] failed to register relationships for {block_id}: {exc}")
        else:
            # ensure metadata updated even when no relationships provided
            block.last_summary_member_count = member_count
            block.last_summary_ts = now
            block.updated_at = now
        if merge_payload:
            self._handle_merge_instructions(block_id, merge_payload)
        if allow_group_scan and self._reevaluate_groups_for_block(block_id):
            self._maybe_refresh_summary(block_id, force=True, allow_group_scan=False)

    def get_block_id_for_fragment(self, fragment_id: str) -> Optional[str]:
        return self._fragment_to_block.get(fragment_id)

    def get_group_id_for_fragment(self, fragment_id: str) -> Optional[str]:
        return self._fragment_to_group.get(fragment_id)

    def get_group_touch_count(self, group_id: str) -> int:
        return self._group_touch_counts.get(group_id, 0)

    def register_block_annotation(self, block_id: str, annotation: Dict[str, object]) -> None:
        """
        Update block-level metadata from an LLM generated annotation payload.
        Expected keys: `summary`, `label`, `relationships`.
        """
        block = self.state.blocks.get(block_id)
        if not block:
            raise BlockNotFoundError(block_id)

        label = annotation.get("label")
        if isinstance(label, str) and label.strip():
            block.label = label.strip()
        summary = annotation.get("summary")
        if isinstance(summary, str) and summary.strip():
            block.summary = summary.strip()

        relationships = annotation.get("relationships")
        if isinstance(relationships, list):
            block.relationships.clear()
            for rel in relationships:
                try:
                    rel_type = BlockRelationshipType(rel["type"])
                    target = str(rel["target"])
                except Exception:
                    continue
                score = float(rel.get("score", 1.0))
                metadata = {k: v for k, v in rel.items() if k not in {"type", "target", "score"}}
                try:
                    self.connect_blocks(block_id, target, rel_type, score=score, metadata=metadata)
                except KeyError:
                    continue
        for fragment_id in block.contents:
            fragment = self.state.fragments.get(fragment_id)
            if fragment:
                self._tag_fragment_with_block(fragment, block)
        merge_payload = annotation.get("merge")
        if merge_payload:
            self._handle_merge_instructions(block_id, merge_payload)
        if block_id in self.state.blocks:
            self._reevaluate_groups_for_block(block_id)
