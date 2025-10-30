from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

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
        group_distance_threshold: float = 0.35,
        block_distance_threshold: float = 0.28,
        summary_refresh_ratio: float = 0.3,
        summary_refresh_interval: timedelta = timedelta(minutes=10),
        canvas_size: Optional[Tuple[float, float]] = None,
        auto_promote_group_size: int = 5,
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

        self._time_anchor: Optional[datetime] = None
        self._fragment_to_group: Dict[str, str] = {}
        self._fragment_to_block: Dict[str, str] = {}
        self._unlabeled_strokes: List[str] = []
        self._block_incoming_counts: Dict[str, int] = defaultdict(int)
        self._group_touch_counts: Dict[str, int] = defaultdict(int)

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

        if not self.state.blocks and not self.state.groups:
            print(f"[graph][cluster] fragment={fragment.fragment_id} creates initial block (cold start)")
            new_block = self._create_block_from_fragment(fragment)
            assignment.status = "block"
            assignment.block_id = new_block.block_id
            assignment.promoted_block_id = new_block.block_id
            return assignment

        block_id, block_distance = self._match_block(feature_vec, fragment_id=fragment.fragment_id)
        if block_id:
            print(
                f"[graph][cluster] fragment={fragment.fragment_id} matched block={block_id} "
                f"distance={block_distance:.3f}"
            )
            block = self.attach_fragment_to_block(block_id, fragment.fragment_id)
            self._tag_fragment_with_block(fragment, block)
            assignment.status = "block"
            assignment.block_id = block_id
            return assignment

        group_id, group_distance = self._match_group(feature_vec, fragment_id=fragment.fragment_id)
        if group_id:
            print(
                f"[graph][cluster] fragment={fragment.fragment_id} matched group={group_id} "
                f"distance={group_distance:.3f}"
            )
            self._assign_to_group(fragment.fragment_id, feature_vec, allow_create=False, existing_group_id=group_id)
            assignment.status = "group"
            assignment.group_id = group_id
            self._group_touch_counts[group_id] += 1
            if self._should_promote_group(group_id):
                print(f"[graph][cluster] promote group={group_id} after assignment {fragment.fragment_id}")
                promoted_block = self.promote_group(group_id)
                assignment.status = "block"
                assignment.block_id = promoted_block.block_id
                assignment.promoted_block_id = promoted_block.block_id
            return assignment

        print(f"[graph][cluster] fragment={fragment.fragment_id} starts new block (no cluster match)")
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
        return new_group.group_id

    def _match_block(
        self,
        feature_vec: FeatureVector,
        *,
        fragment_id: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        best_block_id = None
        best_distance = float("inf")
        for block in self.state.blocks.values():
            block_embedding = self._ensure_block_embedding(block)
            if not block_embedding:
                continue
            distance = cosine_distance(feature_vec, block_embedding)
            if fragment_id:
                print(
                    f"[graph][cluster][block] fragment={fragment_id} candidate={block.block_id} "
                    f"distance={distance:.3f}"
                )
            if distance < best_distance:
                best_distance = distance
                best_block_id = block.block_id
        if best_block_id is None or best_distance > self.block_distance_threshold:
            return None, best_distance
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
                print(
                    f"[graph][cluster][group] fragment={fragment_id} candidate={group.group_id} "
                    f"distance={distance:.3f}"
                )
            if distance < best_distance:
                best_distance = distance
                best_group_id = group.group_id
        if best_group_id is None or best_distance > self.group_distance_threshold:
            return None, best_distance
        return best_group_id, best_distance

    def _should_promote_group(self, group_id: str) -> bool:
        group = self.state.groups.get(group_id)
        if not group or not self.summarizer:
            return False
        if len(group.members) >= self.auto_promote_group_size:
            return True
        touches = self._group_touch_counts.get(group_id, 0)
        return touches >= self.auto_promote_group_size

    def _refresh_block_embedding(self, block: Block) -> None:
        embedding = self._compute_block_embedding(block.contents)
        if embedding:
            block.embedding = embedding

    def _compute_block_embedding(self, fragment_ids: Iterable[str]) -> Optional[List[float]]:
        vectors: List[List[float]] = []
        for fid in fragment_ids:
            fragment = self.state.fragments.get(fid)
            if fragment and fragment.feature_vec:
                vec = list(fragment.feature_vec)
                vectors.append(vec)
        if not vectors:
            return None
        dims = len(vectors[0])
        avg = [0.0] * dims
        count = 0
        for vec in vectors:
            if len(vec) != dims:
                continue
            count += 1
            for idx in range(dims):
                avg[idx] += vec[idx]
        if count == 0:
            return None
        for idx in range(dims):
            avg[idx] /= count
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
        if fragment.fragment_type == FragmentType.TEXT:
            if not self.embedder:
                raise RuntimeError("TextEmbedder is required for text fragments")
            text = fragment.text or ""
            components.extend(self.embedder.embed(text))

        components.extend(self._normalize_bbox(fragment.bbox))
        components.append(self._normalize_timestamp(fragment.timestamp))
        components.append(1.0 if fragment.fragment_type == FragmentType.TEXT else 0.0)
        return components

    def _normalize_bbox(self, bbox: Optional[BBox]) -> List[float]:
        if not bbox:
            return [0.0, 0.0, 0.0, 0.0]
        width, height = self.canvas_size
        width = width or 1.0
        height = height or 1.0
        x0, y0, x1, y1 = bbox
        return [
            x0 / width,
            y0 / height,
            x1 / width,
            y1 / height,
        ]

    def _normalize_timestamp(self, timestamp: Optional[datetime]) -> float:
        if not timestamp:
            return 0.0
        if not self._time_anchor:
            self._time_anchor = timestamp
        delta = (timestamp - self._time_anchor).total_seconds()
        return max(delta, 0.0) / 3600.0  # hours since start

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

    def _maybe_refresh_summary(self, block_id: str, force: bool = False) -> None:
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
        block.summary = self.summarizer.refine_summary(block, fragments)
        block.revision += 1
        block.last_summary_member_count = member_count
        block.last_summary_ts = datetime.utcnow()
        block.updated_at = datetime.utcnow()

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
