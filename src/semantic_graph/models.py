from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
FeatureVector = Sequence[float]


class FragmentType(str, Enum):
    TEXT = "text"
    STROKE = "stroke"


@dataclass(slots=True)
class Fragment:
    fragment_id: str
    fragment_type: FragmentType
    bbox: Optional[BBox] = None
    text: Optional[str] = None
    timestamp: Optional[datetime] = None
    feature_vec: Optional[FeatureVector] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def centroid(self) -> Optional[Tuple[float, float]]:
        if not self.bbox:
            return None
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class GroupState(str, Enum):
    PENDING = "pending"
    STABLE = "stable"
    RETIRED = "retired"


@dataclass
class Group:
    group_id: str
    members: Set[str] = field(default_factory=set)
    prototype_vec: Optional[List[float]] = None
    state: GroupState = GroupState.PENDING
    need_llm_review: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_member(self, fragment_id: str, feature_vec: Optional[FeatureVector] = None) -> None:
        self.members.add(fragment_id)
        self.updated_at = datetime.utcnow()
        if feature_vec is not None:
            if self.prototype_vec is None:
                self.prototype_vec = list(feature_vec)
            else:
                # simple running average
                size = len(self.members)
                alpha = 1.0 / max(size, 1)
                for idx, val in enumerate(feature_vec):
                    if idx >= len(self.prototype_vec):
                        self.prototype_vec.append(val)
                    else:
                        self.prototype_vec[idx] = (1 - alpha) * self.prototype_vec[idx] + alpha * val


class BlockRelationshipType(str, Enum):
    REFINES = "refines"
    COMMENT_ON = "comment_on"
    FLOW_NEXT = "flow_next"


@dataclass
class BlockRelationship:
    source_block_id: str
    target_block_id: str
    rel_type: BlockRelationshipType
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    block_id: str
    label: str
    summary: str
    position: Optional[BBox]
    contents: Set[str] = field(default_factory=set)
    relationships: List[BlockRelationship] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    revision: int = 0
    embedding: Optional[List[float]] = None
    last_summary_member_count: int = 0
    last_summary_ts: datetime = field(default_factory=datetime.utcnow)

    def add_relationship(self, rel: BlockRelationship) -> None:
        self.relationships.append(rel)
        self.updated_at = datetime.utcnow()

    def add_contents(self, fragment_ids: Iterable[str]) -> None:
        before = len(self.contents)
        self.contents.update(fragment_ids)
        after = len(self.contents)
        if after != before:
            self.updated_at = datetime.utcnow()


@dataclass
class ExecutionPlan:
    action: str
    target_block_ids: List[str]
    comment: Optional[str] = None


@dataclass
class AssistantReply:
    assistant_reply: str
    block_annotation: Dict[str, Any]


class BlockNotFoundError(KeyError):
    pass


class FragmentNotFoundError(KeyError):
    pass


class GroupNotFoundError(KeyError):
    pass
