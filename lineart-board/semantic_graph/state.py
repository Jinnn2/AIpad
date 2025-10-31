from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Block, BlockRelationship, Fragment, Group


@dataclass
class GraphState:
    fragments: Dict[str, Fragment] = field(default_factory=dict)
    groups: Dict[str, Group] = field(default_factory=dict)
    blocks: Dict[str, Block] = field(default_factory=dict)

    def clear(self) -> None:
        self.fragments.clear()
        self.groups.clear()
        self.blocks.clear()

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment

    def get_fragment(self, fragment_id: str) -> Fragment:
        try:
            return self.fragments[fragment_id]
        except KeyError as exc:
            raise KeyError(f"fragment not found: {fragment_id}") from exc

    def add_group(self, group: Group) -> None:
        self.groups[group.group_id] = group

    def add_block(self, block: Block) -> None:
        self.blocks[block.block_id] = block

    def remove_group(self, group_id: str) -> None:
        self.groups.pop(group_id, None)

    def connect_blocks(self, relationship: BlockRelationship) -> None:
        src = self.blocks.get(relationship.source_block_id)
        dst = self.blocks.get(relationship.target_block_id)
        if not src or not dst:
            raise KeyError(
                f"invalid relationship: {relationship.source_block_id} -> {relationship.target_block_id} ({relationship.rel_type})"
            )
        src.relationships = [
            rel
            for rel in src.relationships
            if not (rel.target_block_id == relationship.target_block_id and rel.rel_type == relationship.rel_type)
        ]
        src.add_relationship(relationship)
        # Mirror for dst.
        dst.relationships = [
            rel
            for rel in dst.relationships
            if not (rel.target_block_id == relationship.source_block_id and rel.rel_type == relationship.rel_type)
        ]
        mirrored = BlockRelationship(
            source_block_id=relationship.target_block_id,
            target_block_id=relationship.source_block_id,
            rel_type=relationship.rel_type,
            score=relationship.score,
            metadata=relationship.metadata.copy(),
        )
        dst.add_relationship(mirrored)

    def list_blocks(self) -> List[Block]:
        return list(self.blocks.values())

    def list_groups(self) -> List[Group]:
        return list(self.groups.values())
