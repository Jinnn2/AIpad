"""
Semantic graph orchestration package that powers the upgraded workflow.
"""

from .block_manager import BlockManager, BlockSummarizer, TextEmbedder, FragmentAssignment
from .executor import ContextExecutor, FocusContext
from .models import (
    AssistantReply,
    Block,
    BlockRelationship,
    BlockRelationshipType,
    ExecutionPlan,
    Fragment,
    FragmentType,
    Group,
)
from .orchestrator import ConversationOrchestrator, OrchestratorContext, PlanBackend
from .state import GraphState
from .vision import VisionBackend, VisionGrouper, VisionPayload, VisionResult

__all__ = [
    "AssistantReply",
    "Block",
    "BlockManager",
    "BlockRelationship",
    "BlockRelationshipType",
    "BlockSummarizer",
    "ContextExecutor",
    "ConversationOrchestrator",
    "ExecutionPlan",
    "FragmentAssignment",
    "Fragment",
    "FragmentType",
    "FocusContext",
    "GraphState",
    "Group",
    "OrchestratorContext",
    "PlanBackend",
    "TextEmbedder",
    "VisionBackend",
    "VisionGrouper",
    "VisionPayload",
    "VisionResult",
]
