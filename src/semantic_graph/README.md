# Semantic Graph Upgrade

This package implements the four-module upgrade discussed for the canvas knowledge graph:

1. **BlockManager** – incrementally clusters fragments, tracks groups, promotes stable blocks, and keeps their summaries fresh.
2. **VisionGrouper** – batches unlabeled stroke fragments, asks a vision backend for structure, and merges the result back into the graph.
3. **ConversationOrchestrator** – selects the main conversation block, prepares context, and asks an LLM to return an action plan.
4. **PromptExecutor** – expands the plan into a full LLM request, applies annotations, and updates the graph.

### Key Files

| File | Purpose |
| ---- | ------- |
| `models.py` | Core dataclasses for fragments, groups, blocks, relationships, and execution outputs. |
| `state.py` | In-memory store that keeps fragments, groups, and blocks consistent. |
| `block_manager.py` | Group maintenance, promotion into blocks, relationship updates, and summary refresh logic. |
| `vision.py` | Vision pipeline wrapper that handles annotation vs. diagram results. |
| `orchestrator.py` | Module 3 orchestration logic (context selection + plan generation). |
| `executor.py` | Module 4 executor that builds prompts and applies block annotations. |
| `similarity.py` | Shared cosine-distance helper. |

### Integrating with the app

```python
from semantic_graph import (
    BlockManager,
    ConversationOrchestrator,
    PromptExecutor,
    VisionGrouper,
)

graph = BlockManager(embedder=text_embedder, summarizer=block_summarizer)
orchestrator = ConversationOrchestrator(graph, embedder=text_embedder, plan_backend=plan_backend)
executor = PromptExecutor(graph, backend=prompt_backend)
vision = VisionGrouper(graph, backend=vision_backend)
```

*Inject your own backends for embeddings, summarization, vision, and LLM execution. The package stays framework-agnostic so the existing FastAPI service can plug in without tight coupling.*

