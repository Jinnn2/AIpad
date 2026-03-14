# AIPAD Workflow

This document is the engineering workflow reference for the current `AIPAD` codebase.

It is intentionally more concrete than the README:

- it describes the actual runtime call chain,
- names the modules that own each responsibility,
- explains how data moves between frontend, backend, semantic graph, and project storage,
- and documents the current architectural conventions and extension points.

The primary runtime product lives in `lineart-board/`. Root-level `data/`, `experiments/`, and `results/` are auxiliary research or offline-processing folders, not the main interactive app.

---

## 1. System Scope

`AIPAD` is an AI-first whiteboard / note canvas with four tightly coupled layers:

1. Interactive canvas authoring.
2. LLM generation (`light`, `full`, `vision`).
3. Semantic maintenance (`fragment -> group -> block`).
4. Local project persistence (`current`, `commit`, preview snapshot, reopen / checkout).

The system is not a single prompt wrapper. It behaves more like a small runtime:

- the frontend owns interaction state and preview staging,
- the backend owns request validation, session state, and LLM routing,
- the semantic graph owns long-lived structure,
- the project store owns durable storage for reopen / save / commit.

---

## 2. Repository Map

## 2.1 Runtime Core

```text
lineart-board/
  app/
    main.py                 # FastAPI entrypoint and route layer
    schemas.py              # shared request/response models
    llm_client.py           # OpenAI-compatible chat wrapper + logging
    prompting.py            # mode dispatcher
    prompt_agents/          # FULL / LIGHT / VISION prompt builders
    agents/                 # planner / summarizer / vision backend adapters
    graph_runtime.py        # semantic runtime bridge
    session_store.py        # in-memory session storage
    project_store.py        # on-disk project and commit storage
  semantic_graph/
    block_manager.py        # fragment/group/block maintenance
    vision.py               # pending stroke grouping + vision routing
    orchestrator.py         # planner context construction + plan parsing
    executor.py             # planner output -> FULL/LIGHT context execution
    state.py                # graph state container
    models.py               # block/group/fragment dataclasses
  src/
    LineArtBoard.tsx        # frontend state machine and network orchestration
    LineArtUI.tsx           # presentation components and controls
    ai/                     # payload validation / normalization / planning
    textbox/                # text layout and markdown rendering
```

## 2.2 Auxiliary Folders

- `README.md`: product-facing overview.
- root startup note (`*.md` command note): minimal launch instructions for local development.
- `experiments/`: offline scripts for evaluation / data generation.
- `data/`: dataset construction utilities.
- `results/`: experiment outputs.

---

## 3. Technology Stack

## 3.1 Frontend

- React 19
- TypeScript
- Vite
- Konva / React-Konva

## 3.2 Backend

- FastAPI
- Pydantic v2
- OpenAI Python SDK (OpenAI-compatible endpoint)
- python-dotenv

## 3.3 Storage

- Session state: in-memory only
- Project state: local filesystem under `lineart-board/data/projects/`
- Logs: local JSON files under `lineart-board/logs/` or configured log dir

---

## 4. Core Runtime Concepts

## 4.1 Stroke Protocol

The universal protocol between frontend, backend, and LLM is `AIStrokePayload` in `lineart-board/app/schemas.py`.

Important fields:

- `version`
- `intent`
- `canvas`
- `replace`
- `strokes[]`

Each stroke is `AIStrokeV11`:

- `id`
- `tool`
- `points`
- `style`
- `meta`

Supported tools currently include:

- `pen`
- `line`
- `poly`
- `ellipse`
- `text`
- `edit`

This protocol is used for:

- user-authored strokes,
- LLM output,
- frontend preview planning,
- session synchronization,
- project persistence,
- graph ingestion.

## 4.2 Frontend Render Model

The frontend does not render raw protocol strokes directly.

It converts them into `ShapeDraft` objects in `src/ai/plan.ts`, which are Konva-friendly render structures:

- geometric primitives become shape drafts,
- text is laid out through `textbox/layout.ts`,
- markdown-like text is flattened for canvas rendering,
- edit strokes become overlay edit drafts until accepted.

## 4.3 Semantic Graph Model

The semantic graph is maintained in `semantic_graph/` and uses three primary entities:

- `Fragment`: smallest semantic unit, usually one text item or one stroke fragment.
- `Group`: pending or intermediate cluster of related fragments.
- `Block`: stable semantic unit with label, summary, contents, and relationships.

Current semantic hierarchy:

```text
raw stroke/text
  -> fragment
  -> pending group
  -> promoted block
  -> relationships to other blocks
```

## 4.4 Session Model

`app/session_store.py` stores transient runtime state:

- `sid`
- minified stroke window for LLM context
- full stroke snapshot for exact reopen/sync behavior
- `graph_auto` flag
- `graph_runtime`
- bound `project_id`

Sessions are process-local and non-durable.

## 4.5 Project Model

`app/project_store.py` stores durable project state:

- current canvas
- current graph state
- runtime flags
- current preview image
- immutable commits
- WAL operations

This turns the app from a stateless demo into a resumable workspace.

---

## 5. Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `src/LineArtBoard.tsx` | Owns canvas state, pointer tools, AI request lifecycle, Auto Maintain toggles, project flows |
| `src/LineArtUI.tsx` | Stateless-ish UI layer: toolbar, settings, drawers, sidebars, graph panels |
| `src/ai/normalize.ts` | Client-side payload validation and normalization |
| `src/ai/plan.ts` | Converts protocol payload to drawable drafts |
| `src/textbox/layout.ts` | Text wrapping, grow direction, auto expand / shrink |
| `app/main.py` | FastAPI route layer, request logging, `/suggest` branching, payload cleaning |
| `app/llm_client.py` | Calls model, injects multimodal image content, writes raw LLM debug logs |
| `app/prompt_agents/*.py` | Prompt builders for FULL / LIGHT / VISION |
| `app/agents/*.py` | Planner backend, block summarizer, vision backend adapters |
| `app/graph_runtime.py` | Semantic runtime wrapper, graph ingest, planning, executor bridge, state dump/load |
| `semantic_graph/block_manager.py` | Text/stroke registration, clustering, block attach/create/promote, summary refresh |
| `semantic_graph/vision.py` | Stroke-level pending group batching and vision decisions |
| `semantic_graph/orchestrator.py` | Planner prompt construction and plan state transitions |
| `semantic_graph/executor.py` | Builds FULL/LIGHT context from planner result |
| `app/project_store.py` | Local project / commit filesystem storage |

---

## 6. End-to-End Workflows

## 6.1 Human Drawing / Editing Workflow

### Overview

This is the baseline path before any AI is involved.

### Flow

1. User interacts on the Konva stage in `LineArtBoard.tsx`.
2. Pointer handlers (`onMouseDown`, `onMouseMove`, `onMouseUp`) update local authoring state.
3. Completed authoring actions are written into:
   - `shapes` for rendering,
   - `drawStack` for protocol-level replay / sync.
4. `drawStack` is the canonical frontend source for backend synchronization.
5. A debounced sync posts current strokes to `/session/sync`.

### Important implementation detail

The frontend keeps both:

- a visual representation (`ShapeDraft`),
- and a protocol representation (`AIStrokeV11`).

This split is critical because:

- Konva rendering needs geometry/layout objects,
- the backend and LLM need protocol JSON.

---

## 6.2 Ask AI Workflow

The `Ask AI` button is implemented in `LineArtBoard.tsx` through `askAI`.

### Frontend sequence

1. Ensure `sid` exists via `/session/init`.
2. Build `delta` from newly added strokes since last request.
3. Build `context` from the full current `drawStack`.
4. Attach runtime knobs:
   - `mode`
   - `hint`
   - `gen_scale`
   - `temperature`
   - `top_p`
   - `max_tokens`
   - `group_promote_mode`
   - `vision_image_mode`
   - `prefer_explanatory_drawing`
5. POST to `/suggest`.
6. Receive payload.
7. Client-side validate -> normalize -> `planDrafts`.
8. Stage result into `previews`.
9. User explicitly `Accept` or `Dismiss`.

### Accept vs dismiss

- `Accept`: append drafts into `shapes` and `drawStack`, then normal session/graph sync takes over.
- `Dismiss`: clear preview only; canvas state is unchanged.

This separation means LLM output is not durable until user acceptance.

---

## 6.3 `/suggest` Backend Workflow

`/suggest` in `app/main.py` is the central routing point.

### High-level branches

There are three major execution branches:

1. Direct prompting (`light` / `full` / legacy `vision`).
2. Context executor path when Auto Maintain is enabled.
3. Vision 2.0 two-step flow.

### Request preprocessing

Before prompting:

- per-request log folder may be created,
- `hint` is suppressed when auto-complete is on,
- session state is updated from `delta` and `context`,
- graph runtime options may be updated dynamically.

### Direct prompting path

When semantic auto mode is off:

1. Build messages via `app/prompting.py`.
2. Call LLM through `app/llm_client.py`.
3. Run payload cleaning in `main.py`.
4. Validate via `AIStrokePayload`.
5. Return cleaned payload.

### Context executor path

When session graph auto mode is on and mode is `full` or `light`:

1. Sync current strokes into `GraphRuntime`.
2. Call `graph_runtime.run_conversation(...)`.
3. Receive:
   - planner output
   - LLM payload
4. Persist plan bundle into logs.
5. Clean and validate output.
6. Return payload plus planner metadata.

### Payload cleaning

Before returning to frontend, backend normalizes model output:

- invalid color aliases are mapped to allowed palette or fallback black,
- point arrays are clamped and reduced,
- tool-specific shape constraints are enforced,
- malformed strokes are dropped with rule-level diagnostics,
- total failure yields 502 with failed-rule summary.

This is why the frontend usually receives a valid payload even when the raw LLM output is imperfect.

---

## 6.4 Vision 2.0 Workflow

Vision 2.0 is a two-step path inside `/suggest`.

## Step 1: Image inspection

- frontend captures canvas image,
- backend builds a vision-only prompt via `prompt_agents/vision_v2.py`,
- model returns:
  - `analysis`
  - `instruction`

No strokes are returned in step 1.

## Step 2: Guided generation

- backend merges original hint + analysis + instruction,
- then reuses the FULL prompting path,
- model returns actual strokes.

This lets image understanding and stroke generation be decoupled.

---

## 6.5 Auto Complete Workflow

Auto Complete is a frontend-side timer feature.

### Behavior

1. After user activity, `noteUserAction()` may start a 5-second timer.
2. If preview is active, auto complete pauses.
3. When timer expires, frontend programmatically triggers `askAI()`.

Important detail:

- Auto Complete changes when requests fire.
- It does not bypass preview / accept / dismiss semantics.
- When auto-complete is on, user hint may be treated as absent so the model infers intent from latest canvas changes.

---

## 6.6 Text Box Workflow

Text authoring is a first-class path, not a post-hoc overlay.

### Creation / edit

1. User creates or selects a text box.
2. `openTextEditor()` populates `TextEditorState`.
3. `commitTextEditor()` computes final layout using `computeTextBoxLayout`.
4. Text is written back as a normal `text` stroke + `ShapeDraft`.

### Important text metadata

Each text element stores structured metadata such as:

- `text`
- `summary`
- `fontFamily`
- `fontWeight`
- `fontSize`
- `role`
- `growDir`
- computed layout details

### Markdown handling

Backend stores the original text source.

Frontend detects and renders a supported markdown subset:

- headings
- lists
- bold
- inline code

There is no separate markdown fragment type; markdown is a rendering concern on top of the text payload.

### Text completion

Typing `:::` can trigger `/completion`, which returns pure text continuation rather than a full stroke payload.

---

## 6.7 Auto Maintain Workflow

Auto Maintain is the most important nontrivial runtime in the project.

It is enabled from the frontend by `/graph/auto-mode`.

### Activation sequence

1. Frontend ensures session exists.
2. Frontend sends:
   - full stroke snapshot
   - canvas size
   - optional graph image snapshot
3. Backend creates `GraphRuntime` on the session.
4. `graph_auto = true` becomes the request routing condition for planner/executor mode.

### Runtime maintenance loop

Once enabled:

1. Frontend keeps syncing strokes through `/session/sync`.
2. Frontend polls `/graph/state`.
3. Backend incrementally ingests new fragments.
4. Pending stroke groups and semantic blocks evolve continuously.

---

## 7. Semantic Graph Runtime Internals

## 7.1 GraphRuntime

`app/graph_runtime.py` is the integration hub between FastAPI sessions and `semantic_graph/`.

It wires together:

- `BlockManager`
- `VisionGrouper`
- `ConversationOrchestrator`
- `ContextExecutor`
- block summarizer
- planner backend
- vision backend
- embedding client

It also owns:

- graph state import/export,
- manual promotion / selection actions,
- NOOP placeholder generation,
- project persistence compatibility,
- environment-driven tuning.

## 7.2 Fragment Ingestion

`ingest_strokes()` and `sync_strokes_snapshot()` convert accepted strokes into graph fragments.

Current rules:

- text fragments are embedding-aware and may match groups/blocks semantically,
- stroke fragments first enter unlabeled / pending vision flow unless explicitly attached,
- explicit AI graph hints (`targetBlockId`, `blockIntent`, `proposalKey`) can affect routing.

## 7.3 BlockManager

`semantic_graph/block_manager.py` owns semantic clustering and block evolution.

Important behaviors:

- text can cold-start a new block,
- heading-like text is promoted aggressively,
- text can match existing block by embedding distance,
- otherwise it may join or create a group,
- groups can later promote to blocks,
- block summaries and relationships are refreshed via summarizer agent,
- manual overrides can pin fragments to blocks.

Current design intent:

- distance matters,
- semantic similarity matters,
- but block creation and attachment can also be guided by model-provided `meta.graph`.

## 7.4 VisionGrouper

`semantic_graph/vision.py` handles non-text stroke grouping before full semantic promotion.

### Current logic

1. Stroke fragments are spatially clustered into pending vision groups.
2. Groups become ready when:
   - they are explicitly ready,
   - or sufficiently stale and large enough,
   - or manual promotion is requested.
3. Ready groups are converted to `VisionPayload`.
4. Vision backend decides:
   - merge into block,
   - or create/promote as new diagram block.

The current image policy is configurable with:

- `off`
- `auto`
- `always`

`auto` is currently used to keep token cost down and only attach images when the stroke group is complex or ambiguous.

## 7.5 Planner

`semantic_graph/orchestrator.py` builds the planner prompt.

It feeds the planner:

- `LATEST_FRAGMENT`
- `LATEST_CONTEXT`
- current focused block
- related blocks
- related groups
- user input if any

Planner returns:

- `action`
- `targetBlockIds`
- `comment`
- `nextStepHint`

Supported actions:

- `CONTINUE`
- `NOOP`
- `SWITCH`
- `OPEN_RELATED`
- `CLOSE`

## 7.6 Executor

`semantic_graph/executor.py` converts planner result into the final FULL/LIGHT request.

Current behavior:

1. Select seed entities from:
   - `plan.targetBlockIds`
   - then active context
   - then main context
   - then recent blocks as fallback
2. Expand related blocks.
3. Build `block_outline`.
4. Build final context payload.
5. Call FULL or LIGHT prompt builder.
6. Call the LLM.

Important current rule:

- primary blocks contribute full `context.strokes`,
- related blocks are still auto-expanded,
- but related blocks are injected summary-only through `block_outline`,
- this is specifically to reduce context drift caused by overloading FULL with unrelated block strokes.

## 7.7 Planner/Executor Loop

`GraphRuntime.run_conversation()` currently allows planner re-entry for `SWITCH`.

Current loop:

1. planner pass
2. if `SWITCH`, update focus and rerun
3. stop after max 3 passes
4. if `NOOP`, emit a placeholder text payload instead of running FULL
5. otherwise execute FULL/LIGHT

This prevents planner ping-pong and gives deterministic exit behavior.

---

## 8. Prompt Architecture

Prompt assembly is split by responsibility.

## 8.1 Prompt dispatcher

`app/prompting.py` only dispatches by mode.

## 8.2 Prompt agents

`app/prompt_agents/` contains the real prompt definitions:

- `full.py`
- `light.py`
- `vision.py`
- `vision_v2.py`

## 8.3 FULL mode

`full.py` is the main structured generation prompt.

It includes:

- system rules,
- output contract,
- optional samples,
- `planner_next_step`,
- `block_outline`,
- `prefer_explanatory_drawing`,
- `auto_maintain_enabled`.

Additional important behavior:

- FULL compresses pen point lists in prompt context,
- prompt behavior differs depending on whether Auto Maintain is enabled,
- graph-control metadata is only encouraged in maintain-enabled mode.

## 8.4 LIGHT mode

`light.py` is the low-token next-stroke mode.

Characteristics:

- strict single-stroke output,
- compressed minimal context,
- lighter contract,
- optimized for quick continuation.

## 8.5 VISION modes

- `vision.py`: legacy vision wrapper on top of FULL prompt.
- `vision_v2.py`: explicit two-step image understanding and instruction handoff.

---

## 9. Frontend Architecture

## 9.1 `LineArtBoard.tsx`

This file is effectively the frontend runtime controller.

It owns:

- stage interaction
- local canvas state
- undo/redo
- preview lifecycle
- AI request submission
- session sync
- graph polling
- project manager state
- save/commit flows
- text editor modal
- graph inspector overlays

It is state-heavy by design.

This file is the place to inspect when debugging:

- wrong network payloads,
- preview/accept issues,
- graph polling behavior,
- auto-complete triggers,
- project save/checkout behavior.

## 9.2 `LineArtUI.tsx`

This file is mostly presentational and should stay that way.

It contains:

- `TopToolbar`
- `SettingsButton`
- `SidePanel`
- `BottomPanel`
- `GraphBlocksDrawer`
- `AIFeedSidebar`

Design principle:

- `LineArtBoard` owns state and behavior,
- `LineArtUI` renders and forwards events.

## 9.3 Frontend graph features

Current UI graph features include:

- graph inspector toggle,
- block highlight overlays,
- relationship arrows,
- pending vision group overlays,
- manual promote actions,
- fragment box selection,
- manual block create / assign from selection,
- fragment delete path through backend graph removal route.

---

## 10. Project Persistence Workflow

The project manager is not just a screenshot saver. It persists full working state.

## 10.1 Save current

Top-bar `Save`:

1. Ensure project exists or create one.
2. Capture viewport snapshot.
3. Save:
   - `canvas.json`
   - `graph.json`
   - `runtime.json`
   - current preview image
4. Refresh project list/detail.

This updates mutable `current`.

## 10.2 Commit

Project Manager `Commit`:

1. Capture snapshot.
2. Save immutable commit folder.
3. Update project meta `currentRef`.
4. Refresh commit list.

This creates history, more like Git commit than autosave.

## 10.3 Checkout

Checkout replaces current working state with the chosen commit:

- canvas is rebuilt,
- graph runtime state is restored,
- auto-maintain mode is restored,
- graph snapshot polling resumes if needed.

## 10.4 Auto current preview

While a project is open:

- frontend marks current preview dirty when canvas changes,
- every 5 minutes, if visible and dirty, it uploads a fresh current preview,
- this does not create a commit.

---

## 11. Observability and Debugging

## 11.1 Request logs

The main debugging path is `lineart-board/logs/`.

Typical files:

- `input.request.json`
- `input.messages.json`
- `input.messages.step1.json`
- `input.messages.step2.json`
- `output.context.plan.json`
- `output.cleaned.json`
- `output.error.json`

## 11.2 Raw LLM logs

`app/llm_client.py` can persist raw LLM request/response summaries as:

- `llm_YYYYMMDD_HHMMSS.json`

These are the best files for prompt-debugging.

## 11.3 Graph state inspection

`/graph/state` exposes:

- blocks
- fragments
- groups
- pending vision groups

This is the best route for checking:

- why a fragment landed in a block,
- whether a group exists but is not promoted,
- whether the frontend graph overlay is stale.

## 11.4 Typical debug order

When a result looks wrong, inspect in this order:

1. `input.request.json`
2. planner log or `output.context.plan.json`
3. `input.messages*.json`
4. raw `llm_*.json`
5. `output.cleaned.json` or `output.error.json`
6. `/graph/state` or persisted project `graph.json`

---

## 12. Current Architectural Rules

These are important current rules that are easy to violate accidentally.

1. Frontend preview is authoritative before acceptance.
2. Accepted output is only durable after it enters `drawStack` and syncs.
3. Session state is transient; project state is durable.
4. Auto Maintain changes prompt routing, not only graph visuals.
5. Planner and executor are separate responsibilities:
   - planner decides what context is needed,
   - executor decides how to assemble and call FULL/LIGHT.
6. Related blocks are currently summary-only in FULL context expansion.
7. Markdown is stored as source text and rendered on frontend, not transformed into a separate backend type.
8. Project `current` and project `commit` are different concepts and should stay different.

---

## 13. Extension Points

If you want to modify the system cleanly, use these seams.

## 13.1 Change prompting

Edit:

- `app/prompt_agents/full.py`
- `app/prompt_agents/light.py`
- `app/prompt_agents/vision.py`
- `app/prompt_agents/vision_v2.py`

Do not put prompt logic back into `app/prompting.py`.

## 13.2 Change graph clustering

Edit:

- `semantic_graph/block_manager.py`
- `semantic_graph/vision.py`
- `app/graph_runtime.py` env knobs

## 13.3 Change planner logic

Edit:

- `semantic_graph/orchestrator.py`
- `app/agents/planner_backend.py`

## 13.4 Change FULL context assembly

Edit:

- `semantic_graph/executor.py`

This is the correct place for:

- primary vs related context policy,
- block outline formatting,
- group injection policy.

## 13.5 Change persistence

Edit:

- `app/project_store.py`
- `app/main.py`
- frontend project flows in `LineArtBoard.tsx`

---

## 14. Recommended Mental Model

The most useful way to understand AIPAD is:

```text
Frontend canvas = interaction runtime
Session store = transient transport memory
Semantic graph = structured meaning layer
Prompt agents = model interface layer
Project store = durable workspace history
```

If you keep these five layers separate when modifying the code, the project stays understandable.

If you collapse them together, debugging quickly becomes difficult.

---

## 15. Suggested Reading Order for New Contributors

If someone needs to come up to speed quickly, read in this order:

1. `README.md`
2. `workflow.md`
3. `lineart-board/app/main.py`
4. `lineart-board/src/LineArtBoard.tsx`
5. `lineart-board/app/graph_runtime.py`
6. `lineart-board/semantic_graph/orchestrator.py`
7. `lineart-board/semantic_graph/executor.py`
8. `lineart-board/semantic_graph/block_manager.py`
9. `lineart-board/app/project_store.py`
10. `lineart-board/app/prompt_agents/full.py`

That order matches the actual runtime importance of the system.
