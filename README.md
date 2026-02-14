# AIPAD (JinAgent Workspace)

This repository contains the current AIPAD implementation used for AI-assisted canvas drawing, text editing, and semantic graph maintenance.

Main runtime project: `lineart-board/`

## What Is Implemented

- Interactive canvas built with React + Konva.
- AI suggestion pipeline with three prompt modes: `light`, `full`, `vision`.
- Vision 2.0 two-step flow (image analysis -> instruction -> stroke generation).
- Text tool with editable text box, autocomplete trigger (`:::`), and style presets (`body`, `subtitle`, `title`).
- Auto Maintain semantic graph:
  - Fragment -> Group -> Block lifecycle.
  - Planner + executor context selection.
  - Block summaries and inter-block relationships.
- Robust backend cleaning/validation for model output (including invalid color fallback).
- Detailed request/response logging for debugging.

## Tech Stack

- Frontend: React 19, TypeScript, Vite, react-konva.
- Backend: FastAPI, Pydantic v2, Uvicorn.
- LLM/Embedding: OpenAI-compatible API.

## Project Layout

```text
lineart-board/
  app/
    main.py                 # FastAPI entry, /suggest and graph APIs
    prompting.py            # FULL/LIGHT/VISION prompt builders
    prompting.full.original.py
    graph_runtime.py        # graph orchestration runtime
    session_store.py        # in-memory session + stroke minify
    llm_client.py           # OpenAI chat-completions wrapper
    embedding_client.py     # embedding API wrapper
    schemas.py              # shared request/response models
  semantic_graph/
    block_manager.py        # clustering, promotion, summary refresh
    vision.py               # stroke grouping + vision decisions
    orchestrator.py         # planner logic and action parsing
    executor.py             # select blocks and build FULL/LIGHT context
    models.py, state.py
  src/
    LineArtBoard.tsx        # main UI + interaction + API wiring
    LineArtUI.tsx           # toolbar/panels
    ai/                     # payload validation/normalization/planning
    textbox/layout.ts       # text layout and grow direction logic
  logs/                     # runtime IO and LLM debug logs
```

## Quick Start

## 1) Backend

```powershell
cd lineart-board
conda activate lineart
python -m uvicorn app.main:app --reload --reload-exclude "logs" --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`

## 2) Frontend

```powershell
cd lineart-board
npm install
npm run dev
```

Frontend URL: `http://localhost:5173` (sometimes browser uses `http://[::1]:5173`)

## Core APIs

- `GET /health`
- `POST /suggest`
- `POST /completion`
- `POST /session/init`
- `POST /session/sync`
- `POST /graph/auto-mode`
- `GET /graph/state?sid=...`
- `POST /graph/promote-group`

## Suggestion Modes

- `light`: compact context, single-next-stroke style.
- `full`: richer context, supports draw/write/edit with multiple strokes.
- `vision`: image-assisted generation.
  - `vision_version >= 2` uses two-step mode:
    - Step 1: image analysis only.
    - Step 2: inject instruction into full-generation flow.

## Auto Maintain Overview

When Auto Maintain is enabled (`full` mode only from UI):

- Frontend syncs full stroke snapshot and optional graph snapshot image.
- Backend ingests fragments, maintains pending groups, and promotes stable blocks.
- Planner produces `action`, `targetBlockIds`, and `nextStepHint`.
- Executor selects blocks in this order:
  - `plan.targetBlockIds`
  - `active_block_ids`
  - `main_block_id`
  - recently updated blocks
- Executor expands related blocks and builds `block_outline` for FULL prompt context.

## Logging and Debugging

Runtime logs are in `lineart-board/logs/`:

- Per-request folder:
  - `input.request.json`
  - `input.messages*.json`
  - `output.cleaned.json` or `output.error.json`
- LLM dumps:
  - `llm_YYYYMMDD_HHMMSS.json`

Use these files first when investigating:

- `502 invalid payload`: check `output.error.json` and matching `llm_*.json`.
- Auto Maintain switch errors: check browser console + CORS settings.
- Prompt/context mismatch: compare `input.request.json` vs `input.messages.json`.

## Environment Variables (`lineart-board/.env`)

Required:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

Important optional:

- `CORS_ORIGINS`
- `LOG_LLM`, `LOG_IO`, `LOGS_DIR`
- `SESS_MAX_STROKES`, `SESS_KEEP_RECENT`, `SESS_RESAMPLE_STEP`, `SESS_QUANT`
- `PROMPT_SAMPLE_EVERY_N`
- Graph clustering tuning:
  - `GRAPH_VISION_STROKE_THRESHOLD`
  - `GRAPH_VISION_SPATIAL_THRESHOLD`
  - `GRAPH_VISION_AUTO_PROMOTE_CONFIDENCE`
  - `GRAPH_BLOCK_GROUP_DISTANCE_THRESHOLD`
  - `GRAPH_BLOCK_BLOCK_DISTANCE_THRESHOLD`
  - `GRAPH_BLOCK_AUTO_PROMOTE_GROUP_SIZE`

## Current Status

- Production-style flow is implemented end to end for drawing + text + auto-maintain graph.
- Most verification is log-driven and interactive; dedicated automated test suites are not yet present in this repo.
