# -*- coding: utf-8 -*-
from __future__ import annotations
import os, random, time, json, tempfile, math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas import (
    SuggestRequest,
    SuggestResponse,
    AIStrokePayload,
    Health,
    AIStrokeV11,
    StrokeStyle,
    CanvasInfo,
    SyncSessionRequest,
    SyncSessionResponse,
    CompletionRequest,
    CompletionResponse,
    GraphAutoModeRequest,
    GraphAutoModeResponse,
    GraphSnapshotResponse,
    PromoteGroupRequest,
    PromoteGroupResponse,
    PromoteVisionPendingGroupRequest,
    PromoteVisionPendingGroupResponse,
    GraphSelectionBlockActionRequest,
    GraphSelectionBlockActionResponse,
    GraphRemoveFragmentsRequest,
    GraphRemoveFragmentsResponse,
    ProjectListResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectSaveRequest,
    ProjectSaveResponse,
    ProjectCommitRequest,
    ProjectCommitResponse,
    ProjectCommitCheckoutRequest,
    ProjectCommitCheckoutResponse,
    ProjectCurrentSnapshotRequest,
    ProjectCurrentSnapshotResponse,
    ProjectDeleteRequest,
    ProjectDeleteResponse,
    ProjectCommitDeleteRequest,
    ProjectCommitDeleteResponse,
    ProjectOpenRequest,
    ProjectOpenResponse,
    ProjectDetailResponse,
)
from app import prompting
from app.llm_client import call_chat_completions
from starlette.responses import JSONResponse
import re
from app import session_store as S
from app.project_store import ProjectStore
from app.schemas import InitSessionRequest, InitSessionResponse, DeltaPayload
from fastapi import Response


# ------------------------------ Environment --------------------------------- #
# Load .env from the project root so working directory changes do not break configuration.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="LineArt LLM Gateway", version="0.2.2")

# CORS helper: register once and allow localhost/127.0.0.1 (Vite defaults).
# CORS configuration (development friendly).
# Supported modes:
#   1) CORS_ORIGINS="*"          -> allow all origins, credentials disabled.
#   2) CORS_ORIGINS empty         -> allow localhost/127.0.0.1 on any port.
#   3) CORS_ORIGINS=a,b,c         -> allow only the listed origins.
_env_cors = os.getenv("CORS_ORIGINS", "").strip()
if _env_cors == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif _env_cors:
    origins = [o.strip() for o in _env_cors.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Keep explicitly configured origins, while always allowing local dev loopback hosts.
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Default to allowing local dev origins on IPv4/IPv6 loopback.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

MOCK = os.getenv("MOCK_SUGGESTIONS", "false").lower() in ("1","true","yes")
LOG_IO = os.getenv("LOG_IO", "true").lower() in ("1","true","yes")
# Log directory resolution:
# - If LOGS_DIR is set:
#     * Absolute path -> use as-is.
#     * Relative path -> resolve from the project root (parent of app/).
# - Otherwise fall back to the system temp directory.
_root = Path(__file__).resolve().parents[1]  # Project root (one level above app/).
_logs_env = os.getenv("LOGS_DIR", "").strip()
if _logs_env:
    p = Path(_logs_env)
    _LOGS_DIR = p if p.is_absolute() else (_root / p)  # Support relative paths.
else:
    _LOGS_DIR = Path(tempfile.gettempdir()) / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------ Helpers --------------------------------- #
def _now_id() -> str:
    # 20251011-233045-123
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    return ts

def _write_json(dirpath: Path, name: str, data) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _with_llm_usage(base: Optional[Dict[str, Any]], dbg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    usage: Dict[str, Any] = dict(base or {})
    raw_usage = (dbg or {}).get("usage") if isinstance(dbg, dict) else None
    if isinstance(raw_usage, dict):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        ):
            value = raw_usage.get(key)
            if value is not None:
                usage[key] = value
    return usage


PROJECT_STORE = ProjectStore()


def _serialize_session_canvas(sess: S.Session) -> dict:
    full_strokes = []
    for s in (getattr(sess, "full_strokes", None) or []):
        if isinstance(s, dict):
            full_strokes.append(s)
    if not full_strokes:
        for s in (sess.strokes or []):
            if isinstance(s, dict):
                full_strokes.append(s)
    return {
        "schemaVersion": 1,
        "sid": sess.sid,
        "mode": sess.mode,
        "initGoal": sess.init_goal,
        "tags": list(sess.tags or []),
        "createdAtEpoch": sess.created_at,
        "callCount": int(getattr(sess, "call_count", 0) or 0),
        "strokes": full_strokes,
    }


def _serialize_session_runtime(sess: S.Session) -> dict:
    graph_runtime = getattr(sess, "graph_runtime", None)
    return {
        "schemaVersion": 1,
        "sid": sess.sid,
        "graphAuto": bool(getattr(sess, "graph_auto", False)),
        "projectId": getattr(sess, "project_id", None),
        "visionImageMode": getattr(graph_runtime, "vision_image_mode", None) if graph_runtime else None,
        "groupPromoteMode": getattr(graph_runtime, "agent_group_promote_mode", None) if graph_runtime else None,
    }


def _serialize_session_graph(sess: S.Session) -> dict:
    runtime = getattr(sess, "graph_runtime", None)
    if not getattr(sess, "graph_auto", False) or runtime is None:
        return {
            "schemaVersion": 1,
            "graphEnabled": False,
            "snapshot": {"blocks": [], "fragments": [], "groups": [], "visionPendingGroups": []},
            "graphState": None,
        }
    graph_state_dump = None
    try:
        graph_state_dump = runtime.dump_state()
    except Exception as exc:
        print("[project] graph dump failed:", exc)
    snapshot = {}
    try:
        snapshot = runtime.snapshot()
    except Exception as exc:
        print("[project] graph snapshot failed:", exc)
    return {
        "schemaVersion": 1,
        "graphEnabled": True,
        "snapshot": snapshot or {},
        "graphState": graph_state_dump,
    }


def _persist_project_for_session(
    sess: Optional[S.Session],
    *,
    reason: str,
    extra: Optional[dict] = None,
    force_project_id: Optional[str] = None,
) -> Optional[dict]:
    if not sess:
        return None
    project_id = str(force_project_id or getattr(sess, "project_id", None) or "").strip()
    if not project_id:
        return None
    canvas = _serialize_session_canvas(sess)
    graph = _serialize_session_graph(sess)
    runtime = _serialize_session_runtime(sess)
    snapshot_blocks = (((graph or {}).get("snapshot") or {}).get("blocks") or []) if isinstance(graph, dict) else []
    graph_state = (graph or {}).get("graphState") if isinstance(graph, dict) else None
    graph_state_fragments = ((((graph_state or {}).get("graphState") or {}).get("fragments") or {}) if isinstance(graph_state, dict) else {})
    try:
        meta = PROJECT_STORE.save_current(
            project_id,
            canvas=canvas,
            graph=graph,
            runtime=runtime,
            wal_op=reason,
            wal_payload=extra or {},
            meta_patch={
                "stats": {
                    "strokeCount": len(canvas.get("strokes") or []),
                    "blockCount": len(snapshot_blocks) if isinstance(snapshot_blocks, list) else 0,
                    "fragmentCount": len(graph_state_fragments) if isinstance(graph_state_fragments, dict) else 0,
                }
            },
        )
        return meta
    except Exception as exc:
        print(f"[project] save failed ({project_id}):", exc)
        return None

@app.get("/health", response_model=Health)
def health():
    print("CORS_ORIGINS =", os.getenv("CORS_ORIGINS"))
    return Health(
        status="ok",
        model=os.getenv("OPENAI_MODEL")or "unset",
        base_url=os.getenv("OPENAI_BASE_URL")or "unset",
    )

@app.options("/suggest")
def suggest_options():
    # Return 204 for OPTIONS so preflight requests succeed with proper CORS headers.
    return Response(status_code=204)

# Uniform exception handler: keep CORS headers and surface useful diagnostics.
@app.exception_handler(Exception)
async def _unhandled_except(request: Request, exc: Exception):
    # Print the stack trace so terminal logs reveal root causes (model name, auth, upstream, etc.).
    import traceback
    traceback.print_exc()
    # Preserve FastAPI's JSON error format so clients can read the detail field.
    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc.__class__.__name__}: {str(exc)}"})

def _is_finite_number(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False

def _sanitize_xy_points(raw):
    """Normalize arbitrary point lists to [[x, y], ...], dropping None/NaN/Inf entries."""
    out = []
    for p in (raw or []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
            if _is_finite_number(x) and _is_finite_number(y):
                out.append([float(x), float(y)])
    return out

# ---------------------------------------- /suggest endpoint (core) ---------------------------------------- #
@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    """
    Produce line-art suggestions.
    - MOCK=true returns a deterministic demo payload for front-end plumbing.
    - Otherwise call the model and return a v1.1 JSON payload.
    """
    # Create a per-call log directory when IO logging is enabled.
    log_dir = None
    if LOG_IO:
        log_dir = _LOGS_DIR / _now_id()
        try:
            _write_json(log_dir, "input.request.json", req.model_dump())
        except Exception:
            pass
        used_context_executor = False
    obj: Optional[Dict[str, object]] = None
    dbg: Dict[str, object] = {}
    messages: Optional[List[Dict[str, str]]] = None
    auto_complete_on = bool(getattr(req, "auto_complete_enabled", False))
    raw_hint = str(getattr(req, "hint", "") or "").strip()
    effective_hint = "" if auto_complete_on else raw_hint

# Vision 2.0 two-phase mode: pre-process before building messages.
    # Triggered when mode=vision and vision_version>=2.
    if getattr(req, "mode", None) == "vision" and float(getattr(req, "vision_version", 1.0) or 1.0) >= 2.0:
        # Extract a viewport tuple.
        def _extract_viewport(_req: SuggestRequest):
            # Prefer context.canvas.viewport when available.
            try:
                if isinstance(getattr(_req, "context", None), dict):
                    vp = (_req.context.get("canvas") or {}).get("viewport")
                    if vp: return tuple(vp)
            except Exception:
                pass
            try:
                vp = getattr(getattr(_req, "context", None), "canvas", None)
                if vp and getattr(vp, "viewport", None):
                    return tuple(vp.viewport)
            except Exception:
                pass
            return (0, 0, 1024, 768)
        
        #########  
        if int(getattr(req, "seq", 1) or 1) == 1:
            try:
                _vp = _extract_viewport(req)
                req_d = req.model_dump()
                # Provide the viewport at the top level for prompting to read.
                req_d["canvas"] = {"viewport": list(_vp)}
                msgs = prompting.build_vision_v2_step1(req_d)
                if LOG_IO:
                    try: _write_json(log_dir, "input.messages.step1.json", msgs)
                    except Exception: pass
                parsed, dbg = call_chat_completions(msgs, max_tokens=8000)
                analysis = ""
                instruction = ""
                if isinstance(parsed, dict):
                    analysis = str(parsed.get("analysis","") or "").strip()
                    instruction = str(parsed.get("instruction","") or "").strip()
                if not instruction:
                    instruction = (effective_hint or "Make the single best next stroke.")
                if LOG_IO:
                    try: _write_json(log_dir, "output.step1.json", {"analysis":analysis, "instruction":instruction, "raw_text": (dbg or {}).get("raw_text")})
                    except Exception: pass
                # Step-1 emits no strokes; it only returns vision metadata for Step-2.
                return SuggestResponse(
                    payload=AIStrokePayload(version=2, intent="hint", strokes=[]),
                    usage=_with_llm_usage({"mode":"vision-2.0-step1","raw_text": (dbg or {}).get("raw_text")}, dbg),
                    vision2={"analysis": analysis, "instruction": instruction}
                )
            except Exception as e:
                raise HTTPException(502, f"vision 2.0 step1 failed: {e}")
            
        #########
        # Step-2: consume instruction_text and generate concrete strokes.
        else:
            if not getattr(req, "instruction_text", None):
                raise HTTPException(400, "instruction_text required for vision 2.0 step2")
            try:
                # Ensure the top-level canvas.viewport matches Step-1 naming.
                _vp = ( (getattr(getattr(req, "context", None), "canvas", None) or {}).get("viewport", None)
                        if isinstance(getattr(req, "context", None), dict) else None )
                if _vp is None:
                    try:
                        _vp = tuple(getattr(getattr(req, "context", None), "canvas").viewport)
                    except Exception:
                        _vp = (0, 0, 1024, 768)

                # Parse instruction_text (JSON or plain text) returned from Step-1.
                a_text, i_text = "", ""
                _ins_raw = str(getattr(req, "instruction_text") or "").strip()
                try:
                    _maybe = json.loads(_ins_raw)
                    if isinstance(_maybe, dict):
                        a_text = str(_maybe.get("analysis","") or "").strip()
                        i_text = str(_maybe.get("instruction","") or "").strip()
                    else:
                        i_text = _ins_raw
                except Exception:
                    i_text = _ins_raw

                orig_hint = effective_hint
                _parts = []
                if orig_hint: _parts.append(orig_hint)
                if a_text:    _parts.append("Vision Analysis:\n" + a_text)
                if i_text:    _parts.append("Instruction:\n" + i_text)
                enhanced_hint = "\n\n".join(_parts) if _parts else "Make the single best next stroke."

                # Build the prompting request payload using the Step-1 data shape.
                req_d = req.model_dump()
                req_d["canvas"] = {"viewport": list(_vp)}  # Provide viewport for prompting.
                req_d["hint"] = enhanced_hint              # Override hint, reuse the rest of the full logic.
                ctx = req_d.get("context")
                if isinstance(ctx, dict) and isinstance(ctx.get("strokes"), list):
                    for stk in ctx["strokes"]:
                        stk["points"] = _sanitize_xy_points(stk.get("points"))

                # Reuse the full-mode message builder to stay aligned with existing flows.
                from app.schemas import SuggestRequest as _SReq
                _req2_obj = _SReq.model_validate(req_d)
                msgs = prompting.build_messages(_req2_obj, include_sample=True)
                if LOG_IO:
                    try: _write_json(log_dir, "input.messages.step2.json", msgs)
                    except Exception: pass
                obj, dbg = call_chat_completions(messages=msgs)
                # Return data that matches downstream cleanup expectations.
                if not (isinstance(obj, dict) and "version" in obj and "strokes" in obj):
                    raise ValueError("model did not return v1.1 JSON")
                payload = AIStrokePayload(**obj)
                usage = _with_llm_usage({"mode":"vision-2.0-step2", "raw_text": (dbg or {}).get("raw_text")}, dbg)
                return SuggestResponse(payload=payload, usage=usage)
            except Exception as e:
                raise HTTPException(502, f"vision 2.0 step2 failed: {e}")

    # ---- Choose context source: legacy context vs. sid+delta ----
    # Handle vision-mode image inputs.
    if getattr(req, "mode", None) == "vision" and req.image_data and not (float(getattr(req, "vision_version", 1.0) or 1.0) >= 2.0 and int(getattr(req, "seq", 1) or 1) == 2):
        try:
            import base64
            img_data = req.image_data
            # Accept either data URLs or raw base64.
            if img_data.startswith("data:"):
                header, b64 = img_data.split(",", 1)
                mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
            else:
                b64 = img_data
                mime = req.image_mime or "image/png"

            ext = ".jpg" if "jpeg" in mime else ".png"
            img_path = log_dir / f"input.image{ext}"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception as e:
            print(f"[warn] failed to save vision image: {e}")

    # ---- Choose context source: legacy context vs. sid+delta ----
    messages = None
    new_sid: str | None = None
    if req.sid:
        sess = S.get_session(req.sid)
        if not sess:
            # Re-initialize the session automatically if it was lost.
            sess = S.create_session(mode="light_helper", init_goal=effective_hint or None, tags=None)
            new_sid = sess.sid
        # Merge incremental strokes when provided.
        if req.delta and isinstance(req.delta, DeltaPayload):
            S.Session.append_strokes(sess, [s.model_dump() for s in req.delta.strokes])  # type: ignore
        # Replace session state with the provided snapshot to keep deletes/erasures consistent.
        if req.context and isinstance(req.context.strokes, list):
            sess.replace_strokes([s.model_dump() for s in req.context.strokes])  # type: ignore

        graph_runtime = sess.graph_runtime
        runtime_group_promote_mode = getattr(req, "group_promote_mode", None)
        if graph_runtime is not None and runtime_group_promote_mode:
            try:
                graph_runtime.set_group_promotion_mode(runtime_group_promote_mode)
            except Exception as exc:
                print("[graph] set group promote mode failed:", exc)
        runtime_vision_image_mode = getattr(req, "vision_image_mode", None)
        if graph_runtime is not None and runtime_vision_image_mode:
            try:
                graph_runtime.set_vision_image_mode(runtime_vision_image_mode)
            except Exception as exc:
                print("[graph] set vision image mode failed:", exc)
        req_mode = (getattr(req, "mode", None) or "full").lower()
        use_context_executor = bool(sess.graph_auto and graph_runtime is not None and req_mode in {"full", "light"})

        cc = sess.bump()
        include_sample = S.should_include_sample(cc)

        if use_context_executor and graph_runtime is not None:
            if req.context and isinstance(req.context.strokes, list):
                try:
                    graph_runtime.sync_strokes_snapshot([s.model_dump() for s in req.context.strokes])
                except Exception as exc:
                    print("[graph] ingest full context failed:", exc)
            if req.delta and isinstance(req.delta, DeltaPayload):
                try:
                    graph_runtime.ingest_strokes([s.model_dump() for s in req.delta.strokes])
                except Exception as exc:
                    print("[graph] ingest delta failed:", exc)

            plan_bundle = graph_runtime.run_conversation(
                user_input=effective_hint,
                mode=req.mode,
                prefer_explanatory_drawing=getattr(req, "prefer_explanatory_drawing", None),
            )
            obj = plan_bundle.get("payload") or {}
            dbg = {"mode": "context-executor", "plan": plan_bundle.get("plan"), "usage": plan_bundle.get("usage")}
            used_context_executor = True
            try:
                _persist_project_for_session(
                    sess,
                    reason="suggest_context_executor",
                    extra={
                        "mode": str(req.mode or ""),
                        "hasPlan": bool(plan_bundle.get("plan")),
                    },
                )
            except Exception:
                pass
            if LOG_IO and log_dir is not None:
                try:
                    _write_json(log_dir, "output.context.plan.json", plan_bundle)
                except Exception:
                    pass
        else:
            recent = sess.recent_for_model()

            def _r3(v: float) -> float:
                try:
                    return round(float(v), 3)
                except Exception:
                    return float(v)

            lite_ctx = AIStrokePayload(
                version=1,
                intent="complete",
                strokes=[
                    AIStrokeV11(
                        id=s["id"],
                        tool=s.get("tool", "pen"),
                        points=[[ _r3(p[0]), _r3(p[1]) ] for p in s.get("points", [])],
                        style=StrokeStyle(
                            size=((s.get("style") or {}).get("size") or "m"),
                            color=((s.get("style") or {}).get("color") or "black"),
                            opacity=float((s.get("style") or {}).get("opacity") or 1.0),
                        ),
                        meta=s.get("meta") or {},
                    )
                    for s in recent
                ],
            )
            fake = SuggestRequest(
                context=lite_ctx,
                hint=effective_hint,
                model=req.model,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                gen_scale=req.gen_scale,
                prefer_explanatory_drawing=req.prefer_explanatory_drawing,
            )
            messages = prompting.build_messages_by_mode(fake, getattr(req, "mode", None))
            if LOG_IO:
                try:
                    _write_json(log_dir, "input.messages.json", messages)
                except Exception:
                    pass
    else:
        # Remain compatible with legacy full-context payloads.
        if not req.context:
            raise HTTPException(400, "Either {sid, delta} or {context} must be provided.")
        req_d = req.model_dump()
        req_d["hint"] = effective_hint
        req_no_hint = SuggestRequest.model_validate(req_d)
        messages = prompting.build_messages_by_mode(req_no_hint, getattr(req, "mode", None))
        if LOG_IO:
            try: _write_json(log_dir, "input.messages.json", messages)
            except Exception: pass

    # MOCK shortcut: serve a deterministic payload when no key or demo mode.
    if MOCK and not used_context_executor:
        pid = f"ai_{int(time.time())}_{random.randint(1000,9999)}"
        payload = AIStrokePayload(
            version=1,
            intent="complete",
            canvas=req.context.canvas or CanvasInfo(viewport=(0,0,1024,768)),
            strokes=[
                AIStrokeV11(
                    id=f"{pid}_curve",
                    tool="pen",
                    points=[[120,220],[180,260],[260,250],[320,280]],
                    style=StrokeStyle(size="m", color="light-violet", opacity=0.85),
                    meta={"source":"mock","desc":"soft curve"}
                ),
                AIStrokeV11(
                    id=f"{pid}_rect",
                    tool="rect",
                    points=[[420,200],[580,320]],
                    style=StrokeStyle(size="l", color="orange", opacity=0.7),
                    meta={"source":"mock","desc":"rect block"}
                ),
            ],
        )
        if LOG_IO:
          try: _write_json(log_dir, "output.ok.json", payload.model_dump())
          except Exception: pass
        usage = {"stage":"ok"}
        if new_sid:
            usage["new_sid"] = new_sid
        return SuggestResponse(ok=True, payload=payload, usage=usage)
    if not used_context_executor:
        obj, dbg = call_chat_completions(
            messages=messages,
            model=req.model,
            temperature=req.temperature or 0.4,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 10240,
        )
    
    # Normalizer: turn raw LLM strokes into clean renderable data.
    def _clamp01(v: float) -> float:
        try: return max(0.0, min(1.0, float(v)))
        except Exception: return 1.0

    def _r3(v: float) -> float:
        try: return round(float(v), 3)
        except Exception: return float(v)

    _ALLOWED_COLORS = {
        "black", "blue", "green", "grey",
        "light-blue", "light-green", "light-red", "light-violet",
        "orange", "red", "violet", "white", "yellow",
    }
    _COLOR_ALIASES = {
        "gray": "grey",
        "light-gray": "grey",
        "light-grey": "grey",
        "lightgray": "grey",
        "lightgrey": "grey",
        "purple": "violet",
        "light-purple": "light-violet",
        "lightpurple": "light-violet",
        "lightviolet": "light-violet",
        "navy": "blue",
        "skyblue": "light-blue",
        "lightblue": "light-blue",
        "lightgreen": "light-green",
        "lightred": "light-red",
        "brown": "orange",
        "light-brown": "orange",
        "lightbrown": "orange",
        "dark-red": "red",
        "darkred": "red",
    }

    def _normalize_color(value: object) -> str:
        """
        Normalize color names to the allowed palette.
        Unknown colors fallback to black to avoid payload validation errors.
        """
        if value is None:
            return "black"
        token = str(value).strip().lower()
        if not token:
            return "black"
        token = token.replace("_", "-").replace(" ", "-")
        canonical = _COLOR_ALIASES.get(token, token)
        return canonical if canonical in _ALLOWED_COLORS else "black"

    def _limit_points(pts, max_n: int):
        # Resample points evenly up to max_n while keeping endpoints.
        if not isinstance(pts, list) or len(pts) <= max_n: return pts
        # Use only x/y components to avoid mismatched t/pressure entries.
        xy = [(float(p[0]), float(p[1])) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(xy) <= max_n: 
            return [[x,y] for x,y in xy]
        # Compute approximate path length.
        segs = []; total = 0.0
        for i in range(len(xy)-1):
            d = ((xy[i+1][0]-xy[i][0])**2 + (xy[i+1][1]-xy[i][1])**2) ** 0.5
            segs.append(d); total += d
        if total <= 1e-9: 
            return [[xy[0][0], xy[0][1]], [xy[-1][0], xy[-1][1]]]
        out = [xy[0]]
        steps = max_n - 1
        for s in range(1, steps):
            target = total * s / steps
            acc = 0.0
            j = 0
            while j < len(segs) and acc + segs[j] < target:
                acc += segs[j]; j += 1
            if j >= len(segs): 
                out.append(xy[-1]); break
            t = (target - acc) / (segs[j] or 1e-9)
            x = xy[j][0] + t * (xy[j+1][0] - xy[j][0])
            y = xy[j][1] + t * (xy[j+1][1] - xy[j][1])
            out.append((x, y))
        out.append(xy[-1])
        return [[x,y] for x,y in out]

    def _max_deviation(pts) -> float:
        if not isinstance(pts, list) or len(pts) <= 2: return 0.0
        x1,y1 = float(pts[0][0]), float(pts[0][1])
        x2,y2 = float(pts[-1][0]), float(pts[-1][1])
        Cx, Cy = (x2-x1), (y2-y1)
        L2 = Cx*Cx + Cy*Cy
        if L2 <= 1e-9: return 0.0
        import math
        maxd = 0.0
        for k in range(1, len(pts)-1):
            x,y = float(pts[k][0]), float(pts[k][1])
            vx, vy = (x-x1), (y-y1)
            cross = abs(vx*Cy - vy*Cx)
            d = cross / math.sqrt(L2)
            if d > maxd: maxd = d
        return maxd

    class _CleanRuleError(Exception):
        def __init__(self, rule: str, detail: str):
            super().__init__(detail)
            self.rule = str(rule or "UNKNOWN_RULE")
            self.detail = str(detail or "")

    def _fail_rule(rule: str, detail: str):
        raise _CleanRuleError(rule, detail)

    def _summarize_clean_errors(errors: list[dict], max_rules: int = 8, max_samples: int = 3) -> str:
        if not errors:
            return "none"
        counts: Dict[str, int] = {}
        for item in errors:
            rule = str(item.get("rule") or "UNKNOWN_RULE")
            counts[rule] = counts.get(rule, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_rules]
        counts_part = ",".join(f"{k}:{v}" for k, v in ordered)
        sample_bits = []
        for item in errors[:max_samples]:
            sample_bits.append(
                f"idx={item.get('index')},id={item.get('stroke_id')},tool={item.get('tool')},rule={item.get('rule')}"
            )
        if sample_bits:
            return f"{counts_part}; samples={' | '.join(sample_bits)}"
        return counts_part

    def _clean_one(s: dict, gen_scale: int) -> dict:
        tool_in = str(s.get("tool") or "pen").lower()

        # 1) Normalize points: keep up to the first four entries.
        raw_pts = s.get("points") or []
        pts = [
            [float(p[0]), float(p[1])]
            + ([p[2]] if len(p) > 2 else [])
            + ([p[3]] if len(p) > 3 else [])
            for p in raw_pts
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        # edit tool may omit points (text-targeted edit); all others still need >=2 points.
        if tool_in != "edit" and len(pts) < 2:
            _fail_rule("MIN_POINTS", "LLM stroke has <2 points.")

        # 2) Tool-specific cleanup.
        if tool_in == "poly":
            if len(pts) < 3:
                _fail_rule("POLY_MIN_POINTS", "LLM poly needs >= 3 points.")
            # Append the start point when a path should be closed.
            if not (abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6):
                pts.append([pts[0][0], pts[0][1]] + (pts[0][2:] if len(pts[0]) > 2 else []))
            limit = max(4, min(128, int(gen_scale) if gen_scale else 24))
            core = pts[:-1]
            core = _limit_points(core, max(3, limit - 1))
            pts2 = core + [core[0][:2]]
            tool = "poly"

        elif tool_in == "ellipse":
            if len(pts) < 2:
                _fail_rule("ELLIPSE_MIN_POINTS", "LLM ellipse needs 2 points.")
            x0 = min(pts[0][0], pts[-1][0]); y0 = min(pts[0][1], pts[-1][1])
            x1 = max(pts[0][0], pts[-1][0]); y1 = max(pts[0][1], pts[-1][1])
            pts2 = [[x0, y0], [x1, y1]]
            tool = "ellipse"

        elif tool_in == "text":
            # expect at least two points [[x,y],[x2,y2]]
            raw_pts = s.get("points") or []
            if not (isinstance(raw_pts, list) and len(raw_pts) >= 2):
                _fail_rule("TEXT_BOX_MIN_POINTS", "LLM text box needs 2 points.")
            p0 = raw_pts[0]; p1 = raw_pts[1]
            try:
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
            except Exception:
                _fail_rule("TEXT_BOX_INVALID_POINTS", "invalid text box points")

            tx0 = min(x0, x1); ty0 = min(y0, y1)
            tx1 = max(x0, x1); ty1 = max(y0, y1)
            pts2 = [[tx0, ty0], [tx1, ty1]]

            style = s.get("style") or {}
            size = style.get("size") or "m"
            color = _normalize_color(style.get("color"))
            opacity = _clamp01(style.get("opacity", 1.0))

            return {
                "id": str(s.get("id") or f"ai_{int(time.time())}"),
                "tool": "text",
                "points": [[_r3(p[0]), _r3(p[1])] for p in pts2],
                "style": {"size": size, "color": color, "opacity": opacity},
                "meta": s.get("meta") or {},
            }
        elif tool_in == "edit":
            meta = s.get("meta") or {}
            target_id = meta.get("targetId") or meta.get("target_id") or meta.get("target")
            if not target_id:
                _fail_rule("EDIT_MISSING_TARGET_ID", "LLM edit stroke missing targetId.")
            operation = str(meta.get("operation") or "").strip()
            text = str(meta.get("text") or "").strip()
            if not operation:
                _fail_rule("EDIT_MISSING_OPERATION", "LLM edit stroke missing operation description.")
            if not text:
                _fail_rule("EDIT_MISSING_TEXT", "LLM edit stroke missing text content.")
            raw_pts = s.get("points") or []
            pts_edit = []
            if isinstance(raw_pts, list) and len(raw_pts) >= 2:
                try:
                    x0, y0 = float(raw_pts[0][0]), float(raw_pts[0][1])
                    x1, y1 = float(raw_pts[1][0]), float(raw_pts[1][1])
                    tx0 = min(x0, x1); ty0 = min(y0, y1)
                    tx1 = max(x0, x1); ty1 = max(y0, y1)
                    pts_edit = [[_r3(tx0), _r3(ty0)], [_r3(tx1), _r3(ty1)]]
                except Exception:
                    pts_edit = []
            meta_clean = dict(meta)
            meta_clean["targetId"] = str(target_id)
            meta_clean["operation"] = operation
            meta_clean["text"] = text
            meta_clean.pop("content", None)
            style = s.get("style") or {}
            size = style.get("size") or "m"
            color = _normalize_color(style.get("color"))
            opacity = _clamp01(style.get("opacity", 1.0))
            return {
                "id": str(s.get("id") or f"ai_{int(time.time())}"),
                "tool": "edit",
                "points": pts_edit,
                "style": {"size": size, "color": color, "opacity": opacity},
                "meta": meta_clean,
            }
        
        else:
            # Treat near-coincident endpoints as closed.
            def _pts_equal(a, b, eps=1.5):
                return abs(a[0]-b[0]) <= eps and abs(a[1]-b[1]) <= eps
            is_closed = (len(pts) >= 3) and _pts_equal(pts[0], pts[-1])
            if is_closed:
                limit = max(4, min(128, int(gen_scale) if gen_scale else 24))
                pts2 = _limit_points(pts, limit)
                tool = "pen"  # Keep parity with existing strategy: closed non-poly routes stay curves.
            else:
                # For open paths, choose line vs curve based on deviation.
                if _max_deviation(pts) < 0.8 and not _pts_equal(pts[0], pts[-1], eps=1e-6):
                    tool = "line"
                    pts2 = [pts[0][:2], pts[-1][:2]]
                else:
                    tool = "line" if tool_in == "line" else "pen"
                    limit = max(4, min(64, int(gen_scale) if gen_scale else 24))
                    pts2 = _limit_points(pts, limit)

        # 3) Fill in style and metadata defaults.
        style = s.get("style") or {}
        size = style.get("size") or "m"
        color = _normalize_color(style.get("color"))
        opacity = _clamp01(style.get("opacity", 1.0))

        return {
            "id": str(s.get("id") or f"ai_{int(time.time())}"),
            "tool": tool,
            "points": [[_r3(p[0]), _r3(p[1])] for p in pts2],
            "style": {"size": size, "color": color, "opacity": opacity},
            "meta": s.get("meta") or {},
        }

    def _clean_payload(obj, gen_scale: int):
        intent = (obj.get("intent") or "complete") if isinstance(obj, dict) else "complete"
        intent_norm = str(intent).strip().lower()
        canvas = obj.get("canvas") if isinstance(obj, dict) else None
        replace = obj.get("replace") if isinstance(obj, dict) else None

        # Keep and sanitize every stroke.
        strokes_in = (obj.get("strokes") or []) if isinstance(obj, dict) else []
        if not isinstance(strokes_in, list):
            raise HTTPException(502, "LLM returned invalid strokes field.")
        if len(strokes_in) == 0:
            if intent_norm == "noop":
                cleaned = {
                    "version": 1,
                    "intent": "noop",
                    "strokes": [],
                }
                if isinstance(canvas, dict):
                    cleaned["canvas"] = canvas
                if isinstance(replace, list):
                    cleaned["replace"] = [str(x) for x in replace]
                return cleaned, []
            raise HTTPException(502, "LLM returned empty strokes.")
        cleaned_list = []
        clean_errors = []
        for idx, s in enumerate(strokes_in):
            stroke_id = str((s or {}).get("id") or "") if isinstance(s, dict) else ""
            stroke_tool = str((s or {}).get("tool") or "") if isinstance(s, dict) else ""
            try:
                cleaned_list.append(_clean_one(s, gen_scale))
            except _CleanRuleError as e:
                clean_errors.append(
                    {
                        "index": idx,
                        "stroke_id": stroke_id,
                        "tool": stroke_tool,
                        "rule": e.rule,
                        "detail": e.detail,
                    }
                )
                print(
                    f"[clean] drop one stroke: rule={e.rule} detail={e.detail} "
                    f"index={idx} id={stroke_id} tool={stroke_tool}"
                )
            except HTTPException as e:
                clean_errors.append(
                    {
                        "index": idx,
                        "stroke_id": stroke_id,
                        "tool": stroke_tool,
                        "rule": "HTTP_EXCEPTION",
                        "detail": str(getattr(e, "detail", e)),
                    }
                )
                print(
                    f"[clean] drop one stroke: rule=HTTP_EXCEPTION detail={getattr(e, 'detail', e)} "
                    f"index={idx} id={stroke_id} tool={stroke_tool}"
                )
            except Exception as e:
                clean_errors.append(
                    {
                        "index": idx,
                        "stroke_id": stroke_id,
                        "tool": stroke_tool,
                        "rule": "UNEXPECTED_CLEAN_EXCEPTION",
                        "detail": str(e),
                    }
                )
                # Skip invalid strokes but preserve at least one valid entry.
                print(
                    f"[clean] drop one stroke: rule=UNEXPECTED_CLEAN_EXCEPTION detail={e} "
                    f"index={idx} id={stroke_id} tool={stroke_tool}"
                )
        if not cleaned_list:
            summary = _summarize_clean_errors(clean_errors)
            raise HTTPException(502, f"All strokes invalid after cleaning. failed_rules={summary}")

        # Assemble the final payload.
        cleaned = {
            "version": 1,
            "intent": intent,
            "strokes": cleaned_list,
        }
        if isinstance(canvas, dict):
            cleaned["canvas"] = canvas
        if isinstance(replace, list):
            cleaned["replace"] = [str(x) for x in replace]

        return cleaned, clean_errors

    obj_clean, clean_errors = _clean_payload(obj, req.gen_scale or 24)

    # pydantic validation as a safety net.
    try:
        payload = AIStrokePayload.model_validate(obj_clean)
        if LOG_IO and log_dir is not None:
            try: _write_json(log_dir, "output.cleaned.json", payload.model_dump())
            except Exception: pass
        usage = _with_llm_usage({
            "stage": "ok",
            "raw_text": dbg.get("raw_text"),
            "mode": dbg.get("mode"),
            "model": dbg.get("model"),
            "response_id": (dbg.get("response_dump") or {}).get("id"),
        }, dbg)
        plan_info = dbg.get("plan")
        if isinstance(plan_info, dict):
            planner_next_step = str(plan_info.get("nextStepHint") or "").strip()
            if planner_next_step:
                usage["planner_next_step"] = planner_next_step
            target_block_ids = plan_info.get("targetBlockIds")
            if isinstance(target_block_ids, list):
                usage["plan_target_block_ids"] = [str(v) for v in target_block_ids if str(v or "").strip()]
            active_block_ids = plan_info.get("activeBlockIds")
            if isinstance(active_block_ids, list):
                usage["active_block_ids"] = [str(v) for v in active_block_ids if str(v or "").strip()]
            main_block_id = str(plan_info.get("mainBlockId") or "").strip()
            if main_block_id:
                usage["main_block_id"] = main_block_id
            plan_action = str(plan_info.get("action") or "").strip()
            if plan_action:
                usage["plan_action"] = plan_action
        if clean_errors:
            usage["clean_dropped_strokes"] = len(clean_errors)
            usage["clean_failed_rules"] = _summarize_clean_errors(clean_errors)
        if new_sid: usage["new_sid"] = new_sid
        return SuggestResponse(ok=True, payload=payload, usage=usage)
    
    except Exception as e:
        clean_rules = _summarize_clean_errors(clean_errors) if clean_errors else "none"
        if LOG_IO and log_dir is not None:
            try:
                _write_json(
                    log_dir,
                    "output.error.json",
                    {
                        "error": "invalid payload",
                        "detail": str(e),
                        "raw": obj,
                        "raw_text": dbg.get("raw_text"),
                        "clean_errors": clean_errors,
                        "clean_failed_rules": clean_rules,
                    },
                )
            except Exception:
                pass
        raise HTTPException(
            502,
            f"LLM returned invalid payload after cleaning: {e} | clean_failed_rules={clean_rules} | raw={dbg.get('raw_text')!r}",
        )
    

# ---------------------------------------- Session management endpoints ---------------------------------------- #
# Session initialization endpoint.
@app.post("/session/init", response_model=InitSessionResponse)
def session_init(body: InitSessionRequest):
    s = S.create_session(mode=body.mode, init_goal=body.init_goal, tags=body.tags)
    return InitSessionResponse(sid=s.sid, note="ok")


@app.get("/projects", response_model=ProjectListResponse)
def list_projects():
    items = PROJECT_STORE.list_projects()
    return ProjectListResponse(ok=True, projects=items)


@app.post("/project/create", response_model=ProjectCreateResponse)
def project_create(body: ProjectCreateRequest):
    sess = S.get_session(body.sid) if body.sid else None
    if body.sid and not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    try:
        meta = PROJECT_STORE.create_project(name=body.name)
    except Exception as exc:
        raise HTTPException(400, f"project create failed: {exc}")
    project_id = str(meta.get("projectId") or "")
    if sess:
        sess.project_id = project_id
        _persist_project_for_session(sess, reason="project_bind_create", force_project_id=project_id)
    return ProjectCreateResponse(ok=True, projectId=project_id, sid=(sess.sid if sess else body.sid), meta=meta)


@app.post("/project/save", response_model=ProjectSaveResponse)
def project_save(body: ProjectSaveRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    project_id = str(body.project_id or sess.project_id or "").strip()
    if not project_id:
        raise HTTPException(400, "projectId required (session is not bound to a project)")
    sess.project_id = project_id
    meta = _persist_project_for_session(
        sess,
        reason="manual_project_save",
        extra={"note": str(body.note or "")[:200]},
        force_project_id=project_id,
    )
    if meta is None:
        raise HTTPException(500, "project save failed")
    if body.snapshot:
        try:
            PROJECT_STORE.save_current_preview(
                project_id,
                snapshot=body.snapshot.model_dump(),
                note=body.note,
            )
        except Exception as exc:
            raise HTTPException(500, f"project current preview save failed: {exc}")
    return ProjectSaveResponse(ok=True, projectId=project_id, saved=True, meta=meta)


@app.post("/project/current-snapshot", response_model=ProjectCurrentSnapshotResponse)
def project_current_snapshot(body: ProjectCurrentSnapshotRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    project_id = str(body.project_id or sess.project_id or "").strip()
    if not project_id:
        raise HTTPException(400, "projectId required (session is not bound to a project)")
    sess.project_id = project_id
    try:
        preview_meta = PROJECT_STORE.save_current_preview(
            project_id,
            snapshot=body.snapshot.model_dump(),
            note="auto-current-preview",
        )
    except Exception as exc:
        raise HTTPException(500, f"project current snapshot save failed: {exc}")
    return ProjectCurrentSnapshotResponse(
        ok=True,
        projectId=project_id,
        currentPreviewUpdatedAt=str(preview_meta.get("updatedAt") or ""),
    )


@app.post("/project/commit", response_model=ProjectCommitResponse)
def project_commit(body: ProjectCommitRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    project_id = str(body.project_id or sess.project_id or "").strip()
    if not project_id:
        raise HTTPException(400, "projectId required (session is not bound to a project)")
    sess.project_id = project_id
    # Sync current working state first so current/* remains authoritative.
    meta = _persist_project_for_session(
        sess,
        reason="manual_project_commit",
        extra={"message": str(body.message or "")[:500]},
        force_project_id=project_id,
    )
    if meta is None:
        raise HTTPException(500, "project current save failed before commit")
    canvas = _serialize_session_canvas(sess)
    graph = _serialize_session_graph(sess)
    runtime = _serialize_session_runtime(sess)
    try:
        commit_meta = PROJECT_STORE.create_commit(
            project_id,
            canvas=canvas,
            graph=graph,
            runtime=runtime,
            snapshot=body.snapshot.model_dump(),
            message=body.message,
        )
        PROJECT_STORE.save_current_preview(
            project_id,
            snapshot=body.snapshot.model_dump(),
            note=(body.message or "commit"),
        )
        meta_doc = (PROJECT_STORE.load_project(project_id).get("meta") or {})
    except Exception as exc:
        raise HTTPException(500, f"project commit failed: {exc}")
    return ProjectCommitResponse(
        ok=True,
        projectId=project_id,
        commitId=str(commit_meta.get("commitId") or ""),
        meta=meta_doc if isinstance(meta_doc, dict) else (meta or {}),
        commitSummary=commit_meta,
    )


@app.post("/project/commit/checkout", response_model=ProjectCommitCheckoutResponse)
def project_commit_checkout(body: ProjectCommitCheckoutRequest):
    pid = str(body.project_id or "").strip()
    cid = str(body.commit_id or "").strip()
    if not pid or not cid:
        raise HTTPException(400, "projectId and commitId required")
    try:
        PROJECT_STORE.checkout_commit_to_current(pid, cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"commit checkout failed: {exc}")
    opened = project_open(ProjectOpenRequest(projectId=pid, sid=body.sid))
    return ProjectCommitCheckoutResponse(
        ok=True,
        projectId=pid,
        sid=opened.sid,
        graphEnabled=opened.graph_enabled,
        strokeCount=opened.stroke_count,
        graphSummary=opened.graph_summary,
        strokes=opened.strokes,
        checkedOutCommitId=cid,
    )


@app.post("/project/delete", response_model=ProjectDeleteResponse)
def project_delete(body: ProjectDeleteRequest):
    pid = str(body.project_id or "").strip()
    if not pid:
        raise HTTPException(400, "projectId required")
    unbound = False
    if body.sid:
        sess = S.get_session(body.sid)
        if sess and str(getattr(sess, "project_id", "") or "").strip() == pid:
            sess.project_id = None
            unbound = True
    try:
        PROJECT_STORE.delete_project(pid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"project delete failed: {exc}")
    return ProjectDeleteResponse(ok=True, projectId=pid, unboundSession=unbound)


@app.post("/project/commit/delete", response_model=ProjectCommitDeleteResponse)
def project_commit_delete(body: ProjectCommitDeleteRequest):
    pid = str(body.project_id or "").strip()
    cid = str(body.commit_id or "").strip()
    if not pid or not cid:
        raise HTTPException(400, "projectId and commitId required")
    try:
        result = PROJECT_STORE.delete_commit(pid, cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"commit delete failed: {exc}")
    return ProjectCommitDeleteResponse(
        ok=True,
        projectId=pid,
        commitId=cid,
        currentRefCleared=bool((result or {}).get("currentRefCleared")),
    )


@app.post("/project/open", response_model=ProjectOpenResponse)
def project_open(body: ProjectOpenRequest):
    project_id = str(body.project_id or "").strip()
    if not project_id:
        raise HTTPException(400, "projectId required")
    try:
        bundle = PROJECT_STORE.load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(404, f"project not found: {project_id}")
    except Exception as exc:
        raise HTTPException(500, f"project load failed: {exc}")

    canvas = bundle.get("canvas") if isinstance(bundle.get("canvas"), dict) else {}
    graph_doc = bundle.get("graph") if isinstance(bundle.get("graph"), dict) else {}
    runtime_doc = bundle.get("runtime") if isinstance(bundle.get("runtime"), dict) else {}

    sess = S.get_session(body.sid) if body.sid else None
    if sess is None:
        sess = S.create_session(
            mode=str(canvas.get("mode") or "light_helper"),
            init_goal=(str(canvas.get("initGoal")) if canvas.get("initGoal") is not None else None),
            tags=[str(v) for v in (canvas.get("tags") or []) if str(v or "").strip()] if isinstance(canvas.get("tags"), list) else None,
        )
    else:
        sess.mode = str(canvas.get("mode") or sess.mode or "light_helper")
        sess.init_goal = (str(canvas.get("initGoal")) if canvas.get("initGoal") is not None else sess.init_goal)
        if isinstance(canvas.get("tags"), list):
            sess.tags = [str(v) for v in (canvas.get("tags") or []) if str(v or "").strip()]

    try:
        if canvas.get("createdAtEpoch") is not None:
            sess.created_at = float(canvas.get("createdAtEpoch"))
    except Exception:
        pass
    try:
        if canvas.get("callCount") is not None:
            sess.call_count = int(canvas.get("callCount"))
    except Exception:
        pass

    strokes_payload = canvas.get("strokes") or []
    if not isinstance(strokes_payload, list):
        strokes_payload = []
    try:
        sess.replace_strokes([s for s in strokes_payload if isinstance(s, dict)])
    except Exception as exc:
        raise HTTPException(400, f"project canvas restore failed: {exc}")

    graph_enabled = bool(graph_doc.get("graphEnabled"))
    graph_state_dump = graph_doc.get("graphState") if isinstance(graph_doc.get("graphState"), dict) else None
    if graph_enabled and graph_state_dump:
        runtime_canvas_size = None
        runtime_dump = graph_state_dump.get("runtime") if isinstance(graph_state_dump, dict) else None
        if isinstance(runtime_dump, dict):
            cs = runtime_dump.get("canvasSize")
            if isinstance(cs, (list, tuple)) and len(cs) >= 2:
                try:
                    runtime_canvas_size = (float(cs[0]), float(cs[1]))
                except Exception:
                    runtime_canvas_size = None
        runtime = sess.init_graph_runtime(canvas_size=runtime_canvas_size)
        try:
            runtime.load_state(graph_state_dump)
        except Exception as exc:
            sess.disable_graph_runtime()
            raise HTTPException(500, f"project graph restore failed: {exc}")
    else:
        sess.disable_graph_runtime()

    # Restore runtime toggles if present (graph runtime already handles detailed mode state when enabled).
    if isinstance(runtime_doc, dict) and runtime_doc.get("graphAuto") is False:
        sess.disable_graph_runtime()

    sess.project_id = project_id

    graph_summary = {"blocks": 0, "groups": 0, "fragments": 0, "visionPendingGroups": 0}
    if sess.graph_runtime:
        try:
            snap = sess.graph_runtime.snapshot()
            graph_summary = {
                "blocks": len(snap.get("blocks") or []),
                "groups": len(snap.get("groups") or []),
                "fragments": len(snap.get("fragments") or []),
                "visionPendingGroups": len(snap.get("visionPendingGroups") or []),
            }
        except Exception:
            pass

    _persist_project_for_session(
        sess,
        reason="project_open",
        extra={"restoredSid": sess.sid},
        force_project_id=project_id,
    )

    return ProjectOpenResponse(
        ok=True,
        projectId=project_id,
        sid=sess.sid,
        restored=True,
        graphEnabled=bool(sess.graph_auto and sess.graph_runtime),
        strokeCount=len(getattr(sess, "full_strokes", None) or sess.strokes or []),
        graphSummary=graph_summary,
        strokes=[s for s in (getattr(sess, "full_strokes", None) or []) if isinstance(s, dict)],
    )


@app.get("/project/detail", response_model=ProjectDetailResponse)
def project_detail(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        raise HTTPException(400, "project_id required")
    try:
        bundle = PROJECT_STORE.load_project(pid)
    except FileNotFoundError:
        raise HTTPException(404, f"project not found: {pid}")
    except Exception as exc:
        raise HTTPException(500, f"project detail load failed: {exc}")
    current_doc = None
    current_summary = bundle.get("current") if isinstance(bundle.get("current"), dict) else {}
    preview_meta = (current_summary.get("preview") if isinstance(current_summary, dict) else None)
    if isinstance(preview_meta, dict) and preview_meta:
        current_doc = {
            "updatedAt": preview_meta.get("updatedAt") or current_summary.get("previewUpdatedAt"),
            "note": preview_meta.get("note"),
            "mime": preview_meta.get("mime"),
            "width": preview_meta.get("width"),
            "height": preview_meta.get("height"),
            "bbox": preview_meta.get("bbox"),
            "imageUrl": f"/project/current/image?project_id={quote(pid)}",
        }
    commits = []
    for item in (bundle.get("commits") or []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("commitId") or "").strip()
        image_url = None
        if cid:
            image_url = f"/project/commit/image?project_id={quote(pid)}&commit_id={quote(cid)}"
        commits.append({**item, "imageUrl": image_url})
    snapshots = []
    for item in (bundle.get("snapshots") or []):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("snapshotId") or "").strip()
        image_url = None
        if sid:
            image_url = (
                f"/project/snapshot/image?project_id={quote(pid)}&snapshot_id={quote(sid)}"
            )
        snapshots.append({**item, "imageUrl": image_url})
    return ProjectDetailResponse(
        ok=True,
        projectId=pid,
        meta=(bundle.get("meta") or {}),
        current=current_doc,
        commits=commits,
        legacySnapshots=snapshots,
        snapshots=snapshots,
    )


@app.get("/project/current/image")
def project_current_image(project_id: str):
    pid = str(project_id or "").strip()
    if not pid:
        raise HTTPException(400, "project_id required")
    try:
        data, mime = PROJECT_STORE.read_current_preview_image(pid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"current preview image read failed: {exc}")
    return Response(content=data, media_type=mime)


@app.get("/project/commit/image")
def project_commit_image(project_id: str, commit_id: str):
    pid = str(project_id or "").strip()
    cid = str(commit_id or "").strip()
    if not pid or not cid:
        raise HTTPException(400, "project_id and commit_id required")
    try:
        data, mime = PROJECT_STORE.read_commit_preview_image(pid, cid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"commit preview image read failed: {exc}")
    return Response(content=data, media_type=mime)


@app.get("/project/snapshot/image")
def project_snapshot_image(project_id: str, snapshot_id: str):
    pid = str(project_id or "").strip()
    sid = str(snapshot_id or "").strip()
    if not pid or not sid:
        raise HTTPException(400, "project_id and snapshot_id required")
    try:
        data, mime = PROJECT_STORE.read_snapshot_image(pid, sid)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"snapshot image read failed: {exc}")
    return Response(content=data, media_type=mime)


@app.post("/session/sync", response_model=SyncSessionResponse)
def session_sync(body: SyncSessionRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    raw = [s.model_dump() for s in (body.strokes or [])]
    if sess.graph_auto and sess.graph_runtime:
        runtime = sess.graph_runtime
        runtime_vision_image_mode = getattr(body, "vision_image_mode", None)
        if runtime_vision_image_mode:
            try:
                runtime.set_vision_image_mode(runtime_vision_image_mode)
            except Exception as exc:
                print("[graph] set vision image mode failed:", exc)
        if body.graph_snapshot:
            try:
                runtime.update_canvas_snapshot(body.graph_snapshot.model_dump())
            except Exception as exc:
                print("[graph] snapshot update error:", exc)
        if raw:
            try:
                runtime.sync_strokes_snapshot(raw)
            except Exception as exc:
                print("[graph] ingest error:", exc)
    sess.replace_strokes(raw)
    _persist_project_for_session(
        sess,
        reason="session_sync",
        extra={"strokeCount": len(raw)},
    )
    return SyncSessionResponse(ok=True, count=len(sess.strokes))


@app.post("/completion", response_model=CompletionResponse)
def completion(body: CompletionRequest):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text required for completion")
    sys = (
        "You are a precise writing assistant. "
        "Continue the provided passage in the same tone, language, and style. "
        "You should decide to give a short or long continuation based on the input length. "
        "If the format is like term: , you should give precise definitions or explanations of the term. "
        "Do not repeat the existing text. Respond with JSON {\"completion\":\"...\"}."
    )
    user = (
        "Continue this passage (completion). Keep the same style and formatting.\n\n"
        f"{text}"
    )
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    parsed, dbg = call_chat_completions(
        msgs,
        max_tokens=body.max_tokens or 12000,
        temperature=0.6,
        top_p=0.9,
    )
    completion = str(parsed.get("completion") or parsed.get("text") or "").strip()
    if not completion:
        completion = str(dbg.get("raw_text") or "").strip().strip('"')
    if not completion:
        raise HTTPException(502, "Completion model returned empty text.")
    return CompletionResponse(completion=completion)


@app.post("/graph/auto-mode", response_model=GraphAutoModeResponse)
def graph_auto_mode(body: GraphAutoModeRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    if body.enabled:
        runtime = sess.init_graph_runtime(canvas_size=body.canvas_size)
        runtime_vision_image_mode = getattr(body, "vision_image_mode", None)
        if runtime_vision_image_mode:
            try:
                runtime.set_vision_image_mode(runtime_vision_image_mode)
            except Exception as exc:
                print("[graph] set vision image mode failed:", exc)
        strokes = body.strokes or []
        payloads = []
        for stroke in strokes:
            if hasattr(stroke, "model_dump"):
                payloads.append(stroke.model_dump())
            elif isinstance(stroke, dict):
                payloads.append(stroke)
        if payloads:
            try:
                runtime.ingest_strokes(payloads)
            except Exception as exc:
                print("[graph] initial ingest failed:", exc)
        elif sess.strokes:
            try:
                runtime.ingest_strokes(sess.strokes)
            except Exception as exc:
                print("[graph] fallback ingest failed:", exc)
        if body.graph_snapshot:
            try:
                runtime.update_canvas_snapshot(body.graph_snapshot.model_dump())
            except Exception as exc:
                print("[graph] initial snapshot update error:", exc)
        _persist_project_for_session(
            sess,
            reason="graph_auto_mode_on",
            extra={"strokeCount": len(payloads)},
        )
        return GraphAutoModeResponse(ok=True, enabled=True)
    sess.disable_graph_runtime()
    _persist_project_for_session(sess, reason="graph_auto_mode_off")
    return GraphAutoModeResponse(ok=True, enabled=False)


@app.get("/graph/state", response_model=GraphSnapshotResponse)
def graph_state(sid: str):
    sess = S.get_session(sid)
    if not sess or not sess.graph_runtime:
        return GraphSnapshotResponse(blocks=[], fragments=[], groups=[], visionPendingGroups=[])
    snapshot = sess.graph_runtime.snapshot()
    return GraphSnapshotResponse(
        blocks=snapshot.get("blocks", []),
        fragments=snapshot.get("fragments", []),
        groups=snapshot.get("groups", []),
        visionPendingGroups=snapshot.get("visionPendingGroups", []),
    )


@app.post("/graph/promote-group", response_model=PromoteGroupResponse)
def graph_promote_group(body: PromoteGroupRequest):
    sess = S.get_session(body.sid)
    if not sess or not sess.graph_runtime:
        raise HTTPException(404, f"session not found or graph disabled: {body.sid}")
    block = sess.graph_runtime.promote_group_now(body.group_id)
    if not block:
        raise HTTPException(404, f"group not found or already promoted: {body.group_id}")
    payload = {
        "blockId": block.block_id,
        "label": block.label,
        "summary": block.summary,
        "contents": list(block.contents),
    }
    _persist_project_for_session(
        sess,
        reason="graph_promote_group",
        extra={"groupId": body.group_id, "blockId": block.block_id},
    )
    return PromoteGroupResponse(ok=True, block=payload)


@app.post("/graph/promote-vision-group", response_model=PromoteVisionPendingGroupResponse)
def graph_promote_vision_group(body: PromoteVisionPendingGroupRequest):
    sess = S.get_session(body.sid)
    if not sess or not sess.graph_runtime:
        raise HTTPException(404, f"session not found or graph disabled: {body.sid}")
    if body.graph_snapshot:
        try:
            sess.graph_runtime.update_canvas_snapshot(body.graph_snapshot.model_dump())
        except Exception as exc:
            print("[graph] promote vision snapshot update error:", exc)
    ok = sess.graph_runtime.promote_vision_pending_group_now(body.group_id)
    if not ok:
        raise HTTPException(404, f"pending vision group not found or processing failed: {body.group_id}")
    _persist_project_for_session(
        sess,
        reason="graph_promote_vision_group",
        extra={"groupId": body.group_id},
    )
    return PromoteVisionPendingGroupResponse(ok=True)


@app.post("/graph/selection-block-action", response_model=GraphSelectionBlockActionResponse)
def graph_selection_block_action(body: GraphSelectionBlockActionRequest):
    sess = S.get_session(body.sid)
    if not sess or not sess.graph_runtime:
        raise HTTPException(404, f"session not found or graph disabled: {body.sid}")
    try:
        result = sess.graph_runtime.apply_selection_block_action(
            action=body.action,
            fragment_ids=body.fragment_ids,
            target_block_id=body.target_block_id,
            label_hint=body.label_hint,
            focus_after=body.focus_after,
        )
    except Exception as exc:
        raise HTTPException(400, f"selection block action failed: {exc}")
    _persist_project_for_session(
        sess,
        reason="graph_selection_block_action",
        extra={
            "action": body.action,
            "fragmentCount": len(body.fragment_ids or []),
            "targetBlockId": body.target_block_id,
        },
    )
    return GraphSelectionBlockActionResponse(
        ok=True,
        action=str(result.get("action") or body.action),
        block=result.get("block"),
        fragment_ids=[str(fid) for fid in (result.get("fragmentIds") or [])],
    )


@app.post("/graph/remove-fragments", response_model=GraphRemoveFragmentsResponse)
def graph_remove_fragments(body: GraphRemoveFragmentsRequest):
    sess = S.get_session(body.sid)
    if not sess or not sess.graph_runtime:
        raise HTTPException(404, f"session not found or graph disabled: {body.sid}")
    try:
        result = sess.graph_runtime.remove_fragments_now(body.fragment_ids)
    except Exception as exc:
        raise HTTPException(400, f"remove fragments failed: {exc}")

    removed_ids = [str(fid) for fid in (result.get("removedFragmentIds") or []) if str(fid or "").strip()]
    removed_set = set(removed_ids)
    if removed_set:
        # Keep session snapshot aligned to avoid re-ingesting deleted fragments later.
        sess.full_strokes = [
            stroke
            for stroke in (sess.full_strokes or [])
            if str((stroke or {}).get("id") or "").strip() not in removed_set
        ]
        sess.strokes = [
            stroke
            for stroke in (sess.strokes or [])
            if str((stroke or {}).get("id") or "").strip() not in removed_set
        ]
        _persist_project_for_session(
            sess,
            reason="graph_remove_fragments",
            extra={"removedCount": len(removed_set), "fragmentIds": removed_ids[:200]},
        )

    return GraphRemoveFragmentsResponse(
        ok=True,
        removedFragmentIds=removed_ids,
        removedCount=len(removed_ids),
    )

