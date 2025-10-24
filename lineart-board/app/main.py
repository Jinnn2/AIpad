# -*- coding: utf-8 -*-
from __future__ import annotations
import os, random, time, json, tempfile, math
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas import (
    SuggestRequest, SuggestResponse, AIStrokePayload, Health,
    AIStrokeV11, StrokeStyle, CanvasInfo,
    SyncSessionRequest, SyncSessionResponse
)
from app import prompting
from app.llm_client import call_chat_completions
from starlette.responses import JSONResponse
import re
from app import session_store as S
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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Default to allowing http://localhost:* and http://127.0.0.1:* during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
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
                parsed, dbg = call_chat_completions(msgs, max_tokens=800)
                analysis = ""
                instruction = ""
                if isinstance(parsed, dict):
                    analysis = str(parsed.get("analysis","") or "").strip()
                    instruction = str(parsed.get("instruction","") or "").strip()
                if not instruction:
                    instruction = (req.hint or "Make the single best next stroke.")
                if LOG_IO:
                    try: _write_json(log_dir, "output.step1.json", {"analysis":analysis, "instruction":instruction, "raw_text": (dbg or {}).get("raw_text")})
                    except Exception: pass
                # Step-1 emits no strokes; it only returns vision metadata for Step-2.
                return SuggestResponse(
                    payload=AIStrokePayload(version=2, intent="hint", strokes=[]),
                    usage={"mode":"vision-2.0-step1","raw_text": (dbg or {}).get("raw_text")},
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

                orig_hint = str(getattr(req, "hint", "") or "").strip()
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
                usage = {"mode":"vision-2.0-step2", "raw_text": (dbg or {}).get("raw_text")}
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
            sess = S.create_session(mode="light_helper", init_goal=req.hint or None, tags=None)
            new_sid = sess.sid
        # Merge incremental strokes when provided.
        if req.delta and isinstance(req.delta, DeltaPayload):
            S.Session.append_strokes(sess, [s.model_dump() for s in req.delta.strokes])  # type: ignore
        # Replace session state with the provided snapshot to keep deletes/erasures consistent.
        if req.context and isinstance(req.context.strokes, list):
            sess.replace_strokes([s.model_dump() for s in req.context.strokes])  # type: ignore
        # Increment counters used to decide whether to inject samples.
        cc = sess.bump()
        include_sample = S.should_include_sample(cc)
        # Build a lightweight context from the most recent strokes.
        recent = sess.recent_for_model()
        def _r3(v: float) -> float:
            try: return round(float(v), 3)
            except Exception: return float(v)
        lite_ctx = AIStrokePayload(version=1, intent="complete", strokes=[
            AIStrokeV11(
                id=s["id"], tool=s.get("tool","pen"),
                # Keep only x/y components and drop None placeholders.
                points=[[ _r3(p[0]), _r3(p[1]) ] for p in s.get("points", [])],
                style=StrokeStyle(size=((s.get("style") or {}).get("size") or "m"),
                                  color=((s.get("style") or {}).get("color") or "black"),
                                  opacity=float((s.get("style") or {}).get("opacity") or 1.0)),
                meta=s.get("meta") or {}
            ) for s in recent
        ])
        # Pass gen_scale through to bound the number of generated points.
        fake = SuggestRequest(context=lite_ctx, hint=req.hint, model=req.model,
                              temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens,
                              gen_scale=req.gen_scale)
        messages = prompting.build_messages_by_mode(fake, getattr(req, "mode", None))
        if LOG_IO:
            try: _write_json(log_dir, "input.messages.json", messages)
            except Exception: pass
    else:
        # Remain compatible with legacy full-context payloads.
        if not req.context:
            raise HTTPException(400, "Either {sid, delta} or {context} must be provided.")
        messages = prompting.build_messages_by_mode(req, getattr(req, "mode", None))
        if LOG_IO:
            try: _write_json(log_dir, "input.messages.json", messages)
            except Exception: pass

    # MOCK shortcut: serve a deterministic payload when no key or demo mode.
    if MOCK:
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
    obj, dbg = call_chat_completions(
        messages=messages,
        model=req.model,
        temperature=req.temperature or 0.4,
        top_p=req.top_p or 0.95,
        max_tokens=req.max_tokens or 1024,
    )
    
    # Normalizer: turn raw LLM strokes into clean renderable data.
    def _clamp01(v: float) -> float:
        try: return max(0.0, min(1.0, float(v)))
        except Exception: return 1.0

    def _r3(v: float) -> float:
        try: return round(float(v), 3)
        except Exception: return float(v)

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

    def _clean_one(s: dict, gen_scale: int) -> dict:
        # 1) Normalize points: keep up to the first four entries, minimum two points.
        raw_pts = s.get("points") or []
        pts = [
            [float(p[0]), float(p[1])]
            + ([p[2]] if len(p) > 2 else [])
            + ([p[3]] if len(p) > 3 else [])
            for p in raw_pts
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if len(pts) < 2:
            raise HTTPException(502, "LLM stroke has <2 points.")

        tool_in = str(s.get("tool") or "pen").lower()

        # 2) Tool-specific cleanup.
        if tool_in == "poly":
            if len(pts) < 3:
                raise HTTPException(502, "LLM poly needs >= 3 points.")
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
                raise HTTPException(502, "LLM ellipse needs 2 points.")
            x0 = min(pts[0][0], pts[-1][0]); y0 = min(pts[0][1], pts[-1][1])
            x1 = max(pts[0][0], pts[-1][0]); y1 = max(pts[0][1], pts[-1][1])
            pts2 = [[x0, y0], [x1, y1]]
            tool = "ellipse"

        elif tool_in == "text":
            # expect at least two points [[x,y],[x2,y2]]
            raw_pts = s.get("points") or []
            if not (isinstance(raw_pts, list) and len(raw_pts) >= 2):
                raise HTTPException(502, "LLM text box needs 2 points.")
            p0 = raw_pts[0]; p1 = raw_pts[1]
            try:
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
            except Exception:
                raise HTTPException(502, "invalid text box points")

            tx0 = min(x0, x1); ty0 = min(y0, y1)
            tx1 = max(x0, x1); ty1 = max(y0, y1)
            pts2 = [[tx0, ty0], [tx1, ty1]]

            style = s.get("style") or {}
            size = style.get("size") or "m"
            color = style.get("color") or "black"
            opacity = _clamp01(style.get("opacity", 1.0))

            return {
                "id": str(s.get("id") or f"ai_{int(time.time())}"),
                "tool": "text",
                "points": [[_r3(p[0]), _r3(p[1])] for p in pts2],
                "style": {"size": size, "color": color, "opacity": opacity},
                "meta": s.get("meta") or {},
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
        color = style.get("color") or "black"
        opacity = _clamp01(style.get("opacity", 1.0))

        return {
            "id": str(s.get("id") or f"ai_{int(time.time())}"),
            "tool": tool,
            "points": [[_r3(p[0]), _r3(p[1])] for p in pts2],
            "style": {"size": size, "color": color, "opacity": opacity},
            "meta": s.get("meta") or {},
        }

    def _clean_payload(obj, gen_scale: int):
        # Keep and sanitize every stroke.
        strokes_in = (obj.get("strokes") or []) if isinstance(obj, dict) else []
        if not isinstance(strokes_in, list) or len(strokes_in) == 0:
            raise HTTPException(502, "LLM returned empty strokes.")
        cleaned_list = []
        for s in strokes_in:
            try:
                cleaned_list.append(_clean_one(s, gen_scale))
            except Exception as e:
                # Skip invalid strokes but preserve at least one valid entry.
                print("[clean] drop one stroke:", e)
        if not cleaned_list:
            raise HTTPException(502, "All strokes invalid after cleaning.")

        # Pass through intent/canvas/replace when supplied.
        intent = (obj.get("intent") or "complete")
        canvas = obj.get("canvas") or None
        replace = obj.get("replace") or None

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

        return cleaned

    obj_clean = _clean_payload(obj, req.gen_scale or 24)

    # pydantic validation as a safety net.
    try:
        payload = AIStrokePayload.model_validate(obj_clean)
        if LOG_IO and log_dir is not None:
            try: _write_json(log_dir, "output.cleaned.json", payload.model_dump())
            except Exception: pass
        usage = {
            "stage": "ok",
            "raw_text": dbg.get("raw_text"),
            "mode": dbg.get("mode"),
            "model": dbg.get("model"),
            "response_id": (dbg.get("response_dump") or {}).get("id"),
        }
        if new_sid: usage["new_sid"] = new_sid
        return SuggestResponse(ok=True, payload=payload, usage=usage)
    
    except Exception as e:
        if LOG_IO and log_dir is not None:
            try:
                _write_json(
                    log_dir,
                    "output.error.json",
                    {"error": "invalid payload", "detail": str(e), "raw": obj, "raw_text": dbg.get("raw_text")},
                )
            except Exception:
                pass
        raise HTTPException(502, f"LLM returned invalid payload after cleaning: {e} | raw={dbg.get('raw_text')!r}")
    

# ---------------------------------------- Session management endpoints ---------------------------------------- #
# Session initialization endpoint.
@app.post("/session/init", response_model=InitSessionResponse)
def session_init(body: InitSessionRequest):
    s = S.create_session(mode=body.mode, init_goal=body.init_goal, tags=body.tags)
    return InitSessionResponse(sid=s.sid, note="ok")


@app.post("/session/sync", response_model=SyncSessionResponse)
def session_sync(body: SyncSessionRequest):
    sess = S.get_session(body.sid)
    if not sess:
        raise HTTPException(404, f"session not found: {body.sid}")
    raw = [s.model_dump() for s in (body.strokes or [])]
    sess.replace_strokes(raw)
    return SyncSessionResponse(ok=True, count=len(sess.strokes))
