# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import SuggestRequest


LIGHT_SYSTEM = (
    "Role: On-canvas assistant for NEXT-STROKE prediction.\n"
    "Rules:\n"
    " - Output JSON ONLY and MUST conform to AIStrokePayload v1.1.\n"
    " - ABSOLUTE pixel coordinates.\n"
    " - STRICTLY ONE stroke in 'strokes' (exactly one item).\n"
    " - Tool selection: 'line' (2 pts), 'poly' (>=3 vertices), 'ellipse' (2 opposite corners), 'pen' (freeform keypoints).\n"
    " - Prefer concise keypoints; do not densify samples.\n"
    " - If prefer_explanatory_drawing=true in user content, favor a clarifying diagrammatic stroke that explains existing content (not decorative marks).\n"
    " - No markdown / no prose / no comments.\n"
)

LIGHT_CONTRACT = (
    "Return fields: version=1, intent鈭坽'complete','hint','alt'} (prefer 'complete'), strokes[1].\n"
    "Stroke shape:\n"
    " - line: exactly 2 points [[x0,y0],[x1,y1]].\n"
    " - poly: >=3 vertices in order; if closed, last MAY equal first.\n"
    " - ellipse: exactly 2 points as bounding-box diagonal.\n"
    " - pen: multiple keypoints, up to {max_pts}.\n"
    "Style:\n"
    " - size鈭坽's','m','l','xl'}; color鈭坽black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow}; opacity鈭圼0,1].\n"
)

LIGHT_NOTES = (
    "Output JSON only. STRICTLY one stroke. Prefer keypoints under {max_pts}. "
    "If the shape is obviously straight, use 'line'."
)


def _downsample_polyline(points, max_pts=12):
    """
    绾挎€х瓑璺濇娊鏍凤紙鍚袱绔偣锛夛紝浠呬繚鐣?[x, y]锛屽幓闄?t/pressure 绛夊啑浣欍€?
    """
    if not isinstance(points, list) or len(points) <= max_pts:
        return [[float(p[0]), float(p[1])] for p in points]
    import math

    pts = [[float(p[0]), float(p[1])] for p in points]
    seg = []
    total = 0.0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        seg.append(d)
        total += d
    if total <= 1e-9:
        return [pts[0], pts[-1]]
    out = [pts[0]]
    steps = max_pts - 1
    acc = 0.0
    j = 0
    for s in range(1, steps):
        target = total * s / steps
        while j < len(seg) and acc + seg[j] < target:
            acc += seg[j]
            j += 1
        if j >= len(seg):
            out.append(pts[-1])
            break
        t = (target - acc) / (seg[j] or 1e-9)
        x = pts[j][0] + t * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
        out.append([x, y])
    out.append(pts[-1])
    return out


def _compress_context(ctx: dict, keep_last=60, max_pts=12, drop_ai=True) -> dict:
    """
    灏?SuggestContext 鍘嬬缉涓烘瀬绠€鏍煎紡锛屼粎鍖呭惈蹇呰鐨勪汉绫荤瑪鐢伙細
      - 鍙繚鐣欐渶杩?keep_last 鏉?
      - 姣忔潯鏈€澶?max_pts 涓偣锛圼x,y]锛?
      - 鍙€夛細涓㈠純鍘嗗彶 AI 绗旂敾锛堝噺灏戔€滆嚜鎴戝洖澹扳€濓級
      - 涓㈠純 style/meta/canvas 绛夋棤鍏冲瓧娈?
    杈撳嚭缁撴瀯锛堜緵鎻愮ず璇嶉槄璇伙紝涓嶈姹傜鍚?AIStrokePayload锛夛細
      {
        "H": [ [tool:str, [[x,y],...]], ... ],
        "C": [w, h]  # 鍙€夛細鐢诲竷灏哄锛堝瀛樺湪锛?
      }
    """
    out = {"H": []}
    if not isinstance(ctx, dict):
        return out

    strokes = ctx.get("strokes") or []
    if drop_ai:
        strokes = [
            s for s in strokes
            if not (isinstance(s, dict) and (s.get("meta") or {}).get("author") == "ai")
        ]

    if len(strokes) > keep_last:
        strokes = strokes[-keep_last:]

    for s in strokes:
        tool = str(s.get("tool") or "pen")
        pts = s.get("points") or []
        pts2 = _downsample_polyline(pts, max_pts=max_pts)
        out["H"].append([tool, pts2])

    canvas = ctx.get("canvas") or {}
    if (
        isinstance(canvas, dict)
        and "size" in canvas
        and isinstance(canvas["size"], (list, tuple))
        and len(canvas["size"]) >= 2
    ):
        out["C"] = [canvas["size"][0], canvas["size"][1]]

    return out


def build_messages_light(req: SuggestRequest, include_sample: bool = False) -> List[Dict[str, Any]]:
    max_pts = int(getattr(req, "gen_scale", 12) or 12)
    max_pts = max(6, min(24, max_pts))

    ctx = req.context.model_dump()
    mini_ctx = _compress_context(ctx, keep_last=60, max_pts=max_pts, drop_ai=True)

    user_content: Dict[str, Any] = {
        "mode": "light-completion",
        "goal": (req.hint or "Predict the single next stroke (sketch or text) continuing user's intent."),
        "context_min": mini_ctx,
        "contract": LIGHT_CONTRACT.format(max_pts=max_pts),
        "notes": LIGHT_NOTES.format(max_pts=max_pts),
    }
    prefer_explanatory_drawing = getattr(req, "prefer_explanatory_drawing", None)
    if isinstance(prefer_explanatory_drawing, bool):
        user_content["prefer_explanatory_drawing"] = prefer_explanatory_drawing

    if include_sample:
        user_content["sample"] = {
            "version": 1,
            "intent": "complete",
            "strokes": [
                {
                    "id": "ai_next_light_001",
                    "tool": "line",
                    "points": [[320, 180], [460, 220]],
                    "style": {"size": "m", "color": "black", "opacity": 1.0},
                    "meta": {"author": "ai"},
                }
            ],
        }

    return [
        {"role": "system", "content": LIGHT_SYSTEM},
        {"role": "user", "content": f"{user_content}"},
    ]
