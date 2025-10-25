# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
from app.schemas import SuggestRequest
import json

# ============ Full 模式（多笔补全 + 文字理解） ============
# 1) 强 system 约束
SYSTEM_INSTRUCT = (
    "Role: You are an on-canvas work assistant that draw strokes or generate texts based on the HINT and Existing content.\n"
    "Behavior rules:\n"
    " - Return JSON objects that strictly conforms to AIStrokePayload.\n"
    " - Coordinates are ABSOLUTE canvas space (pixels).\n"
    " - You can DRAW, WRITE, or EDIT existing text.\n"
    " - DRAW: use 'pen','line','poly','ellipse' tools to draw shapes/lines.\n"
    " - If you want to add a straight line, use tool='line' and provide exactly 2 points [p0, pn].\n"
    " - If the intent is a CLOSED polygonal shape (rectangle, triangle, loop), use tool='poly' with >=3 vertices.\n"
    " - If you want an ellipse, use tool='ellipse' and provide exactly 2 points [p0, pn] as the bounding-box diagonal.\n"
    " - For freeform curves, use tool='pen' with multiple points.\n"
    " - For pen: provide as many points as possible, up to the limit given.\n"
    "   For poly: points are vertices in order; the last point MUST repeat the first to explicitly close the loop.\n"
    " - Before generating, carefully ANALYZE whether you use a LINE, POLY, ELLIPSE or PEN.\n"
    " - For curves, prefer concise key points; do NOT densely sample every pixel.\n"
    " - The Length Baseline is 200px each segment.\n"
    " - WRITE: use tool='text' to ADD or EDIT text."
    "* points = [[x,y],[x+w,y+h]] where [x,y] is top-left corner, [x+w,y+h] is bottom-right corner.\n"
    "* style.color is the text color (must be from the palette).\n"
    "* meta MUST include:\n"
    "    \"text\": full multiline content,\n"
    "    \"summary\": short summary (<=30 chars),\n"
    "    \"fontFamily\": e.g. \"sans-serif\",\n"
    "    \"fontWeight\": e.g. \"400\" or \"bold\",\n"
    "    \"fontSize\": font size in px,\n"
    "    \"growDir\": one of {\"down\",\"right\",\"up\",\"left\"} (default \"down\").\n"
    "- EDIT text boxes using tool='edit' when you need to modify a previous text stroke.\n"
    "* meta MUST include: targetId (the existing stroke id), operation (<=60 chars describing the intent), content (the rewritten preview text). Optionally include updated text/font metadata.\n"
    "* If you supply points for edit, still use [[x,y],[x+w,y+h]] covering the target area."
)

# 2) 明确的输出契约 + 正确示例
OUTPUT_CONTRACT = (
    "Return fields: version, intent, canvas(optional), replace(optional), strokes[].\n"
    "Constraints:\n"
    " - version = 1 (integer)\n"
    " - intent ∈ {'complete','hint','alt','write'}; prefer 'complete'\n"
    " - You should combine multiple strokes to reach the scale if necessary.\n"
    " - number of strokes: it should MATCH the scale = {max_pts}. If you use little points in each stroke, then increase the number of outputs\n"
    " - Each stroke: { id:string, tool:string in {'pen','line','poly','ellipse','text','edit'}, points:[[x,y,(t?),(pressure?)]...], style{size,color,opacity}, meta }\n"
    " - For 'line': exactly two points [p0, pn].\n"
    " - For 'poly': provide >=3 vertices in order; last MAY equal first to denote closure.\n"
    " - For 'ellipse': exactly two points [p0, pn] as the bounding-box diagonal.\n"
    " - For 'pen': multiple keypoints, prefer concise points up to {max_pts}.\n"
    " - For 'text': points = [[x,y],[x+w,y+h]]; style.color from palette; meta includes text, summary, fontFamily, fontWeight, fontSize, growDir.\n"
    " - For 'edit': meta includes targetId, operation, content (preview text). Points optional but recommended to reuse the target bounding box.\n"
    " - When it is not a line, try to use as much points as limited: {max_pts} \n"
    " - Try to use multiple styles and colors if they MAKE SENCE.\n"
    " - Use reasonable style: size in {'s','m','l','xl'}, opacity in [0,1]\n"
    " - Use colors in palette: black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow\n"
)

SAMPLE_ONE_STROKE = {
    "version": 1,
    "intent": "complete",
    "strokes": [
        {
            "id": "ai_next_001",
            "tool": "pen",
            "points": [[320,180],[360,190],[400,220],[420,250],[480,280]],
            "style": { "size": "m", "color": "red", "opacity": 0.9 },
            "meta": { "author": "ai", "desc": "next curve segment" }
        }
    ]
}
SAMPLE_ONE_POLY = {
    "version": 1,
    "intent": "complete",
    "strokes": [
        {
            "id": "ai_poly_001",
            "tool": "poly",
            "points": [[420,200],[520,200],[470,280],[420,200]],
            "style": { "size": "m", "color": "orange", "opacity": 0.8 },
            "meta": { "author": "ai", "desc": "closed triangle" }
        }
    ]
}
SAMPLE_TEXTBOX = {
    "version": 1,
    "intent": "write",
    "strokes": [
        {
            "id": "ai_text_001",
            "tool": "text",
            "points": [[100,120],[260,200]],
            "style": { "size": "m", "color": "black", "opacity": 1.0 },
            "meta": {
                "text": "电路分析注意：\n1. 节点电位法\n2. 叠加原理",
                "summary": "电路分析要点",
                "fontFamily": "sans-serif",
                "fontWeight": "bold",
                "fontSize": 16,
                "growDir": "down"
            }
        }
    ]
}

def build_messages(req: SuggestRequest, include_sample: bool = True) -> list[dict[str, Any]]:
    """
    Light Helper（只补一笔）消息构造：
    - system：严格协议与行为约束（只返回一个 JSON、只一条 stroke、version=1）
    - user：包含初始化目标（来自 hint）+ 裁剪后的上下文 strokes
    - 最后附上输出契约与正确样例（模型更少犯格式错）
    """
    # 裁剪策略：只保留最近 N 条（避免上下文过长）
    N = 200
    ctx = req.context.model_dump()
    if isinstance(ctx.get("strokes"), list) and len(ctx["strokes"]) > N:
        ctx["strokes"] = ctx["strokes"][-N:]

    # 生成规模（点数）约束：若未给，则默认 16
    max_pts = int(req.gen_scale) if (hasattr(req, "gen_scale") and req.gen_scale) else 16
    max_pts = max(4, min(64, max_pts))
    # 用户内容：把“本次任务目标/初始化描述”放在 hint；上下文包含最近的人类笔画
    user_content = {
        "mode": "draw expert",
        "goal": req.hint or "Draw continuing user's intent.",
        "context": ctx,
        "output_contract": OUTPUT_CONTRACT,
        **({"samples": [SAMPLE_ONE_STROKE, SAMPLE_ONE_POLY]} if include_sample else {}),
        "notes": (
            "Return strokes. If mentioned COMPLETION, just return ONE stroke. JSON object only. "
            "The more {max_pts} given, the longer the stroke you should draw. Baseline is 16 points, refering to about 200px length. "
            "Prefer concise keypoints over dense samples."
        ),
        "Setting": { "Scale": max_pts }
    }
    return [
        {"role": "system", "content": SYSTEM_INSTRUCT},
        {"role": "user", "content": f"{user_content}"}
    ]


def _downsample_polyline(points, max_pts=12):
    """
    线性等距抽样（含两端点），仅保留 [x, y]，去掉 t/pressure 等冗余。
    """
    if not isinstance(points, list) or len(points) <= max_pts:
        # 只截断到 [x,y]
        return [[float(p[0]), float(p[1])] for p in points]
    # 计算分段长度
    import math
    pts = [[float(p[0]), float(p[1])] for p in points]
    seg = []
    total = 0.0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        seg.append(d); total += d
    if total <= 1e-9:
        return [pts[0], pts[-1]]
    out = [pts[0]]
    steps = max_pts - 1
    acc = 0.0; j = 0
    for s in range(1, steps):
        target = total * s / steps
        while j < len(seg) and acc + seg[j] < target:
            acc += seg[j]; j += 1
        if j >= len(seg):
            out.append(pts[-1]); break
        t = (target - acc) / (seg[j] or 1e-9)
        x = pts[j][0] + t * (pts[j+1][0] - pts[j][0])
        y = pts[j][1] + t * (pts[j+1][1] - pts[j][1])
        out.append([x, y])
    out.append(pts[-1])
    return out

def _compress_context(ctx: dict, keep_last=60, max_pts=12, drop_ai=True) -> dict:
    """
    将 SuggestContext 压缩为极简格式，仅包含必要的人类笔画：
      - 只保留最近 keep_last 条
      - 每条最多 max_pts 个点（[x,y]）
      - 可选：丢弃历史 AI 笔画（减少“自我回声”）
      - 丢弃 style/meta/canvas 等无关字段
    输出结构（供提示词阅读，不要求符合 AIStrokePayload）：
      {
        "H": [ [tool:str, [[x,y],...]], ... ],  # 压缩后的人类笔画
        "C": [w, h]  # 可选：画布尺寸（如存在）
      }
    """
    out = {"H": []}
    if not isinstance(ctx, dict):
        return out

    strokes = ctx.get("strokes") or []
    if drop_ai:
        strokes = [s for s in strokes if not (isinstance(s, dict) and (s.get("meta") or {}).get("author") == "ai")]

    if len(strokes) > keep_last:
        strokes = strokes[-keep_last:]

    for s in strokes:
        tool = str(s.get("tool") or "pen")
        pts  = s.get("points") or []
        pts2 = _downsample_polyline(pts, max_pts=max_pts)
        out["H"].append([tool, pts2])

    # 可选带上画布尺寸（若在 context.canvas 有）
    canvas = ctx.get("canvas") or {}
    if isinstance(canvas, dict) and "size" in canvas and isinstance(canvas["size"], (list, tuple)) and len(canvas["size"]) >= 2:
        out["C"] = [canvas["size"][0], canvas["size"][1]]

    return out

# ============ Light Helper 模式（只补“一笔”） ============
LIGHT_SYSTEM = (
    "Role: On-canvas assistant for NEXT-STROKE prediction.\n"
    "Rules:\n"
    " - Output JSON ONLY and MUST conform to AIStrokePayload v1.1.\n"
    " - ABSOLUTE pixel coordinates.\n"
    " - STRICTLY ONE stroke in 'strokes' (exactly one item).\n"
    " - Tool selection: 'line' (2 pts), 'poly' (>=3 vertices), 'ellipse' (2 opposite corners), 'pen' (freeform keypoints).\n"
    " - Prefer concise keypoints; do not densify samples.\n"
    " - No markdown / no prose / no comments.\n"
)

LIGHT_CONTRACT = (
    "Return fields: version=1, intent∈{'complete','hint','alt'} (prefer 'complete'), strokes[1].\n"
    "Stroke shape:\n"
    " - line: exactly 2 points [[x0,y0],[x1,y1]].\n"
    " - poly: >=3 vertices in order; if closed, last MAY equal first.\n"
    " - ellipse: exactly 2 points as bounding-box diagonal.\n"
    " - pen: multiple keypoints, up to {max_pts}.\n"
    "Style:\n"
    " - size∈{'s','m','l','xl'}; color∈{black, blue, green, grey, light-blue, light-green, light-red, light-violet, orange, red, violet, white, yellow}; opacity∈[0,1].\n"
)

def build_messages_light(req: "SuggestRequest", include_sample: bool = False) -> list[dict]:
    """
    轻量补全（仅一笔），极限压缩输入：
      - 丢弃 AI 历史、压缩点数、只保留最近 N 条人类笔画
      - 简短 system + 合同；默认不带样例以省 tokens
      - 强约束 ONLY ONE stroke
    ⚠️ 不影响现有 build_messages 使用方；由前端选择调用本函数即可
    """
    # 规模：默认 12 个点（比常规小，以进一步省 tokens）
    max_pts = int(getattr(req, "gen_scale", 12) or 12)
    max_pts = max(6, min(24, max_pts))

    # 压缩上下文（仅人类笔画）
    ctx = req.context.model_dump()
    mini_ctx = _compress_context(ctx, keep_last=60, max_pts=max_pts, drop_ai=True)

    # 极简 user 内容：目标 + 极简上下文 + 合同
    user_content = {
        "mode": "light-completion",
        "goal": (req.hint or "Predict the single next stroke continuing user's intent."),
        "context_min": mini_ctx,
        "contract": LIGHT_CONTRACT.format(max_pts=max_pts),
        "notes": (
            "Output JSON only. STRICTLY one stroke. Prefer keypoints under {max_pts}. "
            "If the shape is obviously straight, use 'line'."
        ),
    }
    if include_sample:
        # 保留开关，但默认 False 以节省 tokens
        user_content["sample"] = {
            "version": 1,
            "intent": "complete",
            "strokes": [{
                "id": "ai_next_light_001",
                "tool": "line",
                "points": [[320,180],[460,220]],
                "style": {"size": "m", "color": "black", "opacity": 1.0},
                "meta": {"author": "ai"}
            }]
        }

    return [
        {"role": "system", "content": LIGHT_SYSTEM},
        {"role": "user", "content": f"{user_content}"},
    ]

# ============ Vision 模式（画板快照 → 图像提示） ============

VISION_SYSTEM = (
    "Role: On-canvas assistant with visual context. "
    "You are given a snapshot of the canvas; infer user's likely next stroke(s) from the scene."
)

def _vision_user_content(req: "SuggestRequest") -> dict:
    # 文本侧尽量简短，减少 tokens；图像作为主要上下文
    return {
        "mode": "vision",
        "goal": (req.hint or "Predict the next stroke based on the canvas snapshot."),
        "constraints": (
            "Output JSON only (AIStrokePayload v1.1). "
            "Use absolute pixels. Prefer concise keypoints."
        )
    }

def build_messages_vision(req: "SuggestRequest", include_sample: bool = True) -> list[dict]:
    """
    目标：与 full 模式提示词“保持一致”，只额外提供一条包含画板快照的 user 消息。
    - 前置消息：直接复用 build_messages(req, include_sample)
      （system、用户合同、样例等全部与 full 相同）
    - 最后一条：user 消息，内容是一个 dict 字符串，含占位键：
        _image_data: base64（或 dataURL）
        _image_mime: 'image/jpeg' | 'image/png'
      llm_client._inject_vision_content() 会把这条消息转换为
      OpenAI/ChatAnywhere 兼容的多模态 blocks：[text, image_url]
    """
    # 1) 先构造与 full 完全一致的消息
    msgs = build_messages(req, include_sample=include_sample)

    # 2) 追加“图像快照”说明消息（简短文本 + 图像占位）。保持其它指令不变以便对比。
    vision_note = {
        "mode": "vision",
        "note": (
            "The following image is a snapshot of the current canvas. "
            "Keep ALL previous instructions/contracts/examples unchanged (same as FULL mode). "
            "Use the snapshot only as additional context."
        ),
        "_image_data": getattr(req, "image_data", None),
        "_image_mime": getattr(req, "image_mime", None) or "image/jpeg",
        # 可选：带上缩略尺寸供日志参考（无业务含义）
        "snapshot_size": getattr(req, "snapshot_size", None),
    }
    msgs.append({"role": "user", "content": f"{vision_note}"})
    return msgs

def build_vision_v2_step1(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Vision 2.0 - Step 1（图像理解阶段）：
    - 只传快照图像（不传 strokes 点列）
    - 让模型输出一个 JSON：{ analysis, instruction }
      analysis: 对图像中线稿/构图/风格的观察与猜想（简洁要点）
      instruction: 下一步应该画什么（面向“线稿补全/修正”的具体指令）
    """
    view = req["canvas"]["viewport"]
    msg_user_payload = {
        "task": "image_inspection_and_next_move",
        "canvas_viewport": view,
        "hint": req.get("hint") or "",
        "_image_data": req.get("image_data"),
        "_image_mime": req.get("image_mime") or "image/jpeg",
        "rules": [
            "Analysis Part:",
            "Focus on what is ALREADY on the canvas, and give a concise summary.",
            "Use concise language; avoid verbosity.",
            "Instruction Part:",
            "Give clear shape analysis and actionable next-step instruction.",
            "Shapes may include lines, pen(curves), poly, and ellipses.",
            "Examples: 'The most important action is to add a window of the house. You should draw a triangle'.",
            "Return JSON with keys: analysis (~100 words/50字), instruction (~50 words/30字).",
            "instruction should be actionable for next drawing step (concise)."
        ]
    }
    return [
        {"role":"system","content":"You are a precise line-art critic. Analyze the canvas image and propose the single best next stroke idea."},
        {"role":"user","content": json.dumps(msg_user_payload, ensure_ascii=False)}
    ]

def build_full_with_instruction(req: Dict[str, Any], instruction_text: str) -> List[Dict[str, Any]]:
    """
    Full 流程（v1.1 协议不变），但在 system / user 中注入额外的“强化指令”。
    注意：此时允许返回 strokes JSON（v1.1协议），并可使用 req.context.strokes 作为参考。
    """
    view = req["canvas"]["viewport"]
    context = req.get("context") or {}
    strokes = context.get("strokes") or []
    payload = {
        "task": "line_art_next_step",
        "canvas_viewport": view,
        "context_version": context.get("version", 1),
        "intent": context.get("intent", "complete"),
        "strokes": strokes,
        "gen_scale": req.get("gen_scale") or 16,
        "hint": req.get("hint") or "",
        "extra_instruction": instruction_text
    }
    return [
        {"role":"system","content":"You are a structured line-art assistant. Produce JSON {version, intent, strokes[]} with concise geometry (≤ gen_scale points per stroke)."},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]


def build_messages_by_mode(req: "SuggestRequest", mode: str | None, include_sample: bool = True) -> list[dict]:
    """
    供你在后端切换：
      - mode in {'light', 'full'}；其他/None → 走你现有的（不变）
      - 'full'：继续走你当前的 build_messages（保持兼容）
      - 'light'：走上面的 build_messages_light
    """
    m = (mode or "").lower()
    if m == "light":
        return build_messages_light(req, include_sample=False)
    if m == "full":
        # 复用你当前的主模式构造器（保持原样）
        return build_messages(req, include_sample=include_sample)
    if m == "vision":
        return build_messages_vision(req, include_sample=include_sample)
    # 默认：不改变你现有行为
    return build_messages(req, include_sample=include_sample)
