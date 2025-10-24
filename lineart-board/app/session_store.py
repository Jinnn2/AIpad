# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from time import time
import os
import math
import random
import string

# 会话配置（可用 .env 覆盖）
MAX_STROKES = int(os.getenv("SESS_MAX_STROKES", "120"))     # 每个会话保存的最近笔画上限（根）
KEEP_RECENT = int(os.getenv("SESS_KEEP_RECENT", "200"))       # 每次送给模型的最近笔画数
RESAMPLE_STEP = float(os.getenv("SESS_RESAMPLE_STEP", "3"))  # 等距重采样步长（像素）
# 默认不量化（保留浮点）；如需贴格子可在 .env 里显式设 SESS_QUANT=1
QUANT = float(os.getenv("SESS_QUANT", "0"))                  # 量化网格（像素）
SAMPLE_EVERY_N = int(os.getenv("PROMPT_SAMPLE_EVERY_N", "1"))# 样例注入频率：1=每次；3=每3次；0=仅首轮

def _gen_sid() -> str:
    suf = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"sess_{int(time())}_{suf}"

def _resample_even(points: List[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    if len(points) <= 2 or step <= 0: 
        return points
    # 逐段累计长度，按等距插值
    segs, total = [], 0.0
    for i in range(len(points)-1):
        dx = points[i+1][0]-points[i][0]
        dy = points[i+1][1]-points[i][1]
        d = math.hypot(dx, dy)
        segs.append(d); total += d
    if total == 0: 
        return [points[0], points[-1]]
    out = [points[0]]
    dist, acc, i = step, 0.0, 0
    while dist < total and i < len(segs):
        seg = segs[i]
        if acc + seg >= dist:
            t = (dist - acc) / seg
            x = points[i][0] + t * (points[i+1][0] - points[i][0])
            y = points[i][1] + t * (points[i+1][1] - points[i][1])
            out.append((x, y))
            dist += step
        else:
            acc += seg; i += 1
    out.append(points[-1])
    return out

def _quantize(points: List[Tuple[float, float]], q: float) -> List[Tuple[float, float]]:
    if q <= 0: return points
    return [(round(x / q) * q, round(y / q) * q) for x, y in points]

def _r3(v: float) -> float:
    try:
        return round(float(v), 3)
    except Exception:
        return float(v)

def _merge_collinear_slopes(points: List[Tuple[float,float]], slope_eps: float = 0.01) -> List[Tuple[float,float]]:
    """与你前端一致的斜率法：连续三点 a,b,c，若共线则删 b。"""
    n = len(points)
    if n <= 2: return points
    out: List[Tuple[float,float]] = [points[0]]
    for i in range(1, n - 1):
        ax, ay = out[-1]
        bx, by = points[i]
        cx, cy = points[i+1]
        dx1, dy1 = (bx - ax), (by - ay)
        dx2, dy2 = (cx - bx), (cy - by)
        s1 = None if dx1 == 0 else dy1 / dx1
        s2 = None if dx2 == 0 else dy2 / dx2
        collinear = False
        if s1 is None and s2 is None:
            collinear = True
        elif (s1 is None) != (s2 is None):
            collinear = False
        else:
            collinear = abs(s1 - s2) < slope_eps
        if not collinear:
            out.append((bx, by))
    out.append(points[-1])
    return out

def _minify_stroke(stk: dict) -> dict:
    """下采样+量化，仅保留必要字段，适合发给 LLM。"""
    out = {
        "id": str(stk.get("id", "h")),
        "tool": str(stk.get("tool", "pen")),
        "style": stk.get("style") or None,
        "meta": stk.get("meta") or None,
    }
    pts = stk.get("points") or []
    # 取 x,y 两列，忽略 t/pressure
    xy = [(_r3(p[0]), _r3(p[1])) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    tool = out["tool"]
    if tool == "poly":
        # poly：保留顶点，不做等距重采样；仅做共线删点与量化
        if len(xy) >= 3:
            # 若首尾重复，先去重；再做共线合并（严格，避免吃掉拐角）
            if xy[0] == xy[-1] and len(xy) > 3:
                xy = xy[:-1]
            xy = _merge_collinear_slopes(xy, 0.0)
        xy = _quantize(xy, QUANT)
        out["points"] = [[_r3(x), _r3(y)] for x, y in xy]  # 统一三位小数
        return out
    if tool == "ellipse":
        # ellipse：始终保留两端点（包围盒对角），不做等距重采样
        if len(xy) >= 2:
            x0 = min(xy[0][0], xy[-1][0]); y0 = min(xy[0][1], xy[-1][1])
            x1 = max(xy[0][0], xy[-1][0]); y1 = max(xy[0][1], xy[-1][1])
            xy = [(x0, y0), (x1, y1)]
        elif len(xy) == 1:
            xy = [(xy[0][0], xy[0][1]), (xy[0][0] + 0.001, xy[0][1])]
        xy = _quantize(xy, QUANT)
        out["points"] = [[_r3(x), _r3(y)] for x, y in xy]
        return out
    # 其他工具：先共线删点 →（必要时）等距重采样 → 再次合并 → 量化
    if len(xy) >= 3:
        xy = _merge_collinear_slopes(xy, 0.01)
    if len(xy) > 2 and RESAMPLE_STEP > 0:
        xy = _resample_even(xy, RESAMPLE_STEP)
        if len(xy) >= 3:
            xy = _merge_collinear_slopes(xy, 0.01)
    xy = _quantize(xy, QUANT)
    out["points"] = [[_r3(x), _r3(y)] for x, y in xy]
    return out

def _points_equal(a, b, eps=1e-6):
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps

def _r3(v: float) -> float:
    try:
        return round(float(v), 3)
    except Exception:
        return float(v)


def _ensure_poly_closed(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """若最后一点不等于第一点则补齐闭合；全部坐标统一到三位小数。"""
    if not points:
        return points
    pts = [(_r3(x), _r3(y)) for x, y in points]
    if not _points_equal(pts[0], pts[-1]):
        pts.append(pts[0])
    return pts

def _ensure_line_valid(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """保证直线的两个端点不相同；若相同则轻微扰动第二点避免零长度。"""
    if not points:
        return points
    pts = [(_r3(x), _r3(y)) for x, y in points]
    # 仅保留首尾两个点（line 语义）
    if len(pts) >= 2:
        p0, p1 = pts[0], pts[-1]
    else:
        p0 = pts[0]; p1 = pts[0]
    if _points_equal(p0, p1):
        # 加一个非常小的扰动，方向取 (1,0)
        p1 = (_r3(p1[0] + 0.001), _r3(p1[1]))
    return [p0, p1]

@dataclass
class Session:
    sid: str
    mode: str = "light_helper"
    init_goal: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    call_count: int = 0
    # 最近笔画（已“最小化”）
    strokes: List[dict] = field(default_factory=list)

    def append_strokes(self, new_strokes: List[dict]) -> None:
        if not new_strokes: 
            return
        compact = [_minify_stroke(s) for s in new_strokes]
        self.strokes.extend(compact)
        # 保持上限
        over = max(0, len(self.strokes) - MAX_STROKES)
        if over > 0:
            self.strokes = self.strokes[over:]

    def recent_for_model(self) -> List[dict]:
        """
        返回给 LLM 的“最近窗口”笔画（覆盖 KEEP_RECENT），并做少量一致性兜底：
        - poly: 保证闭合（首尾点一致）
        - line: 保证非零长度（首尾点不完全相同）
        - 其它：维持 append/replace 时已做过的最小化与三位小数
        """
        k = min(KEEP_RECENT, len(self.strokes))
        if k <= 0:
            return []

        window = self.strokes[-k:]
        out: List[dict] = []
        for s in window:
            tool = str((s.get("tool") or "pen")).lower()
            pts = s.get("points") or []
            # 保障类型 & 三位小数
            pts2 = [(_r3(float(p[0])), _r3(float(p[1]))) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]

            if tool == "poly":
                pts2 = _ensure_poly_closed(pts2)
            elif tool == "line":
                pts2 = _ensure_line_valid(pts2)
            elif tool == "ellipse":
                if len(pts2) >= 2:
                    x0 = min(pts2[0][0], pts2[-1][0]); y0 = min(pts2[0][1], pts2[-1][1])
                    x1 = max(pts2[0][0], pts2[-1][0]); y1 = max(pts2[0][1], pts2[-1][1])
                    pts2 = [(x0, y0), (x1, y1)]
                elif len(pts2) == 1:
                    pts2 = [pts2[0], (_r3(pts2[0][0] + 0.001), pts2[0][1])]
            else:
                # 其它工具已在 _minify_stroke 中三位小数，这里再兜底一次
                pts2 = [(_r3(x), _r3(y)) for x, y in pts2]

            out.append({
                "id": str(s.get("id")),
                "tool": tool,
                "style": s.get("style") or None,
                "meta": s.get("meta") or None,
                "points": [[x, y] for x, y in pts2],
            })
        return out


    def bump(self) -> int:
        self.call_count += 1
        return self.call_count

    @staticmethod
    def _r3(v):
        try:
            return round(float(v), 3)
        except Exception:
            return float(v)

    @classmethod
    def _minify_stroke(cls, s: dict) -> dict:
        """规整/轻量化：仅 [x,y] 两列，并把坐标统一 round(.,3)"""
        pts = s.get("points") or []
        slim = []
        for p in pts:
            # 允许 [x,y] 或 [x,y,t,pressure]，只取前两列，并做 3 位小数
            x = cls._r3(p[0])
            y = cls._r3(p[1])
            slim.append([x, y])
        style = (s.get("style") or {})
        return {
            "id": s.get("id"),
            "tool": s.get("tool", "pen"),
            "points": slim,
            "style": {
                "size": style.get("size", "m"),
                "color": style.get("color", "black"),
                "opacity": float(style.get("opacity") or 1.0),
            },
            "meta": s.get("meta") or {},
        }

    def replace_strokes(self, strokes: list[dict]) -> None:
        """
        覆盖式同步：把前端“当前仍存在的所有笔画”作为快照替换到会话中。
        用于对齐擦除/撤销/重做后的真实画布。
        """
        mini = [self._minify_stroke(s) for s in (strokes or [])]
        # 可按容量限制裁剪（例如最近 500 条），避免过大
        self.strokes = mini[-500:]
# 简单内存会话表（开发环境足够；需要可换成 Redis）
_SESS: Dict[str, Session] = {}

def create_session(mode: str = "light_helper", init_goal: Optional[str] = None, tags: Optional[List[str]] = None) -> Session:
    s = Session(sid=_gen_sid(), mode=mode or "light_helper", init_goal=init_goal, tags=tags or [])
    _SESS[s.sid] = s
    return s

def get_session(sid: str) -> Optional[Session]:
    return _SESS.get(sid)

def should_include_sample(call_count: int) -> bool:
    # 0 → 仅首轮；1 → 每轮；N → 每 N 轮
    if SAMPLE_EVERY_N <= 0:
        return call_count <= 1
    if SAMPLE_EVERY_N == 1:
        return True
    return (call_count % SAMPLE_EVERY_N) == 1
