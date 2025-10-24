import React, { useCallback, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Group, Line as KLine, Rect as KRect, Ellipse as KEllipse, Text as KText } from 'react-konva'

/**
 * LineArt (Konva) — MIT-friendly board with:
 * - TopBar: grid/snap, brush size/color, import/export
 * - AI v1.1: validate / normalize / plan drafts
 * - AI preview / accept / dismiss
 * - Freehand pen tool: mouse draw → Chaikin smooth → even resample → stack
 * - Vector eraser: radius-based masking → split strokes → replace & stack sync
 * - Undo/Redo: minimal history snapshots for shapes & drawStack
 * - Export human strokes (AI v1.1)
 *
 * 注意：
 * 1) 若报“应为表达式”，通常是某段函数/对象没正确闭合的 } 或 )；
 *    这份完整替换版已通过语法检查，可直接覆盖本地文件。
 * 2) 画笔事件绑定在 <Stage> 上；预览/正式形状分别渲染在不同 Layer。
 */

/* ------------------------------
 * 0) Types (AI protocol v1.1)
 * ------------------------------ */

export type ColorName =
  | 'black' | 'blue' | 'green' | 'grey'
  | 'light-blue' | 'light-green' | 'light-red' | 'light-violet'
  | 'orange' | 'red' | 'violet' | 'white' | 'yellow'

export type AIStrokeV11 = {
  id: string
  tool: 'pen' | 'line' | 'bezier' | 'rect' | 'ellipse' | 'poly' | 'eraser' | string
  // points: [x, y, t?, pressure?]
  points: Array<[number, number, number?, number?]>
  style?: { size?: 's'|'m'|'l'|'xl'; color?: ColorName; opacity?: number }
  meta?: Record<string, any>
}

type PromptMode = "full" | "light" | "vision";

export type AIStrokePayload = {
  canvas?: { width?: number; height?: number; viewport?: [number, number, number, number] }
  strokes: AIStrokeV11[]
  intent?: 'complete' | 'hint' | 'alt'
  replace?: string[]
  version?: number
}

/* ----------------------------------------------
 * 1) Validation & normalization (no external deps)
 * ---------------------------------------------- */

const COLORS: ColorName[] = [
  'black','blue','green','grey','light-blue','light-green','light-red','light-violet','orange','red','violet','white','yellow'
]

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))
// ===== Zoom config =====
const ZOOM_MIN = 0.2
const ZOOM_MAX = 8
const ZOOM_STEP = 1.06 // 每次滚轮的缩放倍率（>1 放大，<1 缩小）

export function validateAIStrokePayload(raw: unknown): { ok: boolean; errors: string[]; payload?: AIStrokePayload } {
  const errors: string[] = []
  if (typeof raw !== 'object' || raw === null) return { ok: false, errors: ['Payload must be an object'] }
  const obj = raw as any
  if (!Array.isArray(obj.strokes)) errors.push('strokes must be an array')

  const safe: AIStrokePayload = {
    version: typeof obj.version === 'number' ? obj.version : 1,
    strokes: Array.isArray(obj.strokes) ? obj.strokes : [],
    intent: obj.intent,
    replace: obj.replace,
    canvas: obj.canvas,
  }

  if (Array.isArray(obj.strokes)) {
    obj.strokes.forEach((s: any, i: number) => {
      if (!s || typeof s !== 'object') errors.push(`strokes[${i}] not an object`)
      if (!s?.id || typeof s.id !== 'string') errors.push(`strokes[${i}].id missing`)
      if (!s?.tool || typeof s.tool !== 'string') errors.push(`strokes[${i}].tool missing`)
      if (!Array.isArray(s?.points) || s.points.length < 2) errors.push(`strokes[${i}].points need >=2 points`)
    })
  }

  return { ok: errors.length === 0, errors, payload: errors.length === 0 ? safe : undefined }
}

function clampPoints(points: AIStrokeV11['points'], viewport?: [number, number, number, number]) {
  if (!viewport) return points
  const [vx, vy, vw, vh] = viewport
  const x0 = vx, y0 = vy, x1 = vx + vw, y1 = vy + vh
  return points.map(([x, y, t, p]) => [clamp(x, x0, x1), clamp(y, y0, y1), t, p])
}

function dedupeConsecutive(points: AIStrokeV11['points']) {
  const out: typeof points = []
  let lx = NaN, ly = NaN
  for (const [x,y,t,p] of points) {
    if (x !== lx || y !== ly) {
      out.push([x,y,t,p])
      lx = x; ly = y
    }
  }
  return out
}

// Ramer–Douglas–Peucker 简化：减少多余点（提升性能）
function simplifyRDP(points: AIStrokeV11['points'], eps = 0.4) {
  if (points.length <= 2) return points
  const keep = new Array(points.length).fill(false)
  keep[0] = keep[points.length-1] = true
  const stack: [number,number][] = [[0, points.length-1]]

  const dist = (p:[number,number], a:[number,number], b:[number,number]) => {
    const [x,y]=p,[x1,y1]=a,[x2,y2]=b
    const A=x-x1,B=y-y1,C=x2-x1,D=y2-y1
    const dot=A*C+B*D, len=C*C+D*D
    const t=len? Math.max(0,Math.min(1,dot/len)) : 0
    const px=x1+t*C, py=y1+t*D
    return Math.hypot(x-px, y-py)
  }

  while (stack.length) {
    const [i,j]=stack.pop()!
    let idx=-1, maxd=-1
    for (let k=i+1; k<j; k++) {
      const d = dist([points[k][0],points[k][1]],[points[i][0],points[i][1]],[points[j][0],points[j][1]])
      if (d>maxd) { maxd=d; idx=k }
    }
    if (maxd>eps && idx!==-1) { keep[idx]=true; stack.push([i,idx],[idx,j]) }
  }
  return points.filter((_,i)=>keep[i])
}

function normalizeStroke(s: AIStrokeV11, viewport?: [number,number,number,number]): AIStrokeV11 {
  const size = (s.style?.size as 's'|'m'|'l'|'xl') ?? 'm'
  const color = COLORS.includes((s.style?.color as ColorName)) ? (s.style!.color as ColorName) : 'black'
  const opacity = typeof s.style?.opacity==='number' ? clamp(s.style!.opacity!,0,1) : 1
  // 基础清洗
  const raw = dedupeConsecutive(clampPoints(s.points, viewport))
  // pen 的闭合曲线要避免被 RDP 简化成“首末相同的两点”
  const closed = (s.tool !== 'poly') && isClosedStroke(raw)
  // poly 关闭简化；pen-closed 也关闭简化；其余轻度简化
  const eps = (s.tool === 'poly' || closed) ? 0 : 0.4
  let pts = eps === 0 ? raw : simplifyRDP(raw, eps)
  // 兜底：若简化后只剩 ≤2 点且首末重合，则回退原点列
  if (pts.length <= 2) {
    const [x0, y0] = pts[0] ?? [NaN, NaN]
    const [xn, yn] = pts[pts.length - 1] ?? [NaN, NaN]
    if (Number.isFinite(x0) && Math.hypot((xn ?? x0) - x0, (yn ?? y0) - y0) <= 1e-6) {
      pts = raw
    }
  }
  return { id: String(s.id), tool: String(s.tool), points: pts, style: { size, color, opacity }, meta: s.meta ?? {} }
}

function computeBounds(points: AIStrokeV11['points']) {
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity
  for (const [x,y] of points) { if (x<minX) minX=x; if (y<minY) minY=y; if (x>maxX) maxX=x; if (y>maxY) maxY=y }
  return { x:minX, y:minY, w: Math.max(1,maxX-minX), h: Math.max(1,maxY-minY) }
}

// 闭合判定：首尾距离在容差内视为闭合
function isClosedStroke(points: AIStrokeV11['points'], tol = 1.5) {
  if (!points || points.length < 3) return false
  const [x0, y0] = points[0]
  const [xn, yn] = points[points.length - 1]
  return Math.hypot(xn - x0, yn - y0) <= tol
}

export function normalizeAIStrokePayload(raw: AIStrokePayload) {
  const viewport = raw.canvas?.viewport
  const strokes = raw.strokes.map(s=>normalizeStroke(s, viewport))
  const payloadId = `ai_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`
  return { payloadId, strokes, intent: raw.intent, replace: raw.replace, viewport }
}


/* ------------------------------
 * 2) Draft planning (engine-agnostic)
 * ------------------------------ */

export type ShapeDraft = {
  id: string
  kind: 'pen'|'line'|'rect'|'ellipse'|'poly'|'polyline'|'erase'
  x: number
  y: number
  w?: number
  h?: number
  points?: Array<{ x:number; y:number; pressure?: number }>
  style?: { size:'s'|'m'|'l'|'xl'; color: ColorName; opacity:number }
  meta?: Record<string,any>
}

export function planDrafts(norm: ReturnType<typeof normalizeAIStrokePayload>): ShapeDraft[] {
  const drafts: ShapeDraft[] = []
  for (const s of norm.strokes) {
    const { x, y, w, h } = computeBounds(s.points)
    const style = {
      size: (s.style?.size ?? 'm') as 's'|'m'|'l'|'xl',
      color: (s.style?.color ?? 'black') as ColorName,
      opacity: s.style?.opacity ?? 1,
    }
    switch (s.tool) {
      case 'pen':
      case 'bezier':
        drafts.push({
          id: s.id, kind: 'pen', x, y,
          points: s.points.map(([px,py,,pr])=>({ x: px - x, y: py - y, pressure: pr })),
          style, meta: { ...(s.meta ?? {}), curve: true },
        })
        break
      case 'line':
        drafts.push({
          id: s.id, kind: 'line', x, y,
          points: s.points.map(([px,py])=>({ x: px - x, y: py - y })),
          style, meta: { ...s.meta },
        })
        break
      case 'rect':
        drafts.push({ id: s.id, kind: 'rect', x, y, w, h, style, meta: { ...s.meta } })
        break
      case 'ellipse':
        drafts.push({ id: s.id, kind: 'ellipse', x, y, w, h, style, meta: { ...s.meta } })
        break
      case 'poly':
        drafts.push({
          id: s.id, kind: 'poly', x, y, w, h,
          points: s.points.map(([px,py])=>({ x: px - x, y: py - y })),
          style, meta: { ...s.meta },
        })
        break
      case 'eraser':
        drafts.push({ id: s.id, kind: 'erase', x, y, style, meta: { ...s.meta, op: 'erase' } })
        break
      default:
        drafts.push({
          id: s.id, kind: 'polyline', x, y, w, h,
          points: s.points.map(([px,py])=>({ x: px - x, y: py - y })),
          style, meta: { ...(s.meta ?? {}), curve: true },
        })
    }
  }
  return drafts
}

/* ---------------------------------------
 * 3) Konva board impl + Drawing Tool
 * --------------------------------------- */

// 映射画笔粗细到 strokeWidth，可按需调整
const SIZE_TO_WIDTH: Record<'s'|'m'|'l'|'xl', number> = { s: 2, m: 4, l: 6, xl: 10 }

// 颜色映射（可替换成主题系统）
const colorToStroke = (c: ColorName) => {
  switch (c) {
    case 'grey': return '#888'
    case 'light-blue': return '#7db3ff'
    case 'light-green': return '#6fd37f'
    case 'light-red': return '#ff7d7d'
    case 'light-violet': return '#b38bff'
    case 'orange': return '#ff9a3b'
    case 'violet': return '#7a5cff'
    case 'yellow': return '#ffeb3b'
    default: return c // 'black'|'blue'|'green'|'red'|'white'
  }
}

// Chaikin 平滑：用“角切”获得接近二次贝塞尔的平滑效果
function chaikin(points: Array<[number, number]>, iterations = 2): Array<[number, number]> {
  if (points.length <= 2) return points
  let pts = points
  for (let it = 0; it < iterations; it++) {
    const next: Array<[number, number]> = []
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i]
      const [x1, y1] = pts[i + 1]
      next.push([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1])
      next.push([0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1])
    }
    pts = [pts[0], ...next, pts[pts.length - 1]]
  }
  return pts
}

// 均匀采样：按固定步长生成等间距点列，便于擦除/重采样
function resampleEvenly(points: Array<[number, number]>, step = 3): Array<[number, number]> {
  if (points.length <= 2) return points
  const segs: number[] = []
  let total = 0
  for (let i = 0; i < points.length - 1; i++) {
    const dx = points[i + 1][0] - points[i][0]
    const dy = points[i + 1][1] - points[i][1]
    const d = Math.hypot(dx, dy)
    segs.push(d); total += d
  }
  if (total === 0) return [points[0], points[points.length - 1]]
  const out: Array<[number, number]> = [points[0]]
  let dist = step
  let acc = 0
  let i = 0
  while (dist < total && i < segs.length) {
    const segLen = segs[i]
    if (acc + segLen >= dist) {
      const t = (dist - acc) / segLen
      const x = points[i][0] + t * (points[i + 1][0] - points[i][0])
      const y = points[i][1] + t * (points[i + 1][1] - points[i][1])
      out.push([x, y]); dist += step
    } else {
      acc += segLen; i++
    }
  }
  out.push(points[points.length - 1])
  return out
}

// ===== Geometry helpers moved to module scope (avoid TDZ in hooks) =====
// 1) 最大偏差：判断“近似直线”
function geomMaxDeviationFromChord(pts: Array<[number, number]>) {
  if (pts.length <= 2) return 0
  const [x1, y1] = pts[0], [x2, y2] = pts[pts.length - 1]
  const Cx = x2 - x1, Cy = y2 - y1
  const L2 = Cx * Cx + Cy * Cy || 1e-9
  let maxd = 0
  for (let i = 1; i < pts.length - 1; i++) {
    const [x, y] = pts[i]
    const vx = x - x1, vy = y - y1
    const cross = Math.abs(vx * Cy - vy * Cx)
    const d = cross / Math.sqrt(L2)
    if (d > maxd) maxd = d
  }
  return maxd
}

// 2) Ramer–Douglas–Peucker 关键点
function geomRdpKeypoints(pts: Array<[number, number]>, eps: number) {
  if (pts.length <= 2) return pts
  const keep = new Array(pts.length).fill(false)
  keep[0] = keep[pts.length - 1] = true
  const stack: [number, number][] = [[0, pts.length - 1]]
  const dist = (p: [number, number], a: [number, number], b: [number, number]) => {
    const [x, y] = p, [x1, y1] = a, [x2, y2] = b
    const A = x - x1, B = y - y1, C = x2 - x1, D = y2 - y1
    const dot = A * C + B * D, len = C * C + D * D
    const t = len ? Math.max(0, Math.min(1, dot / len)) : 0
    const px = x1 + t * C, py = y1 + t * D
    return Math.hypot(x - px, y - py)
  }
  while (stack.length) {
    const [i, j] = stack.pop()!
    let idx = -1, maxd = -1
    for (let k = i + 1; k < j; k++) {
      const d = dist(pts[k], pts[i], pts[j])
      if (d > maxd) { maxd = d; idx = k }
    }
    if (maxd > eps && idx !== -1) { keep[idx] = true; stack.push([i, idx], [idx, j]) }
  }
  return pts.filter((_, i) => keep[i])
}

// 3) 限制关键点数量（按弧长等距抽样到 N）
function geomLimitPoints(pts: Array<[number, number]>, maxN: number) {
  if (pts.length <= maxN) return pts
  const stepCount = maxN - 1
  const segs: number[] = []; let total = 0
  for (let i = 0; i < pts.length - 1; i++) {
    const d = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]); segs.push(d); total += d
  }
  if (total <= 1e-6) return [pts[0], pts[pts.length - 1]]
  const out: [number, number][] = [pts[0]]
  for (let s = 1; s < stepCount; s++) {
    const target = (total * s) / stepCount
    let acc = 0, i = 0
    while (i < segs.length && acc + segs[i] < target) { acc += segs[i]; i++ }
    if (i >= segs.length) { out.push(pts[pts.length - 1]); break }
    const t = (target - acc) / (segs[i] || 1e-9)
    const x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    const y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
    out.push([x, y])
  }
  out.push(pts[pts.length - 1])
  return out
}

// 简洁版：用斜率判定共线。如果连续三点 a,b,c 斜率差<0.01则删 b。
// - 只有一个斜率不存在 → 不共线；
// - 都不存在（垂直线）→ 共线；
// - 都存在且差值<0.01 → 共线；
function mergeCollinear(pts: Array<[number, number]>, slopeEps = 0.01): Array<[number, number]> {
  if (pts.length <= 2) return pts;

  const out: Array<[number, number]> = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = out[out.length - 1];
    const b = pts[i];
    const c = pts[i + 1];

    const dx1 = b[0] - a[0];
    const dy1 = b[1] - a[1];
    const dx2 = c[0] - b[0];
    const dy2 = c[1] - b[1];

    const slope1 = dx1 === 0 ? null : dy1 / dx1;
    const slope2 = dx2 === 0 ? null : dy2 / dx2;

    let collinear = false;
    if (slope1 === null && slope2 === null) collinear = true;
    else if (slope1 === null || slope2 === null) collinear = false;
    else if (Math.abs(slope1 - slope2) < slopeEps) collinear = true;

    if (!collinear) out.push(b);
  }
  out.push(pts[pts.length - 1]);
  return out;
}

// —— 公用：直线偏差检测（最大偏离首尾连线的距离）——
function maxDeviationToLine(pts: Array<[number, number]>): number {
  if (!Array.isArray(pts) || pts.length <= 2) return 0
  const [x1, y1] = pts[0], [x2, y2] = pts[pts.length - 1]
  const Cx = x2 - x1, Cy = y2 - y1
  const L2 = Cx * Cx + Cy * Cy
  if (L2 <= 1e-9) return 0
  let maxd = 0
  for (let k = 1; k < pts.length - 1; k++) {
    const [x, y] = pts[k]
    const vx = x - x1, vy = y - y1
    const cross = Math.abs(vx * Cy - vy * Cx)
    const d = cross / Math.sqrt(L2)
    if (d > maxd) maxd = d
  }
  return maxd
}

// 将绝对坐标点列转换为 AI 协议笔画 + Draft（局部坐标），带“直线识别”
function buildAIStrokeAndDraft(
  absPoints: Array<[number, number]>,
  style: { size: 's'|'m'|'l'|'xl'; color: ColorName; opacity: number },
  meta?: Record<string, any>
) {
  // 先执行“连续三点同斜率”化简，保证存栈即瘦身
  absPoints = mergeCollinear(absPoints, 0.01)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of absPoints) { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y }

  // —— 直线识别阈值（像素）：与后端保持一致（约 0.8px）
  const isLine = maxDeviationToLine(absPoints) < 0.8
  const idPrefix = isLine ? 'line' : 'pen'
  const id = `${idPrefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`

  if (isLine) {
    // 仅保留首尾两点；Draft.kind = 'line'
    const p0 = absPoints[0], pn = absPoints[absPoints.length - 1]
    const local2 = [
      { x: p0[0] - minX, y: p0[1] - minY },
      { x: pn[0] - minX, y: pn[1] - minY },
    ]
    const aiStroke: AIStrokeV11 = {
      id,
      tool: 'line',
      points: [p0, pn],
      style,
      meta: { author: 'human', ...(meta ?? {}) },
    }
    const draft: ShapeDraft = {
      id, kind: 'line', x: minX, y: minY, points: local2, style, meta,
    }
    return { aiStroke, draft }
  } else {
    // 普通手绘多段：仍按 pen 输出全部点
    const local = absPoints.map(([x, y]) => ({ x: x - minX, y: y - minY }))
    const aiStroke: AIStrokeV11 = {
      id,
      tool: 'pen',
      points: absPoints.map(([x, y]) => [x, y]),
      style,
      meta: { author: 'human', ...(meta ?? {}) },
    }
    const draft: ShapeDraft = {
      id, kind: 'pen', x: minX, y: minY, points: local, style, meta,
    }
    return { aiStroke, draft }
  }
}

// 将 Draft（局部坐标）还原为 AI 协议笔画（绝对坐标），支持 line/pen
 function draftToAIStroke(d: ShapeDraft): AIStrokeV11 | null {
   // —— line：需要两端点（来自 points）
   if (d.kind === 'line') {
     if (!d.points || d.points.length < 2) return null
     const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    // 保障仅两端点
    const p0 = pointsAbs[0], pn = pointsAbs[pointsAbs.length - 1]
    return {
      id: d.id,
      tool: 'line',
      points: [p0, pn],
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) }
    }
  }
  // —— pen / polyline：按点列还原
  if (d.kind === 'pen' || d.kind === 'polyline') {
    if (!d.points || d.points.length < 2) return null
    const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    return {
      id: d.id,
      tool: 'pen',
      points: pointsAbs,
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) }
    }
  }
  // —— poly：按顶点还原
  if (d.kind === 'poly') {
    if (!d.points || d.points.length < 2) return null
    const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    return {
      id: d.id,
      tool: 'poly',
      points: pointsAbs,
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) }
    }
  }
  // —— ellipse：由 x,y,w,h 还原为“包围盒对角”两点；无需 d.points
  if (d.kind === 'ellipse') {
    const x0 = d.x
    const y0 = d.y
    const x1 = d.x + (d.w ?? 0)
    const y1 = d.y + (d.h ?? 0)
    // 极小尺寸保护：避免零长度
    if (Math.abs(x1 - x0) < 1e-9 && Math.abs(y1 - y0) < 1e-9) return null
    const p0: [number, number] = [Math.min(x0, x1), Math.min(y0, y1)]
    const p1: [number, number] = [Math.max(x0, x1), Math.max(y0, y1)]
    return {
      id: d.id,
      tool: 'ellipse',
      points: [p0, p1],
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) }
    }
  }
  return null
}

// 预览项结构：一个 payloadId 对应一组草案
type PreviewEntry = { payloadId: string; drafts: ShapeDraft[] }

export default function LineArtBoard() {
  // 画布尺寸（可改成 ResizeObserver）
  const [size] = useState({ width: window.innerWidth, height: window.innerHeight })
  const askAIRef = useRef<null | (() => void)>(null)
  const stageRef = useRef<any>(null)
  // 顶部 UI 状态
  const [showGrid, setShowGrid] = useState(true)
  const [snap, setSnap] = useState(true)
  // 新增：控制“拐点是否在松手后转为曲线”
  const [curveTurns, setCurveTurns] = useState(true)

  // 已接受（正式）形状
  const [shapes, setShapes] = useState<ShapeDraft[]>([])

  // 问答模式
  const [mode, setMode] = useState<PromptMode>("full");

  // 画笔配置（与 AI 协议中的 style 对齐）
  const [brushSize, setBrushSize] = useState<'s'|'m'|'l'|'xl'>('m')
  const [brushColor, setBrushColor] = useState<ColorName>('black')
  const currentBrush = useMemo(() => ({
    tool: 'pen' as const,
    style: { size: brushSize, color: brushColor as ColorName, opacity: 1 },
    meta: { author: 'human' } as Record<string, any>,
  }), [brushSize, brushColor])
  // 提示词（用于后端 /suggest 的 hint）
  const [hint, setHint] = useState<string>('Try to help the user draw line art.')
  // AI 生成规模（约束模型返回的点数上限；也影响我们本地上传的人类关键点密度）
  const [aiScale, setAiScale] = useState<number>(16) // 4~64 之间可调，默认 16
  // 绘制中的状态与“原始浮点坐标”（绝对像素，不做吸附/取整）
  const [isDrawing, setIsDrawing] = useState(false)
  const [rawPoints, setRawPoints] = useState<number[]>([])  // [x0,y0,x1,y1,...] (float)

  // 人类笔画栈（供擦除/撤销使用）
  type DrawStackEntry = { ai: AIStrokeV11; draft: ShapeDraft }
  const [drawStack, setDrawStack] = useState<DrawStackEntry[]>([])
  // -------- 工具模式：'pen' 或 'eraser' --------
  const [toolMode, setToolMode] = useState<'pen' | 'eraser' | 'ellipse' | 'hand'>('pen')
  const [eraserRadius, setEraserRadius] = useState<number>(14) // 像素
  const [boxDraft, setBoxDraft] = useState<ShapeDraft | null>(null)
  // 橡皮光标圆位置（仅用于可视化半径）
  const [eraserCursor, setEraserCursor] = useState<{x:number;y:number}|null>(null)
  // 单次擦除手势（按下->抬起）只 pushHistory 一次
  const eraseGestureStarted = useRef(false)
   // NEW: 画布视图（无限画板）的位置
  const [view, setView] = useState<{x:number; y:number; scale:number}>({ x: 0, y: 0, scale: 1 })
  const [isPanning, setIsPanning] = useState(false)
  // 滚轮缩放：以鼠标所在位置为缩放中心
  const onWheelZoom = useCallback((e: any) => {
    // Konva 转发的原生事件
    const evt: WheelEvent = e?.evt
    if (!evt) return
    // 阻止页面滚动和浏览器缩放
    evt.preventDefault()
    // 可选：按住 Ctrl 时不处理，交给浏览器（如果你不想这样，删掉这两行）
    // if (evt.ctrlKey) return

    const stage = e.target.getStage?.()
    const ptr = stage?.getPointerPosition?.()
    if (!ptr) return

    // 当前缩放与位置
    const oldScale = view.scale
    // 计算缩放方向：deltaY>0 缩小，<0 放大
    const direction = evt.deltaY > 0 ? -1 : 1
    const scaleBy = direction > 0 ? ZOOM_STEP : 1 / ZOOM_STEP
    let newScale = oldScale * scaleBy
    newScale = clamp(newScale, ZOOM_MIN, ZOOM_MAX)

    // 鼠标点对应的世界坐标（缩放前）
    const worldX = (ptr.x - view.x) / oldScale
    const worldY = (ptr.y - view.y) / oldScale

    // 调整视图位置，使缩放后鼠标仍指向同一世界坐标
    const newX = ptr.x - worldX * newScale
    const newY = ptr.y - worldY * newScale

    setView(v => ({ ...v, x: newX, y: newY, scale: newScale }))
  }, [view])

  // 屏幕坐标(鼠标) → 世界坐标(数据/几何计算)
  const screenToWorld = useCallback((sx:number, sy:number) => {
    return { x: (sx - view.x) / view.scale, y: (sy - view.y) / view.scale }
  }, [view])

  // ===== Canvas → JPEG Base64（最长边≤1024）=====
  const snapshotCanvas = useCallback(async (
    maxSize = 768,
    mime: "image/jpeg" | "image/png" = "image/jpeg",
    quality = 0.6
  ): Promise<{ data: string|null; w: number; h: number; mime: string }> => {
    const stage = stageRef.current
    if (!stage) return { data: null, w: 0, h: 0, mime }
    // 原始 dataURL（Konva 会把舞台渲染到一个 <canvas>）
    const srcUri: string = stage.toDataURL({ pixelRatio: 1 })
    // 用离屏 <canvas> 做等比缩放 + 压缩
    const img = new Image()
    img.src = srcUri
    await new Promise<void>(r => { img.onload = () => r() })
    let { width: w, height: h } = img
    const scale = Math.min(1, maxSize / Math.max(w, h))
    const tw = Math.round(w * scale), th = Math.round(h * scale)
    const cvs = document.createElement("canvas")
    cvs.width = tw; cvs.height = th
    const ctx = cvs.getContext("2d")!
    ctx.drawImage(img, 0, 0, tw, th)
    const outUri = cvs.toDataURL(mime, quality)
    const base64 = outUri.split(",")[1] || null
    return { data: base64, w: tw, h: th, mime }
  }, [])

  // 预览寄存：payloadId -> drafts
  const [previews, setPreviews] = useState<Record<string, PreviewEntry>>({})
  const [currentPayloadId, setCurrentPayloadId] = useState<string | null>(null)
  // AI Feed：记录每次 AI 返回包中的 id/desc（最近 50 条）
  const [aiFeed, setAiFeed] = useState<Array<{payloadId:string; time:number; items:{id:string; desc?:string}[] }>>([])
  // 会话：后端的 sid；lastSentIndex 仅用于计算 delta
  const [sid, setSid] = useState<string | null>(null)
  const [visionVersion, setVisionVersion] = useState<number>(2.0)
  const lastSentIndexRef = useRef<number>(0)
  // 同步防抖计时器
  const syncTimerRef = useRef<number | null>(null)

  // 三位小数工具（减少传输与日志体积；后端也有兜底）
  const round3 = (v:number) => Math.round(v * 1000) / 1000

  // 将 drawStack.ai（绝对坐标）打包为 AI 协议 strokes
  const packAllStrokes = useCallback(() => {
    return drawStack.map(e => ({
      ...e.ai,
      points: e.ai.points.map(([x,y,t,p]) => [round3(x), round3(y), t, p]) as any,
    }))
  }, [drawStack])

  // -------- 简单历史栈（撤销/重做） --------
  type Snapshot = { shapes: ShapeDraft[]; drawStack: DrawStackEntry[] }
  const [past, setPast] = useState<Snapshot[]>([])
  const [future, setFuture] = useState<Snapshot[]>([])
  const pushHistory = useCallback((snap?: Snapshot) => {
    setPast(p => [...p, snap ?? { shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }])
    setFuture([]) // 新操作后清空未来分支
  }, [shapes, drawStack])
  const undo = useCallback(() => {
    setPast(p => {
      if (p.length === 0) return p
      const last = p[p.length - 1]
      setFuture(f => [{ shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }, ...f])
      setShapes(JSON.parse(JSON.stringify(last.shapes)))
      setDrawStack(JSON.parse(JSON.stringify(last.drawStack)))
      return p.slice(0, -1)
    })
  }, [shapes, drawStack])
  const redo = useCallback(() => {
    setFuture(f => {
      if (f.length === 0) return f
      const head = f[0]
      setPast(p => [...p, { shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }])
      setShapes(JSON.parse(JSON.stringify(head.shapes)))
      setDrawStack(JSON.parse(JSON.stringify(head.drawStack)))
      return f.slice(1)
    })
  }, [shapes, drawStack])

  // ===== 自动同步到后端会话：drawStack 或 sid 变化后 300ms 覆盖式提交 =====
  const syncSession = useCallback(async (curSid: string) => {
    try {
      const strokes = packAllStrokes()
      const res = await fetch('http://localhost:8000/session/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sid: curSid, strokes }),
      })
      if (!res.ok) console.warn('session/sync failed', await res.text())
    } catch (err) {
      console.warn('session/sync error', err)
    }
  }, [packAllStrokes])

  React.useEffect(() => {
    if (!sid) return
    if (syncTimerRef.current) window.clearTimeout(syncTimerRef.current)
    syncTimerRef.current = window.setTimeout(() => {
      syncSession(sid)
    }, 300) as unknown as number
    return () => {
      if (syncTimerRef.current) {
        window.clearTimeout(syncTimerRef.current)
        syncTimerRef.current = null
      }
    }
  }, [drawStack, sid, syncSession])

    // —— 自动补全开关 & 倒计时（5s） —— 
  const [autoComplete, setAutoComplete] = useState<boolean>(false)
  const [autoCountdown, setAutoCountdown] = useState<number|null>(null)
  const autoTimerRef = useRef<number | ReturnType<typeof setTimeout> | null>(null)
  const autoTickerRef = useRef<number | ReturnType<typeof setInterval> | null>(null)
  const isApplyingAIRef = useRef(false) // 防止“接受预览并入”被误判为用户操作

  const hasActivePreview = useMemo(() => {
    // 有“当前正在预览的AI笔画” → 认为存在预览
    return Object.keys(previews || {}).length > 0
  }, [previews])

  const clearAutoTimer = useCallback(() => {
    if (autoTimerRef.current) { clearTimeout(autoTimerRef.current as any); autoTimerRef.current = null }
    if (autoTickerRef.current) { clearInterval(autoTickerRef.current as any); autoTickerRef.current = null }
    setAutoCountdown(null)
  }, [])

  // 导出/导入（仅导出正式 shapes 的 JSON）
  const exportJSON = useCallback(() => {
    const blob = new Blob([JSON.stringify({ shapes }, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'lineart_konva.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }, [shapes])

  const fileRef = useRef<HTMLInputElement | null>(null)
  const importJSON = useCallback(async (file: File) => {
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      if (Array.isArray(data.shapes)) setShapes(data.shapes)
    } catch { alert('Invalid JSON') }
  }, [])

  // 导出所有“人类笔画”为 AI v1.1 的 strokes（方便喂给 /suggest）
  const exportHumanStrokesAI = useCallback(() => {
    const strokes = drawStack.map(d => d.ai)
    const payload = { version: 1, intent: 'complete', strokes }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'human_strokes_ai_v1.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }, [drawStack])

  // AI：只把 JSON 写到 localStorage，便于手动测试
  const applyAIStub = useCallback(() => {
    const raw = prompt('Paste AI suggestions JSON (v1.1)')
    if (!raw) return
    try {
      const payload = JSON.parse(raw) as AIStrokePayload
      localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload))
      alert('Saved to localStorage.ai_suggestions_v1')
    } catch { alert('Invalid JSON') }
  }, [])

  // 当“画板元素数量”发生变化时，清除尚未应用的预览
  React.useEffect(() => {
    if (Object.keys(previews).length > 0) {
      setPreviews({})
      setCurrentPayloadId(null)
    }
  }, [shapes.length])
  // 预览：localStorage -> 校验 -> 规范化 -> 生成草案 -> 存入 previews
  const previewAI = useCallback(() => {
    const raw = localStorage.getItem('ai_suggestions_v1')
    if (!raw) { alert('No ai_suggestions_v1 in localStorage'); return }
    try {
      const obj = JSON.parse(raw) as AIStrokePayload
      // 记录 feed
      const items = (obj.strokes||[]).map(s=>({ id: s.id, desc: (s.meta as any)?.desc }))
      setAiFeed(prev=>([{ payloadId: 'local_'+Date.now().toString(36), time: Date.now(), items }, ...prev].slice(0,50)))
      const v = validateAIStrokePayload(obj)
      if (!v.ok || !v.payload) { alert('Invalid payload: ' + v.errors.join('; ')); return }
      const norm = normalizeAIStrokePayload(v.payload)
      const drafts = planDrafts(norm)
      setPreviews(prev => ({ ...prev, [norm.payloadId]: { payloadId: norm.payloadId, drafts } }))
      setCurrentPayloadId(norm.payloadId)
      // 一旦进入“预览中”，就停止自动补全倒计时
      clearAutoTimer()
      alert(`Preview created: ${drafts.length} shapes\nPayloadId: ${norm.payloadId}`)
    } catch { alert('Invalid JSON in localStorage') }
  }, [])

  const noteUserAction = useCallback((opts?: { forceStart?: boolean }) => {
    const forceStart = !!opts?.forceStart
    // 仅负责自动补全倒计时的启动/重置；不在这里清预览（见补丁2）
    // 条件：开关打开，且“当前无预览”或“强制启动”
    if (autoComplete && (!hasActivePreview || forceStart)) {
      clearAutoTimer()
      setAutoCountdown(5)
      // 每秒可视倒计时
      autoTickerRef.current = setInterval(() => {
        setAutoCountdown((sec) => (sec == null ? null : Math.max(0, sec - 1)))
      }, 1000)
      // 5 秒后触发 askAI
      autoTimerRef.current = setTimeout(() => {
        clearAutoTimer()
        // 触发“发送请求”：相当于点击 Ask AI 按钮
        askAIRef.current && askAIRef.current()
      }, 5000)
    } else {
      // 其他情况（比如关开关/有预览）就确保没有倒计时
      clearAutoTimer()
    }
  }, [autoComplete, hasActivePreview, clearAutoTimer])

  // 接受：把预览草案并入正式 shapes 并移除预览
  const acceptAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    const entry = previews[currentPayloadId]
    if (!entry) { alert('Preview not found'); return }
    // 记录历史
    pushHistory()
    // 1) 合并到正式图层
    setShapes(prev => [...prev, ...entry.drafts])
    // 2) 同步写入 drawStack（关键：让橡皮/撤销对 AI 生效）
    setDrawStack(prev => {
      const toAppend = entry.drafts
        .map(d => draftToAIStroke(d))
        .filter((s): s is AIStrokeV11 => !!s)
        .map((ai) => ({ ai, draft: entry.drafts.find(dd => dd.id === ai.id)! }))
      return [...prev, ...toAppend]
    })
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    // 将“接受预览”视为一次用户操作：启动/重置自动补全倒计时
    noteUserAction({ forceStart: true })
  }, [currentPayloadId, previews, pushHistory, noteUserAction])

  // 丢弃：仅移除预览
  const dismissAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    // 如果打开自动补全：下次用户再动笔会重新进入5秒倒计时
    clearAutoTimer()
  }, [currentPayloadId])
  // ====== Ask AI：调用后端并自动预览 ======

  const askAI = useCallback(async () => {
    try {
      // 1) 会话确保
      let curSid = sid
      if (!curSid) {
        const r0 = await fetch('http://localhost:8000/session/init', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'light_helper', init_goal: hint }),
        })
        if (!r0.ok) {
          const t = await r0.text().catch(()=> '')
          throw new Error(`init failed: ${r0.status} ${r0.statusText}\n${t}`)
        }
        const j0 = await r0.json()
        curSid = j0.sid
        setSid(curSid)
        lastSentIndexRef.current = 0
      }

      // 2) 计算 delta（仅新增部分），并在发送前做一次轻量化简
      const from = lastSentIndexRef.current
      const deltaStrokes = drawStack.slice(from).map(e => {
        const s = e.ai
        const xy = (s.points || []).map(p => [p[0], p[1]] as [number, number])
        const slim = mergeCollinear(xy, 0.01)
        return { ...s, points: slim.map(([x, y]) => [x, y] as [number, number]) }
      })
      lastSentIndexRef.current = drawStack.length

      // Vision 模式：准备快照
      let image_data: string | null = null
      let image_mime: "image/jpeg" | "image/png" = "image/jpeg"
      let snapshot_size: [number, number] | undefined
      if (mode === "vision") {
        const snap = await snapshotCanvas(1024, "image/jpeg", 0.7)
        if (snap.data) {
          image_data = snap.data
          image_mime = snap.mime as any
          snapshot_size = [snap.w, snap.h]
        }
      }

      // 3) 组织请求（带上画布 viewport，方便后端裁边/验证）
      const snapshot = packAllStrokes()
      const baseReq = {
        sid: curSid!,
        canvas: { viewport: [0, 0, size.width, size.height] as [number, number, number, number] },
        delta: { strokes: deltaStrokes },
        context: { version: 1, intent: 'complete', strokes: snapshot },
        hint,
        gen_scale: aiScale,
        mode, // ✅ 关键：三种模式之一
        vision_version: visionVersion,
        ...(mode === "vision" ? {
          image_data,
          image_mime,
          snapshot_size,
        } : {})
      }

      const doPost = async (payload: any) =>
        fetch('http://localhost:8000/suggest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })

      // === Vision 2.0 双阶段 ===
      if (mode === "vision" && visionVersion >= 2) {
        // Step-1：不给点列，只做图像理解
        const req1 = { ...baseReq, seq: 1, context: { version: 1, intent: 'hint', strokes: [] } }
        let res1 = await doPost(req1)
        if (!res1.ok) {
          const t = await res1.text().catch(()=> '')
          throw new Error(`Vision 2.0 step1 failed: ${res1.status} ${res1.statusText}\n${t}`)
        }
        const data1 = await res1.json()
        const v2 = data1?.vision2 || {}
        const instruction: string = (v2.instruction || '').toString()
        // 没拿到 instruction 就直接退化为“把 hint 当指令”
        const inst = instruction || hint || 'Make the single best next stroke.'

        // Step-2：把 instruction 注入 full 流程（服务器端会按 full 处理）
        const req2 = {
          sid: curSid!,
          // 仅挑必要字段，避免把 baseReq 里的 image_data 等带过去
          canvas: { viewport: [0, 0, size.width, size.height] as [number, number, number, number] },
          delta: { strokes: deltaStrokes },
          // 仍用完整上下文（包含当前所有 strokes）
          context: { version: 1, intent: 'complete', strokes: snapshot },

          //把 Step-1 的 analysis + 最终指令 一并传给后端
          instruction_text: JSON.stringify({
            analysis: (v2.analysis || '').toString(),
            instruction: inst
          }),

          //让后端走 Vision 2.0 的 Step-2 分支（不会再注入图片）
          mode: "vision",
          vision_version: visionVersion,
          seq: 2,

          // 其他你已有的参数（例如 hint/gen_scale）也可以带上
          hint,
          gen_scale: aiScale,
        }
        let res2 = await doPost(req2)
        if (!res2.ok) {
          const t = await res2.text().catch(()=> '')
          throw new Error(`Vision 2.0 step2 failed: ${res2.status} ${res2.statusText}\n${t}`)
        }
        const data2 = await res2.json()
        if (data2?.usage?.new_sid) setSid(String(data2.usage.new_sid))
        const payload2 = data2?.payload
        if (!payload2) throw new Error('No payload in step2 response')
        localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload2))
        const items2 = (payload2.strokes || []).map((s:any) => ({ id: s.id, desc: (s.meta as any)?.desc }))
        setAiFeed(prev => ([{ payloadId: 'srv_'+Date.now().toString(36), time: Date.now(), items: items2 }, ...prev].slice(0, 50)))
        const v = validateAIStrokePayload(payload2)
        if (!v.ok || !v.payload) throw new Error('Invalid AI payload: ' + v.errors.join('; '))
        const norm = normalizeAIStrokePayload(v.payload)
        const drafts = planDrafts(norm)
        setPreviews(prev => ({ ...prev, [norm.payloadId]: { payloadId: norm.payloadId, drafts } }))
        setCurrentPayloadId(norm.payloadId)
        return
      }

      // === 旧流程（Vision 1.0 / full / light） ===
      let res = await doPost({ ...baseReq, sid: curSid! })
      if (!res.ok) {
        // 会话失效：重建后重试一次
        const txt = await res.text().catch(()=>'')
        if (res.status === 404 && /session not found/i.test(txt)) {
          const r1 = await fetch('http://localhost:8000/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'light_helper', init_goal: hint }),
          })
          if (r1.ok) {
            const j1 = await r1.json()
            curSid = j1.sid
            setSid(curSid)
            res = await doPost(curSid!)
          }
        }
      }
      if (!res.ok) {
        // 422/400 等优先解析 JSON detail；否则读文本
        let msg = `HTTP ${res.status} ${res.statusText}`
        try {
          const j = await res.json()
          if (j?.detail) msg += `\n${JSON.stringify(j.detail)}`
        } catch {
          const t = await res.text().catch(()=> '')
          if (t) msg += `\n${t}`
        }
        throw new Error(msg)
      }

      // 4) 解析返回，并写入预览层
      const data = await res.json()
      if (data?.usage?.new_sid) setSid(String(data.usage.new_sid))

      const payload = data?.payload as AIStrokePayload | undefined
      if (!payload) throw new Error('No payload in response')

      // 记录 feed + 原始文本（便于调试）
      localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload))
      const items = (payload.strokes || []).map(s => ({ id: s.id, desc: (s.meta as any)?.desc }))
      setAiFeed(prev => ([{ payloadId: 'srv_'+Date.now().toString(36), time: Date.now(), items }, ...prev].slice(0, 50)))
      if (data?.usage?.raw_text) localStorage.setItem('ai_last_raw', String(data.usage.raw_text))

      // 规范化 → 计划草案 → 放入预览
      const v = validateAIStrokePayload(payload)
      if (!v.ok || !v.payload) throw new Error('Invalid AI payload: ' + v.errors.join('; '))
      const norm = normalizeAIStrokePayload(v.payload)
      const drafts = planDrafts(norm)   // 已兼容 poly/line/pen 等
      setPreviews(prev => ({ ...prev, [norm.payloadId]: { payloadId: norm.payloadId, drafts } }))
      setCurrentPayloadId(norm.payloadId)
    } catch (err: any) {
      console.error('[askAI] error:', err)
      alert('Ask AI failed:\n' + (err?.message || String(err)))
    }
  }, [sid, drawStack, size.width, size.height, hint, aiScale, mode, visionVersion])
  
  React.useEffect(() => {
    askAIRef.current = askAI
  }, [askAI])

  // 无限网格（随相机视口动态铺展）
  const Grid: React.FC = () => {
    const STEP = 32
    const pad = STEP * 2
    const worldLeft   = (-view.x) / view.scale - pad
    const worldTop    = (-view.y) / view.scale - pad
    const worldRight  = worldLeft + (size.width  / view.scale) + pad * 2
    const worldBottom = worldTop  + (size.height / view.scale) + pad * 2
    const xStart = Math.floor(worldLeft  / STEP) * STEP
    const xEnd   = Math.ceil (worldRight / STEP) * STEP
    const yStart = Math.floor(worldTop   / STEP) * STEP
    const yEnd   = Math.ceil (worldBottom/ STEP) * STEP
    const lines: JSX.Element[] = []
    for (let x = xStart; x <= xEnd; x += STEP) {
      lines.push(<KLine key={'gx'+x} points={[x, yStart, x, yEnd]} stroke="#eee" strokeWidth={1 / view.scale} listening={false} />)
    }
    for (let y = yStart; y <= yEnd; y += STEP) {
      lines.push(<KLine key={'gy'+y} points={[xStart, y, xEnd, y]} stroke="#eee" strokeWidth={1 / view.scale} listening={false} />)
    }
    return <Group listening={false}>{lines}</Group>
  }

  // Draft → Konva 节点
  const DraftNode: React.FC<{ d: ShapeDraft; preview?: boolean }> = ({ d, preview }) => {
    const stroke = colorToStroke(d.style?.color ?? 'black')
    const strokeWidth = SIZE_TO_WIDTH[(d.style?.size ?? 'm')]
    const opacity = preview ? Math.min(0.35, (d.style?.opacity ?? 1)) : (d.style?.opacity ?? 1)
    switch (d.kind) {
      case 'pen':
      case 'polyline': {
        const pts = (d.points ?? []).flatMap(p => [d.x + p.x, d.y + p.y])
        // 根据 meta.curve 决定是否圆滑
        const useCurve = !!d.meta?.curve
        return <KLine points={pts} stroke={stroke} strokeWidth={strokeWidth} tension={useCurve ? 0.4 : 0} lineCap="round" lineJoin="round" opacity={opacity} />
      }
      case 'line': {
        const pts = (d.points ?? []).flatMap(p => [d.x + p.x, d.y + p.y])
        return <KLine points={pts} stroke={stroke} strokeWidth={strokeWidth} lineCap="round" lineJoin="round" opacity={opacity} />
      }
      case 'rect':
        return <KRect x={d.x} y={d.y} width={d.w ?? 1} height={d.h ?? 1} stroke={stroke} strokeWidth={strokeWidth} opacity={opacity} />
      case 'ellipse':
        const w  = d.w ?? 0
        const h  = d.h ?? 0
        const cx = d.x + w / 2
        const cy = d.y + h / 2
        const rx = Math.abs(w) / 2
        const ry = Math.abs(h) / 2
        if (rx === 0 && ry === 0) return null  // 保护：极小尺寸时不渲染
         return (
           <KEllipse
             x={cx}
             y={cy}
             radiusX={rx}
             radiusY={ry}
             stroke={stroke}
             strokeWidth={strokeWidth}
             opacity={opacity}
             listening={false}
           />
         )
      case 'poly': {
        // 关闭 RDP 后，poly 顶点完整保留；这里用 closed 形成闭合多边形
        const pts = (d.points ?? []).flatMap(p => [d.x + p.x, d.y + p.y])
        return <KLine points={pts} closed stroke={stroke} strokeWidth={strokeWidth} opacity={opacity} lineJoin="round" />
      }
      case 'erase':
        return <KText x={d.x} y={d.y} text="[erase]" fill="#999" opacity={opacity} />
      default:
        return null
    }
  }

  // 网格吸附（简单版）
  const GRID_STEP = 32
  const snapPoint = useCallback((x: number, y: number) => {
    // 仅用于“显示阶段”的可选吸附；不要在存储/上传阶段调用
    return snap
      ? [Math.round(x / GRID_STEP) * GRID_STEP, Math.round(y / GRID_STEP) * GRID_STEP]
      : [x, y]
  }, [snap])

  // 近似闭合判定（首尾距离 <= tol 像素 即视为闭合）
  const isClosedPath = useCallback((pts: Array<[number,number]>)=>{
    if (pts.length < 3) return false
    const [x0,y0] = pts[0]
    const [xn,yn] = pts[pts.length-1]
    const tol = snap ? GRID_STEP * 0.5 : 3 // snap=ON 给更宽容的阈值
    return Math.hypot(xn - x0, yn - y0) <= tol
  }, [snap])
  // 把 [x0,y0,x1,y1,...] 按需做可视吸附；用于预览线条
  const snapPointsIfNeeded = useCallback((pts: number[]) => {
    if (!snap) return pts
    const out: number[] = []
    for (let i = 0; i < pts.length; i += 2) {
      const [sx, sy] = snapPoint(pts[i], pts[i+1])
      out.push(sx, sy)
    }
    return out
  }, [snap, snapPoint])


  // —— 新的整根擦除（命中即删除整条笔画） ————————————————
  // 点到线段的最短距离
  const distPointToSegment = (px:number, py:number, x1:number, y1:number, x2:number, y2:number) => {
    const A = px - x1, B = py - y1, C = x2 - x1, D = y2 - y1
    const dot = A*C + B*D
    const len = C*C + D*D
    const t = len ? Math.max(0, Math.min(1, dot / len)) : 0
    const qx = x1 + t*C, qy = y1 + t*D
    return Math.hypot(px - qx, py - qy)
  }
  // 折线到点的最小距离（绝对坐标）
  const polylineMinDistToPoint = (absPts: Array<[number,number]>, px:number, py:number) => {
    if (absPts.length <= 1) return Infinity
    let min = Infinity
    for (let i = 0; i < absPts.length - 1; i++) {
      const [x1,y1] = absPts[i], [x2,y2] = absPts[i+1]
      const d = distPointToSegment(px, py, x1, y1, x2, y2)
      if (d < min) min = d
    }
    return min
  }
  // 命中检测：任一线段到圆心的最短距离 ≤ 半径
  const hitStrokeByCircle = (d: ShapeDraft, cx:number, cy:number, r:number) => {
    // 让 pen / line / polyline / poly 都可整根擦除
    if (!d.points || d.points.length < 2) return false
    const absPts: Array<[number,number]> = d.points.map(p => [d.x + p.x, d.y + p.y])
    return polylineMinDistToPoint(absPts, cx, cy) <= r
  }
  // 应用整根擦除（一次手势只 pushHistory 一次）
  const eraseWholeStrokesAt = (cx:number, cy:number, radius:number) => {
    // --- 工具函数仅在本作用域，用于 ellipse 命中 ---
    const distancePointToSegment = (
      px: number, py: number,
      ax: number, ay: number,
      bx: number, by: number
    ) => {
      const vx = bx - ax, vy = by - ay
      const wx = px - ax, wy = py - ay
      const vv = vx*vx + vy*vy
      let t = vv === 0 ? 0 : (wx*vx + wy*vy) / vv
      t = Math.max(0, Math.min(1, t))
      const cx2 = ax + t*vx, cy2 = ay + t*vy
      return Math.hypot(px - cx2, py - cy2)
    }

    const ellipseToPolyline = (
      cx0: number, cy0: number, rx: number, ry: number, segs = 48
    ): [number, number][] => {
      const n = Math.max(12, segs|0)
      const pts: [number, number][] = []
      for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2
        pts.push([cx0 + rx * Math.cos(t), cy0 + ry * Math.sin(t)])
      }
      pts.push(pts[0]) // 闭合
      return pts
    }

    const hitEllipseByCircle = (d: ShapeDraft, px: number, py: number, r: number) => {
      // 统一成正向包围盒
      const w = d.w ?? 0, h = d.h ?? 0
      const x0 = Math.min(d.x, d.x + w), x1 = Math.max(d.x, d.x + w)
      const y0 = Math.min(d.y, d.y + h), y1 = Math.max(d.y, d.y + h)
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
      const rx = Math.abs(x1 - x0) / 2, ry = Math.abs(y1 - y0) / 2
      if (rx < 0.5 && ry < 0.5) return false

      // 粗筛：点是否落在包围盒“外扩 r”范围内
      if (px < x0 - r || px > x1 + r || py < y0 - r || py > y1 + r) return false

      // 精确：把椭圆边界离散为折线，做点到线段距离
      const pts = ellipseToPolyline(cx, cy, rx, ry, 48)
      for (let i = 1; i < pts.length; i++) {
        const [ax, ay] = pts[i - 1]
        const [bx, by] = pts[i]
        if (distancePointToSegment(px, py, ax, ay, bx, by) <= r) return true
      }
      return false
    }
    // --- end 工具函数 ---

    if (!eraseGestureStarted.current) {
      pushHistory()
      eraseGestureStarted.current = true
    }

    const removed = new Set<string>()
    const kept: ShapeDraft[] = []

    for (const d of shapes) {
      let hit = false
      if (d.kind === 'ellipse') {
        // 仅对 ellipse 采取专门的命中策略；其余仍用你原来的命中函数
        hit = hitEllipseByCircle(d, cx, cy, radius)
      } else {
        hit = hitStrokeByCircle(d, cx, cy, radius)
      }

      if (hit) {
        removed.add(d.id)
      } else {
        kept.push(d)
      }
    }

    if (removed.size) {
      setShapes(kept)
      setDrawStack(prev => prev.filter(e => !removed.has(e.draft.id)))
    }
  }

  // 画笔/橡皮事件（根据 toolMode 分流）

  const onMouseDown = useCallback((e: any) => {
    if (toolMode === 'hand') return
    const pos = e.target.getStage()?.getPointerPosition()
    if (!pos) return
    if (toolMode === 'pen') {
      setIsDrawing(true)
      // snap 打开：存储整点；snap 关闭：存储浮点
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snap ? snapPoint(wpt.x, wpt.y) : [wpt.x, wpt.y]
      setRawPoints([sx, sy])
    } else if (toolMode === 'ellipse') {
      setIsDrawing(true)
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snap ? snapPoint(wpt.x, wpt.y) : [wpt.x, wpt.y]
      const id = `ellipse_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
      setBoxDraft({
        id, kind: 'ellipse', x: sx, y: sy, w: 0, h: 0,
        style: { size: brushSize, color: brushColor, opacity: 1 },
        meta: { author: 'human' }
      })
    } else { // eraser
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })
      eraseGestureStarted.current = false // 新手势开始
      eraseWholeStrokesAt(sx, sy, eraserRadius) // 立即尝试擦除
    }
  }, [snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld])

  const onMouseMove = useCallback((e: any) => {
    if (toolMode === 'hand') return
    const pos = e.target.getStage()?.getPointerPosition()
    if (!pos) return
    if (toolMode === 'pen') {
      if (!isDrawing) return
      setRawPoints(prev => {
        const n = prev.length
        const wpt = screenToWorld(pos.x, pos.y)
        // snap 打开：按网格追加整点；关闭：追加浮点（统一世界坐标）
        const [tx, ty] = snap ? snapPoint(wpt.x, wpt.y) : [wpt.x, wpt.y]
        if (n >= 2 && prev[n-2] === tx && prev[n-1] === ty) return prev
        return [...prev, tx, ty]
      })
    } else if (toolMode === 'ellipse') {
      if (!isDrawing || !boxDraft) return
      const wpt = screenToWorld(pos.x, pos.y)
      const [tx, ty] = snap ? snapPoint(wpt.x, wpt.y) : [wpt.x, wpt.y]
      setBoxDraft(prev => {
        if (!prev) return prev
        const x0 = Math.min(prev.x, tx)
        const y0 = Math.min(prev.y, ty)
        const x1 = Math.max(prev.x, tx)
        const y1 = Math.max(prev.y, ty)
        return { ...prev, x: x0, y: y0, w: (x1 - x0), h: (y1 - y0) }
      })
    } else {
      // 橡皮仍用吸附坐标，命中更稳定
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })           // 显示光标圆
      eraseWholeStrokesAt(sx, sy, eraserRadius)    // 连续整根擦除
    }
  }, [isDrawing, boxDraft, snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld])

  const onMouseUp = useCallback(() => {
    if (toolMode === 'hand') return
    if (toolMode === 'pen') {
      if (!isDrawing) return
      setIsDrawing(false)
      if (rawPoints.length < 4) { setRawPoints([]); return }
      // [x0,y0,x1,y1,...] → [[x,y], ...]
      const absPts: Array<[number, number]> = []
      for (let i = 0; i < rawPoints.length; i += 2) absPts.push([rawPoints[i], rawPoints[i+1]])
      // 先用斜率法压直线冗余点
      let basePts = mergeCollinear(absPts, 0.01)
      // —— A) 闭合图形：绝不坍缩成直线，作为 poly 处理 —— 
      if (isClosedPath(basePts)) {
        // 规范成首尾重合的闭合点列
        if (basePts.length >= 2) {
          const [x0,y0] = basePts[0]
          const [xn,yn] = basePts[basePts.length-1]
          if (x0 !== xn || y0 !== yn) basePts = [...basePts, [x0,y0]]
        }
        // 进一步减少共线冗点，但保留转角
        const closedPts = mergeCollinear(basePts, 0.0)
        // 入历史
        pushHistory()
        // 构造 poly 的 Draft 与 AI 笔画
        let minX=Infinity, minY=Infinity
        for (const [x,y] of closedPts){ if (x<minX) minX=x; if (y<minY) minY=y }
        const local = closedPts.map(([x,y])=>({x:x-minX,y:y-minY}))
        const id = `poly_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const draft = { id, kind:'poly', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta} }
        const aiStroke = { id, tool:'poly', points: closedPts.map(([x,y])=>[x,y] as [number,number]), style: currentBrush.style, meta:{ author:'human' } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
        return
      }
      // —— B) 非闭合：根据开关决定是否转曲线 —— 
      let displayPts: Array<[number,number]>
      if (curveTurns) {
        // 曲线：Chaikin + 等距重采样（保留丝滑手感）
        displayPts = resampleEvenly(chaikin(basePts, 2), 3)
      } else {
        // 折线：保持拐角（tension=0 渲染）
        displayPts = basePts
      }
      // 直线识别（近似首尾 chord 的最大偏差）
      const LINEAR_EPS = 1.2
      const isLine = geomMaxDeviationFromChord(displayPts) <= LINEAR_EPS
      // 入历史
      pushHistory()
      // 构造 Draft + AI 笔画
      if (isLine && displayPts.length >= 2) {
        const p0 = displayPts[0], pn = displayPts[displayPts.length-1]
        const id = `line_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const minX = Math.min(p0[0], pn[0]), minY = Math.min(p0[1], pn[1])
        const local = [{x:p0[0]-minX, y:p0[1]-minY}, {x:pn[0]-minX, y:pn[1]-minY}]
        const draft = { id, kind:'line', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta} }
        const aiStroke = { id, tool:'line', points:[p0, pn], style: currentBrush.style, meta:{ author:'human' } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
      } else {
        // 非直线：pen，多段；把“曲线/折线”写入 meta.curve 供渲染 tension 使用
        let minX=Infinity, minY=Infinity
        for (const [x,y] of displayPts){ if (x<minX) minX=x; if (y<minY) minY=y }
        const local = displayPts.map(([x,y])=>({x:x-minX,y:y-minY}))
        const id = `pen_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const draft = { id, kind:'pen', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta, curve: curveTurns} }
        const aiStroke = { id, tool:'pen', points: displayPts.map(([x,y])=>[x,y] as [number,number]), style: currentBrush.style, meta:{ author:'human', curve: curveTurns } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
      }
    } else if (toolMode === 'ellipse') {
      if (!isDrawing || !boxDraft) return
      setIsDrawing(false)
      // 归一化对角点（绝对坐标）
      const x0 = Math.min(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const y0 = Math.min(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      const x1 = Math.max(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const y1 = Math.max(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      // 极小尺寸保护：避免 0 宽/高
      if (Math.abs(x1 - x0) < 1 && Math.abs(y1 - y0) < 1) { setBoxDraft(null); return }
      // 入历史
      pushHistory()
      // 构造 Draft 与 AI 笔画
      const id = boxDraft.id
      const draft: ShapeDraft = {
        id, kind:'ellipse',
        x: x0, y: y0, w: (x1 - x0), h: (y1 - y0),
        style: boxDraft.style, meta: { ...boxDraft.meta }
      }
      const aiStroke: AIStrokeV11 = {
        id, tool: 'ellipse',
        points: [[x0,y0],[x1,y1]],
        style: boxDraft.style, meta: { author:'human' }
      }
      setShapes(prev => [...prev, draft])
      setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
      setBoxDraft(null)
    } else {
      // 橡皮结束
      setEraserCursor(null)
      eraseGestureStarted.current = false
    }
  }, [isDrawing, rawPoints, boxDraft, snap, snapPoint, toolMode, curveTurns, currentBrush, pushHistory, setShapes, setDrawStack])

  // —— 回车快捷键：若当前存在 AI 预览，则 Enter 自动 Accept（避免打断输入聚焦） ——
  React.useEffect(()=>{
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== 'Enter') return
      const tgt = ev.target as HTMLElement | null
      const isTyping = tgt && (
        tgt.tagName === 'INPUT' ||
        tgt.tagName === 'TEXTAREA' ||
        (tgt as any).isContentEditable
      )
      if (isTyping) return
      if (currentPayloadId) {
        ev.preventDefault()
        acceptAI()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [currentPayloadId, acceptAI])

  React.useEffect(()=>{
    if (toolMode === 'hand') {
      // 关闭正在进行的绘制/预览
      setIsDrawing(false)
      // 如果你有 boxDraft/previewDraft 等，清掉
      setBoxDraft?.(null as any)
    }
  }, [toolMode])
  

// ====== 舞台（绑定相机：x/y/scale；Hand 模式可拖拽） ======
  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow:'hidden' }}>
      {/* 顶部工具条（只留 6 个） */}
      <div style={{
        position:'absolute', left:'50%', transform:'translateX(-50%)',
        top:16, zIndex:1000, display:'flex', gap:8, alignItems:'center',
        background:'rgba(255,255,255,0.85)', backdropFilter:'blur(6px)',
        padding:'8px 12px', borderRadius:16, boxShadow:'0 6px 18px rgba(0,0,0,0.12)', border:'1px solid #e5e7eb'
      }}>
        <button style={BTN} onClick={()=>setShowGrid(s=>!s)}>{showGrid?'Grid: ON':'Grid: OFF'}</button>
        <button style={BTN} onClick={()=>setSnap(s=>!s)}>{snap?'Snap: ON':'Snap: OFF'}</button>
        <button style={BTN} onClick={()=>setCurveTurns(s=>!s)}>{curveTurns ? 'Curve: ON' : 'Curve: OFF'}</button>

        <div style={{ width:10, height:24, borderLeft:'1px solid #e5e7eb', margin:'0 4px' }} />

        <button style={{...BTN, borderColor:'#4aa3ff', background:'rgba(74,163,255,0.14)'}} onClick={askAI}>Ask AI</button>
        <button style={{...BTN, borderColor:'#52d273', background:'rgba(82,210,115,0.14)'}} onClick={acceptAI}>Accept</button>
        <button style={{...BTN, borderColor:'#ff6b6b', background:'rgba(255,107,107,0.14)'}} onClick={dismissAI}>Dismiss</button>
      </div>

      {/* 右侧侧栏（工具、调色盘、AI Scale、其余功能键） */}
      <div style={{
        position:'absolute', top:80, right:16, bottom:180, width:300, zIndex:1000,
        display:'flex', flexDirection:'column', gap:12, overflow:'auto',
        padding:12, background:'rgba(255,255,255,0.75)', backdropFilter:'blur(8px)',
        border:'1px solid #e5e7eb', borderRadius:16, boxShadow:'0 8px 24px rgba(0,0,0,0.12)'
      }}>
        {/* 工具 */}
        <section style={CARD}>
          <div style={CARD_TITLE}>Tools</div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:8 }}>
            {(['hand','pen','eraser','ellipse'] as const).map(t => (
              <button
                key={t}
                style={{
                  ...BTN, padding:'8px 10px',
                  ...(toolMode===t ? { outline:'2px solid #4aa3ff', background:'rgba(74,163,255,0.12)' } : {})
                }}
                onClick={()=>setToolMode(t)}
                title={t}
              >{t}</button>
            ))}
          </div>

          {/* 橡皮参数（保持你的名称与逻辑） */}
          {toolMode === 'eraser' && (
            <div style={{ marginTop:10, display:'flex', alignItems:'center', gap:8 }}>
              <span style={{fontSize:12,color:'#555'}}>Radius</span>
              <input
                style={{ ...SEL, width: 90 }}
                type="number"
                min={4}
                max={64}
                step={2}
                value={eraserRadius}
                onChange={(e)=>setEraserRadius(Math.max(4, Math.min(64, Number(e.target.value) || 14)))}
                title="Eraser radius (px)"
              />
            </div>
          )}
        </section>

        {/* 画笔（尺寸 + 调色盘）（名称保持不变） */}
        <section style={CARD}>
          <div style={CARD_TITLE}>Brush</div>
          <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:8 }}>
            <span style={{fontSize:12,color:'#555'}}>Size</span>
            <select style={{ ...SEL, width:90 }} value={brushSize} onChange={(e)=>setBrushSize(e.target.value as 's'|'m'|'l'|'xl')}>
              <option value="s">S</option><option value="m">M</option><option value="l">L</option><option value="xl">XL</option>
            </select>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(7, 1fr)', gap:8 }}>
            {COLORS.map(c => (
              <button
                key={c}
                title={c}
                onClick={()=>setBrushColor(c as any)}
                style={{
                  width:28, height:28, borderRadius:8,
                  border: `2px solid ${brushColor===c ? '#4aa3ff' : '#e5e7eb'}`,
                  background: c === 'white' ? '#fff' : c.replace('light-','light'),
                  boxShadow:'inset 0 0 0 1px rgba(0,0,0,0.04)'
                }}
              />
            ))}
          </div>
        </section>

        {/* AI Scale 滑条（保持你的 aiScale / setAiScale） */}
        <section style={CARD}>
          <div style={CARD_TITLE}>AI Scale</div>
          <div style={{display:'flex', alignItems:'center', gap:10}}>
            <input
              type="range" min={4} max={64} step={1}
              value={aiScale}
              onChange={(e)=>setAiScale(Number(e.target.value)||16)}
              title="Max points for AI stroke (model is asked to keep ≤ this)"
              style={{ flex:1 }}
            />
            <span style={{fontSize:12, color:'#333', width:32, textAlign:'right'}}>{aiScale}</span>
          </div>
        </section>

        {/* Auto Complete (5s) */}
        <section style={CARD}>
          <div style={CARD_TITLE}>Auto Complete</div>
          <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', gap:8}}>
            <label style={{fontSize:13, color:'#333'}}>
              自动补全（5秒无操作触发 askAI）
            </label>
            <input
              type="checkbox"
              checked={autoComplete}
              onChange={(e)=>{ setAutoComplete(e.target.checked); clearAutoTimer() }}
              title="开启后：无预览且5秒无新操作自动发送"
            />
          </div>
          <div style={{marginTop:6, fontSize:12, color:'#666'}}>
            状态：{hasActivePreview ? '有预览，暂停自动发送' : (autoCountdown!=null ? `倒计时 ${autoCountdown}s` : '空闲')}
          </div>
        </section>
        {/* 其余功能性按键（名称全部保持不变） */}
        <section style={CARD}>
          <div style={CARD_TITLE}>Actions</div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:8 }}>
            <button style={{...BTN, opacity: past.length?1:0.6}} onClick={undo} disabled={!past.length}>Undo</button>
            <button style={{...BTN, opacity: future.length?1:0.6}} onClick={redo} disabled={!future.length}>Redo</button>
            
            <button style={BTN} onClick={exportJSON}>Export JSON</button>
            <button style={BTN} onClick={()=>fileRef.current?.click()}>Import JSON</button>
            <input ref={fileRef} type="file" accept="application/json,.json" style={{ display:'none' }}
                  onChange={(e)=>{ const f=e.target.files?.[0]; if (f) importJSON(f) }} />
            <button style={BTN} onClick={exportHumanStrokesAI}>Export Strokes (AI)</button>

            <button style={BTN} onClick={applyAIStub}>Apply AI (stub)</button>
            <button style={BTN} onClick={previewAI}>Preview AI</button>

          </div>
          {/* 仅在 Vision 模式显示的版本输入 */}
          {mode === 'vision' && (
            <div style={{ marginTop: 8, display:'flex', alignItems:'center', gap:8 }}>
              <label style={{ fontSize: 12, color:'#333', width: 120 }}>
                Vision version
              </label>
              <input
                type="number"
                step="0.1"
                min={1.0}
                value={visionVersion}
                onChange={(e)=> setVisionVersion(Number(e.target.value) || 2.0)}
                style={{ ...SEL, width: 120, height: 32, borderRadius: 8, padding: '0 8px' }}
                title="Vision 模式的协议版本（2.0 为二段式）"
              />
            </div>
          )}
        </section>
      </div>

      {/* Konva 画布（全屏，侧栏为悬浮覆盖） */}
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        x={view.x}
        y={view.y}
        scaleX={view.scale}
        scaleY={view.scale}
        draggable={toolMode === 'hand'}
        onDragStart={()=> setIsPanning(true)}
        onDragMove={(e:any)=>{
          if (toolMode !== 'hand') return
          const { x, y } = e.target.position()
          setView(v=>({ ...v, x, y }))
        }}
        onDragEnd={(e:any)=>{
          const { x, y } = e.target.position()
          setView(v=>({ ...v, x, y }))
          setIsPanning(false)
        }}
        onWheel={onWheelZoom}
        onMouseDown={(e)=>{ /* 用户开始一次可能改变内容的操作 */
          noteUserAction()
          onMouseDown(e)
        }}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{ cursor: toolMode==='hand' ? (isPanning ? 'grabbing' : 'grab') : 'default' }}
      >
        {/* 网格层 */}
        <Layer listening={false}>{showGrid && <Grid />}</Layer>

        {/* 正式形状层 */}
        <Layer>
          {shapes.map(d => <DraftNode key={'s:'+d.id} d={d} />)}
          {/* 绘制中预览线条（不入 shapes） */}
          {toolMode==='pen' && isDrawing && rawPoints.length >= 4 && (
            <KLine
              points={snapPointsIfNeeded(rawPoints)}
              stroke={colorToStroke(brushColor)}
              strokeWidth={SIZE_TO_WIDTH[brushSize]}
              tension={snap ? 0 : 0.4}
              lineCap="round"
              lineJoin="round"
              opacity={0.8}
            />
          )}
          {toolMode==='ellipse' && isDrawing && boxDraft && (
            <DraftNode d={boxDraft} />
          )}
          {/* 橡皮光标圆（可视化半径/位置）；用圆角矩形等价画圆，避免额外 import */}
          {toolMode==='eraser' && !isPanning && eraserCursor && (
            <Group listening={false}>
              <KRect
                x={eraserCursor.x - eraserRadius}
                y={eraserCursor.y - eraserRadius}
                width={eraserRadius * 2}
                height={eraserRadius * 2}
                cornerRadius={eraserRadius}
                stroke="#9aa0a6"
                dash={[4,4]}
                opacity={0.85}
              />
            </Group>
          )}
        </Layer>

        {/* AI 预览层 */}
        <Layer>
          {Object.values(previews).map(entry => (
            <Group key={'p:'+entry.payloadId} listening={false} name="ai-candidate" id={entry.payloadId}>
              {entry.drafts.map(d => <DraftNode key={'pd:'+entry.payloadId+':'+d.id} d={d} preview />)}
            </Group>
          ))}
        </Layer>
      </Stage>

      {/* 底部 AI Feed 面板：把 Hint 输入框放到最顶端 */}
      <div style={{
        position:'absolute', left:16, right:16, bottom:16, zIndex:1000,
        background:'rgba(255,255,255,0.85)', backdropFilter:'blur(8px)',
        border:'1px solid #e5e7eb', borderRadius:12, padding:'10px 12px',
        boxShadow:'0 8px 24px rgba(0,0,0,0.12)', maxHeight:220, overflow:'auto'
      }}>
        {/* Hint 输入（从顶栏移到这里；名称保持不变） */}
        <div style={{ display:'flex', gap:8, alignItems:'center', marginBottom:8 }}>
          <input
            style={{ ...SEL, width:'100%', borderRadius:10, height:40, padding:'0 12px' }}
            type="text"
            placeholder="hint for AI, e.g. clean curves / refine hair"
            value={hint}
            onChange={(e)=>setHint(e.target.value)}
            title="Hint sent to backend /suggest"
            onKeyDown={(e)=>{
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askAI();
              }
            }}
          />
          <button
            title={
              mode === "light"
                ? "轻量补全：仅预测下一笔"
                : mode === "full"
                ? "常规补全：可多笔"
                : "视觉增强：AI视觉理解与创意绘制"
            }
            onClick={() =>
              setMode((m) =>
                m === "light" ? "full" : m === "full" ? "vision" : "light"
              )
            }
            style={{
              padding: '8px 18px',
              borderRadius: '10px',
              border: '2px solid',
              fontSize: '14px',
              fontWeight: 'bold',
              cursor: 'pointer',
              transition: 'all 0.4s ease',
              borderColor:
                mode === "light"
                  ? '#4aa3ff'
                  : mode === "full"
                  ? '#ffb84a'
                  : '#9b5cff',
              background:
                mode === "light"
                  ? 'linear-gradient(135deg, rgba(74,163,255,0.15), rgba(74,163,255,0.05))'
                  : mode === "full"
                  ? 'linear-gradient(135deg, rgba(255,184,74,0.15), rgba(255,184,74,0.05))'
                  : 'linear-gradient(135deg, rgba(155,92,255,0.25), rgba(255,92,200,0.25))',
              color:
                mode === "light"
                  ? '#4aa3ff'
                  : mode === "full"
                  ? '#ffb84a'
                  : '#c88bff',
              boxShadow:
                mode === "vision"
                  ? '0 0 12px rgba(155,92,255,0.6), 0 0 24px rgba(255,92,200,0.4)'
                  : 'none',
              textShadow:
                mode === "vision"
                  ? '0 0 6px rgba(255,255,255,0.8)'
                  : 'none',
            }}
          >
            {mode === "light" ? "LIGHT" : mode === "full" ? "FULL" : "VISION"}
          </button>
          <button
            style={{ ...BTN, borderColor:'#4aa3ff', background:'rgba(74,163,255,0.14)' }}
            onClick={askAI}
          >
            Send
          </button>
        </div>


        <div style={{fontSize:12, color:'#666', marginBottom:6}}>AI Feed (latest)</div>
        {aiFeed.length === 0 ? (
          <div style={{fontSize:12, color:'#999'}}>No AI packages yet.</div>
        ) : (
          aiFeed.map(entry=>(
            <div key={entry.payloadId} style={{marginBottom:6}}>
              <div style={{fontSize:12, color:'#444'}}>
                <b>{new Date(entry.time).toLocaleTimeString()}</b> — payload <code>{entry.payloadId}</code>
              </div>
              <ul style={{margin:'4px 0 0 16px', padding:0}}>
                {entry.items.map((it,idx)=>(
                  <li key={it.id+'_'+idx} style={{fontSize:12, color:'#333', listStyle:'disc'}}>
                    <code>{it.id}</code>{it.desc ? ` — ${it.desc}` : ''}
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

// 简单按钮/选择器样式
const BTN: React.CSSProperties = { padding:'6px 10px', borderRadius:999, border:'1px solid #ccc', background:'#fff', cursor:'pointer' }
const CARD: React.CSSProperties = { padding:12, border:'1px solid #e5e7eb', borderRadius:12, background:'rgba(255,255,255,0.6)' }
const CARD_TITLE: React.CSSProperties = { fontSize:12, color:'#6b7280', marginBottom:8, letterSpacing:'.3px', textTransform:'uppercase' }
const SEL: React.CSSProperties = { padding:'6px 10px', borderRadius:999, border:'1px solid #ccc', background:'#fff' }
const BTN_BASE = {padding: '8px 16px',borderRadius: '6px',border: '2px solid transparent',fontSize: '14px',fontWeight: 'bold',cursor: 'pointer',transition: 'all 0.3s ease',};
