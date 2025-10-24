import type { AIStrokePayload, AIStrokeV11, ColorName } from './types'

export const COLORS: ColorName[] = [
  'black','blue','green','grey','light-blue','light-green','light-red','light-violet','orange','red','violet','white','yellow'
]

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))

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

function clampPoints(points: AIStrokeV11['points'], viewport?: [number, number, number, number]): AIStrokeV11['points'] {
  if (!viewport) return points
  const [vx, vy, vw, vh] = viewport
  const x0 = vx, y0 = vy, x1 = vx + vw, y1 = vy + vh
  return points.map(([x, y, t, p]) => [clamp(x, x0, x1), clamp(y, y0, y1), t, p]) as AIStrokeV11['points']
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

function simplifyRDP(points: AIStrokeV11['points'], eps = 0.4): AIStrokeV11['points'] {
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
    const seg = stack.pop()!
    const i = seg[0], j = seg[1]
    let idx=-1, maxd=-1
    for (let k=i+1; k<j; k++) {
      const d = dist([points[k][0],points[k][1]],[points[i][0],points[i][1]],[points[j][0],points[j][1]])
      if (d>maxd) { maxd=d; idx=k }
    }
    if (maxd>eps && idx!==-1) { keep[idx]=true; stack.push([i,idx],[idx,j]) }
  }
  return points.filter((_,i)=>keep[i]) as AIStrokeV11['points']
}

export function computeBounds(points: AIStrokeV11['points']) {
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity
  for (const [x,y] of points) { if (x<minX) minX=x; if (y<minY) minY=y; if (x>maxX) maxX=x; if (y>maxY) maxY=y }
  return { x:minX, y:minY, w: Math.max(1,maxX-minX), h: Math.max(1,maxY-minY) }
}

export function isClosedStroke(points: AIStrokeV11['points'], tol = 1.5) {
  if (!points || points.length < 3) return false
  const [x0, y0] = points[0]
  const [xn, yn] = points[points.length - 1]
  return Math.hypot(xn - x0, yn - y0) <= tol
}

function normalizeStroke(s: AIStrokeV11, viewport?: [number,number,number,number]): AIStrokeV11 {
  const size = (s.style?.size as 's'|'m'|'l'|'xl') ?? 'm'
  const color = COLORS.includes((s.style?.color as ColorName)) ? (s.style!.color as ColorName) : 'black'
  const opacity = typeof s.style?.opacity==='number' ? clamp(s.style!.opacity!,0,1) : 1
  const raw = dedupeConsecutive(clampPoints(s.points, viewport))
  const closed = (s.tool !== 'poly') && isClosedStroke(raw)
  const eps = (s.tool === 'poly' || closed) ? 0 : 0.4
  let pts: AIStrokeV11['points'] = eps === 0 ? raw : simplifyRDP(raw, eps)
  if (pts.length <= 2) {
    const [x0, y0] = pts[0] ?? [NaN, NaN]
    const [xn, yn] = pts[pts.length - 1] ?? [NaN, NaN]
    if (Number.isFinite(x0) && Math.hypot((xn ?? x0) - x0, (yn ?? y0) - y0) <= 1e-6) {
      pts = raw
    }
  }
  return { id: String(s.id), tool: String(s.tool), points: pts, style: { size, color, opacity }, meta: s.meta ?? {} }
}

export function normalizeAIStrokePayload(raw: AIStrokePayload) {
  const viewport = raw.canvas?.viewport
  const strokes = raw.strokes.map(s=>normalizeStroke(s, viewport))
  const payloadId = `ai_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`
  return { payloadId, strokes, intent: raw.intent, replace: raw.replace, viewport }
}
