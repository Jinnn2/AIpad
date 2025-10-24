import type { AIStrokeV11 } from './types'
import type { ShapeDraft } from './plan'

// Chaikin smoothing yields a smooth polyline approximation of the input points.
export function chaikin(points: Array<[number, number]>, iterations = 2): Array<[number, number]> {
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

// Evenly resample a polyline to keep a consistent segment length.
export function resampleEvenly(points: Array<[number, number]>, step = 3): Array<[number, number]> {
  if (points.length <= 2) return points
  const segs: number[] = []
  let total = 0
  for (let i = 0; i < points.length - 1; i++) {
    const dx = points[i + 1][0] - points[i][0]
    const dy = points[i + 1][1] - points[i][1]
    const d = Math.hypot(dx, dy)
    segs.push(d)
    total += d
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
      out.push([x, y])
      dist += step
    } else {
      acc += segLen
      i++
    }
  }
  out.push(points[points.length - 1])
  return out
}

// Measure the maximum deviation to the chord joining the first/last point.
export function geomMaxDeviationFromChord(pts: Array<[number, number]>): number {
  if (pts.length <= 2) return 0
  const [x1, y1] = pts[0]
  const [x2, y2] = pts[pts.length - 1]
  const cx = x2 - x1
  const cy = y2 - y1
  const l2 = cx * cx + cy * cy || 1e-9
  let maxd = 0
  for (let i = 1; i < pts.length - 1; i++) {
    const [x, y] = pts[i]
    const vx = x - x1
    const vy = y - y1
    const cross = Math.abs(vx * cy - vy * cx)
    const d = cross / Math.sqrt(l2)
    if (d > maxd) maxd = d
  }
  return maxd
}

// Merge consecutive collinear points to keep path data compact.
export function mergeCollinear(pts: Array<[number, number]>, slopeEps = 0.01): Array<[number, number]> {
  if (pts.length <= 2) return pts

  const out: Array<[number, number]> = [pts[0]]
  for (let i = 1; i < pts.length - 1; i++) {
    const a = out[out.length - 1]
    const b = pts[i]
    const c = pts[i + 1]

    const dx1 = b[0] - a[0]
    const dy1 = b[1] - a[1]
    const dx2 = c[0] - b[0]
    const dy2 = c[1] - b[1]

    const slope1 = dx1 === 0 ? null : dy1 / dx1
    const slope2 = dx2 === 0 ? null : dy2 / dx2

    let collinear = false
    if (slope1 === null && slope2 === null) collinear = true
    else if (slope1 === null || slope2 === null) collinear = false
    else if (Math.abs(slope1 - slope2) < slopeEps) collinear = true

    if (!collinear) out.push(b)
  }
  out.push(pts[pts.length - 1])
  return out
}

// Convert a stored draft back to the AI stroke payload shape.
export function draftToAIStroke(d: ShapeDraft): AIStrokeV11 | null {
  if (d.kind === 'line') {
    if (!d.points || d.points.length < 2) return null
    const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    const p0 = pointsAbs[0]
    const pn = pointsAbs[pointsAbs.length - 1]
    return {
      id: d.id,
      tool: 'line',
      points: [p0, pn],
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) },
    }
  }

  if (d.kind === 'pen' || d.kind === 'polyline') {
    if (!d.points || d.points.length < 2) return null
    const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    return {
      id: d.id,
      tool: 'pen',
      points: pointsAbs,
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) },
    }
  }

  if (d.kind === 'poly') {
    if (!d.points || d.points.length < 2) return null
    const pointsAbs = d.points.map(p => [d.x + p.x, d.y + p.y] as [number, number])
    return {
      id: d.id,
      tool: 'poly',
      points: pointsAbs,
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) },
    }
  }

  if (d.kind === 'ellipse') {
    const x0 = d.x
    const y0 = d.y
    const x1 = d.x + (d.w ?? 0)
    const y1 = d.y + (d.h ?? 0)
    if (Math.abs(x1 - x0) < 1e-9 && Math.abs(y1 - y0) < 1e-9) return null
    const p0: [number, number] = [Math.min(x0, x1), Math.min(y0, y1)]
    const p1: [number, number] = [Math.max(x0, x1), Math.max(y0, y1)]
    return {
      id: d.id,
      tool: 'ellipse',
      points: [p0, p1],
      style: d.style ? { ...d.style } : undefined,
      meta: { author: 'ai', ...(d.meta ?? {}) },
    }
  }

  if (d.kind === 'text') {
    const x0 = d.x
    const y0 = d.y
    const x1 = d.x + (d.w ?? 0)
    const y1 = d.y + (d.h ?? 0)

    const p0: [number, number] = [Math.min(x0, x1), Math.min(y0, y1)]
    const p1: [number, number] = [Math.max(x0, x1), Math.max(y0, y1)]

    return {
      id: d.id,
      tool: 'text',
      points: [p0, p1],
      style: d.style ? { ...d.style } : undefined,
      meta: {
        ...(d.meta ?? {}),
        text: d.text ?? '',
        summary: d.summary ?? '',
      }
    }
  }
  return null
}
