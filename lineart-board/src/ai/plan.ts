import type { ColorName } from './types'
import { computeBounds, normalizeAIStrokePayload } from './normalize'

export type ShapeDraft = {
  id: string
  kind: 'pen'|'line'|'rect'|'ellipse'|'poly'|'polyline'|'erase'|'text'
  x: number
  y: number
  w?: number
  h?: number
  points?: Array<{ x:number; y:number; pressure?: number }>
  text?: string
  summary?: string
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
      case 'text':
        const [p0, p1] = s.points || []
        if (!p0 || !p1)
          throw new Error(`Text stroke ${s.id} must have exactly 2 points.`)
        const x0 = Math.min(p0[0], p1[0])
        const y0 = Math.min(p0[1], p1[1])
        const x1 = Math.max(p0[0], p1[0])
        const y1 = Math.max(p0[1], p1[1])

        drafts.push({
          id: s.id,
          kind: 'text',
          x: x0,
          y: y0,
          w: x1 - x0,
          h: y1 - y0,
          text: (s.meta as any)?.text || '',
          summary: (s.meta as any)?.summary || '',
          style: {
            size: s.style?.size ?? 'm',
            color: s.style?.color ?? 'black',
            opacity: s.style?.opacity ?? 1
          },
          meta: {
            ...s.meta,
            // we keep font info etc in meta.fontFamily/fontWeight/fontSize/growDir
          }
        })
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

