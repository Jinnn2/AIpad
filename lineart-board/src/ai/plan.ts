import type { ColorName } from './types'
import { computeBounds, normalizeAIStrokePayload } from './normalize'
import { computeTextBoxLayout, DEFAULT_TEXTBOX_LINE_HEIGHT } from '../textbox/layout'

export type ShapeDraft = {
  id: string
  kind: 'pen'|'line'|'rect'|'ellipse'|'poly'|'polyline'|'erase'|'text'|'edit'
  x: number
  y: number
  w?: number
  h?: number
  points?: Array<{ x:number; y:number; pressure?: number }>
  text?: string
  summary?: string
  style?: { size:'s'|'m'|'l'|'xl'; color: ColorName; opacity:number }
  meta?: Record<string,any>
  targetId?: string
  operation?: string
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
      case 'text': {
        const [p0, p1] = s.points || []
        if (!p0 || !p1)
          throw new Error(`Text stroke ${s.id} must have exactly 2 points.`)
        const x0 = Math.min(p0[0], p1[0])
        const y0 = Math.min(p0[1], p1[1])
        const x1 = Math.max(p0[0], p1[0])
        const y1 = Math.max(p0[1], p1[1])

        const rawMeta = (s.meta ?? {}) as Record<string, any>
        const textContent = String(rawMeta.text ?? '')
        const summary = String(rawMeta.summary ?? '')
        const fontFamily = String(rawMeta.fontFamily ?? 'sans-serif')
        const fontSize = Number(rawMeta.fontSize ?? 16) || 16
        const fontWeight = String(rawMeta.fontWeight ?? '400')
        const growDir = (rawMeta.growDir as any) ?? 'down'
        const baseWidth = Number(rawMeta.configuredWidth ?? rawMeta.baseWidth ?? (x1 - x0)) || (x1 - x0) || 240
        const baseHeight = Number(rawMeta.configuredHeight ?? rawMeta.baseHeight ?? (y1 - y0)) || (y1 - y0) || 160
        const padding = Number(rawMeta.padding ?? 0)
        const rawLineHeight = Number(rawMeta.lineHeight)
        const lineHeight = Number.isFinite(rawLineHeight) && rawLineHeight > 0 ? rawLineHeight : DEFAULT_TEXTBOX_LINE_HEIGHT

        const layout = computeTextBoxLayout({
          text: textContent,
          fontFamily,
          fontSize,
          fontWeight,
          baseWidth,
          baseHeight,
          growDir,
          padding,
          lineHeight,
        })

       const posX = x0 + layout.offsetX
       const posY = y0 + layout.offsetY
        const actualLineHeight = fontSize * layout.lineHeight
        const heightPadding = Math.min(actualLineHeight * 0.35, 16)
        const paddedHeight = layout.height + heightPadding

       drafts.push({
         id: s.id,
         kind: 'text',
         x: posX,
         y: posY,
         w: layout.width,
         h: paddedHeight,
         text: textContent,
         summary,
          style: {
            size: s.style?.size ?? 'm',
            color: s.style?.color ?? 'black',
            opacity: s.style?.opacity ?? 1
          },
          meta: {
            ...rawMeta,
            text: textContent,
            summary,
            fontFamily,
            fontWeight,
            fontSize,
            growDir,
            configuredWidth: baseWidth,
            configuredHeight: baseHeight,
            baseWidth: layout.baseWidth,
            baseHeight: layout.baseHeight,
            lineHeight: layout.lineHeight,
            padding: layout.padding,
            contentWidth: layout.contentWidth,
           contentHeight: layout.contentHeight,
           lineCount: layout.lineCount,
           renderedText: layout.renderedText,
           heightPadding,
         }
       })
        break
      }
      case 'edit': {
        const meta = (s.meta ?? {}) as Record<string, any>
        const rawTarget = meta.targetId ?? meta.target_id ?? meta.target ?? meta.id
        if (!rawTarget) break
        const targetId = String(rawTarget)
        const operation = String(meta.operation ?? '').trim()
        const content = String(meta.content ?? '').trim()
        const [p0, p1] = s.points || []
        const x0 = p0 ? Math.min(p0[0], (p1 ?? p0)[0]) : 0
        const y0 = p0 ? Math.min(p0[1], (p1 ?? p0)[1]) : 0
        const x1 = p1 ? Math.max(p0 ? p0[0] : 0, p1[0]) : x0 + 160
        const y1 = p1 ? Math.max(p0 ? p0[1] : 0, p1[1]) : y0 + 80
        drafts.push({
          id: s.id,
          kind: 'edit',
          x: x0,
          y: y0,
          w: x1 - x0,
          h: y1 - y0,
          text: content,
          summary: '',
          style,
          meta: {
            ...meta,
            targetId,
            operation,
            content,
          },
          targetId,
          operation,
        })
        break
      }
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
