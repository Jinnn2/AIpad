export type GrowDirection = 'down' | 'up' | 'left' | 'right' | 'right-down'

export type TextBoxLayoutInput = {
  text: string
  fontFamily: string
  fontSize: number
  fontWeight: string
  baseWidth: number
  baseHeight: number
  growDir: GrowDirection
  padding?: number
  lineHeight?: number
}

export type TextBoxLayoutOutput = {
  width: number
  height: number
  contentWidth: number
  contentHeight: number
  baseWidth: number
  baseHeight: number
  padding: number
  lineHeight: number
  offsetX: number
  offsetY: number
  lineCount: number
  lines: string[]
  renderedText: string
}

export const DEFAULT_TEXTBOX_LINE_HEIGHT = 1.4
const DEFAULT_PADDING = 0
const MIN_WIDTH = 80

let measureCtx: CanvasRenderingContext2D | null = null

const ensureMeasureContext = () => {
  if (typeof document === 'undefined') return null
  if (measureCtx) return measureCtx
  const canvas = document.createElement('canvas')
  canvas.width = 1
  canvas.height = 1
  measureCtx = canvas.getContext('2d')
  return measureCtx
}

const buildFont = (size: number, weight: string, family: string) => {
  const fallback = 'sans-serif'
  const trimmed = (family || '').split(',').map((f) => f.trim()).filter(Boolean)
  const primary = trimmed.length ? trimmed[0] : fallback
  const safeFamily = /\s/.test(primary) ? `"${primary}"` : primary
  return `${weight || '400'} ${size}px ${safeFamily}`
}

const createMeasure = (fontFamily: string, fontSize: number, fontWeight: string) => {
  const ctx = ensureMeasureContext()
  const fallbackWidth = (txt: string) => txt.length * fontSize * 0.6
  if (!ctx) return fallbackWidth
  try {
    ctx.font = buildFont(fontSize, fontWeight, fontFamily)
    ctx.textBaseline = 'top'
    ctx.textAlign = 'left'
  } catch {
    return fallbackWidth
  }
  return (txt: string) => ctx.measureText(txt).width
}

export const measureTextWidth = (
  text: string,
  fontFamily: string,
  fontSize: number,
  fontWeight: string,
) => {
  const measure = createMeasure(fontFamily, fontSize, fontWeight)
  return measure(String(text ?? ''))
}

const normalizeParagraph = (paragraph: string) => paragraph.replace(/\s+/g, ' ').trim()

type WrappedResult = {
  lines: string[]
  maxLineWidth: number
  lineHeight: number
}

const wrapTextWithWidth = (
  text: string,
  width: number,
  measure: (txt: string) => number,
  lineHeight: number,
): WrappedResult => {
  if (!Number.isFinite(width) || width <= 0) width = MIN_WIDTH
  const paragraphs = text.replace(/\r/g, '').split('\n')
  const lines: string[] = []
  let maxLineWidth = 0

  const pushLine = (line: string) => {
    const content = line.trim()
    const finalLine = content.length ? content : ''
    lines.push(finalLine)
    maxLineWidth = Math.max(maxLineWidth, measure(finalLine))
  }

  const pushChunks = (segment: string) => {
    if (!segment) {
      pushLine('')
      return
    }
    let chunk = ''
    for (const ch of segment) {
      const candidate = chunk + ch
      if (measure(candidate) <= width || !chunk) {
        chunk = candidate
      } else {
        pushLine(chunk)
        chunk = ch
      }
    }
    if (chunk) pushLine(chunk)
  }

  for (const paragraph of paragraphs) {
    const normalized = paragraph.replace(/\s+/g, ' ').trim()
    if (!normalized) {
      pushLine('')
      continue
    }

    const words = normalized.split(' ')
    let current = words.shift() ?? ''
    maxLineWidth = Math.max(maxLineWidth, measure(current))

    for (const word of words) {
      const next = current.length ? `${current} ${word}` : word
      if (measure(next) <= width) {
        current = next
        maxLineWidth = Math.max(maxLineWidth, measure(current))
      } else {
        if (current) pushChunks(current)
        if (measure(word) <= width) {
          current = word
          maxLineWidth = Math.max(maxLineWidth, measure(current))
        } else {
          let chunk = ''
          for (const ch of word) {
            const candidate = chunk + ch
            if (measure(candidate) > width && chunk) {
              pushLine(chunk)
              chunk = ch
            } else {
              chunk = candidate
            }
          }
          current = chunk
          maxLineWidth = Math.max(maxLineWidth, measure(current))
        }
      }
    }

    pushChunks(current)
  }

  if (!lines.length) {
    lines.push('')
    maxLineWidth = Math.max(maxLineWidth, measure(''))
  }

  return { lines, maxLineWidth, lineHeight }
}

type LayoutComputation = {
  width: number
  height: number
  contentWidth: number
  contentHeight: number
  lineCount: number
  offsetX: number
  offsetY: number
  lines: string[]
}

const computeLayout = (
  text: string,
  measure: (txt: string) => number,
  fontSize: number,
  baseWidth: number,
  baseHeight: number,
  growDir: GrowDirection,
  padding: number,
  lineHeightMultiplier: number,
): LayoutComputation & { baseW: number; baseH: number } => {
  const lineHeight = fontSize * lineHeightMultiplier
  const minWidth = Math.max(MIN_WIDTH, fontSize * 4)
  const minHeight = Math.max(lineHeight, fontSize + padding)

  const normalizedParagraphs = text.replace(/\r/g, '').split('\n')
  const naturalWidth = Math.max(
    minWidth,
    ...normalizedParagraphs.map((p) => measure(normalizeParagraph(p))),
  )

  const baseW = Math.max(baseWidth || 0, minWidth)
  const baseH = Math.max(baseHeight || 0, minHeight)
  const rightDownGrow = growDir === 'right-down'
  const growLeft = growDir === 'left'
  const growRight = growDir === 'right' || rightDownGrow
  const growUp = growDir === 'up'
  const growDown = growDir === 'down' || rightDownGrow
  const horizontalGrow = growLeft || growRight
  const verticalGrow = growUp || growDown
  const pureHorizontalGrow = horizontalGrow && !verticalGrow

  let width = rightDownGrow ? baseW : Math.max(Math.min(naturalWidth, baseW), minWidth)
  if (pureHorizontalGrow && naturalWidth > width) {
    width = naturalWidth
  }
  let wrapped = wrapTextWithWidth(text, width, measure, lineHeight)
  let height = Math.max(wrapped.lines.length * lineHeight, minHeight)

  if (rightDownGrow) {
    // right-down: wrap in the current box first, then auto-fit proportionally
    // (shrink/expand) while keeping the base aspect ratio.
    const minScale = Math.max(
      minWidth / Math.max(baseW, 1e-6),
      minHeight / Math.max(baseH, 1e-6),
    )
    const evalRightDownScale = (scale: number) => {
      const clampedScale = Math.max(minScale, scale)
      const nextWidth = Math.max(minWidth, baseW * clampedScale)
      const nextWrapped = wrapTextWithWidth(text, nextWidth, measure, lineHeight)
      const requiredHeight = Math.max(nextWrapped.lines.length * lineHeight, minHeight)
      const boxHeight = Math.max(minHeight, baseH * clampedScale)
      const fits = requiredHeight <= boxHeight + 1e-3
      return {
        scale: clampedScale,
        width: nextWidth,
        wrapped: nextWrapped,
        requiredHeight,
        boxHeight,
        fits,
      }
    }

    const baseEval = evalRightDownScale(1)
    if (baseEval.fits) {
      // Auto-shrink: find the smallest proportional box that still fits the wrapped text.
      let lo = minScale
      let hi = 1
      let best = baseEval
      for (let i = 0; i < 12; i++) {
        const mid = (lo + hi) / 2
        const midEval = evalRightDownScale(mid)
        if (midEval.fits) {
          best = midEval
          hi = mid
        } else {
          lo = mid
        }
      }
      width = best.width
      wrapped = best.wrapped
      height = best.boxHeight
    } else {
      // Auto-expand: keep wrapping and grow proportionally until the wrapped text fits.
      let scale = Math.max(1, baseEval.requiredHeight / Math.max(baseH, 1e-6))
      let best = baseEval
      for (let i = 0; i < 8; i++) {
        const nextEval = evalRightDownScale(scale)
        best = nextEval
        const requiredScale = Math.max(1, nextEval.requiredHeight / Math.max(baseH, 1e-6))
        if (requiredScale <= scale + 1e-3) {
          break
        }
        scale = requiredScale
      }
      width = best.width
      wrapped = best.wrapped
      height = Math.max(best.boxHeight, best.requiredHeight)
    }
  } else if (height <= baseH) {
    height = Math.min(height, baseH)
  } else if (horizontalGrow) {
    const maxWidth = width + Math.max(fontSize, baseW)
    let targetWidth = width
    while (wrapped.lines.length * lineHeight > baseH && targetWidth < maxWidth) {
      targetWidth += fontSize
      const next = wrapTextWithWidth(text, targetWidth, measure, lineHeight)
      wrapped = next
    }
    width = Math.max(targetWidth, width)
    height = Math.max(wrapped.lines.length * lineHeight, baseH)
  } else {
    height = Math.max(wrapped.lines.length * lineHeight, baseH)
  }

  const contentWidth = Math.max(wrapped.maxLineWidth, Math.min(width, naturalWidth))
  const contentHeight = Math.max(wrapped.lines.length * lineHeight, lineHeight)

  let offsetX = 0
  let offsetY = 0
  if (width > baseW && horizontalGrow) {
    offsetX = growLeft ? baseW - width : 0
  }
  if (height > baseH && verticalGrow) {
    offsetY = growUp ? baseH - height : 0
  }

  return {
    width,
    height,
    contentWidth,
    contentHeight,
    lineCount: wrapped.lines.length,
    lines: wrapped.lines,
    offsetX,
    offsetY,
    baseW,
    baseH,
  }
}

export const computeTextBoxLayout = (input: TextBoxLayoutInput): TextBoxLayoutOutput => {
  const fontSize = Math.max(8, input.fontSize || 16)
  const padding = Math.max(0, input.padding ?? DEFAULT_PADDING)
  const lineHeightMultiplier = input.lineHeight && input.lineHeight > 0 ? input.lineHeight : DEFAULT_TEXTBOX_LINE_HEIGHT
  const measure = createMeasure(input.fontFamily, fontSize, input.fontWeight)

  const layout = computeLayout(
    input.text || '',
    measure,
    fontSize,
    Math.max(input.baseWidth, padding * 2),
    Math.max(input.baseHeight, fontSize),
    input.growDir,
    padding,
    lineHeightMultiplier,
  )

  return {
    width: layout.width,
    height: layout.height,
    contentWidth: layout.contentWidth,
    contentHeight: layout.contentHeight,
    baseWidth: layout.baseW,
    baseHeight: layout.baseH,
    padding,
    lineHeight: lineHeightMultiplier,
    offsetX: layout.offsetX,
    offsetY: layout.offsetY,
    lineCount: layout.lineCount,
    lines: layout.lines,
    renderedText: layout.lines.join('\n'),
  }
}

export const applyGrowOffset = (
  x: number,
  y: number,
  offsetX: number,
  offsetY: number,
) => ({
  x: x + offsetX,
  y: y + offsetY,
})
