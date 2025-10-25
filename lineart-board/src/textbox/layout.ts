export type GrowDirection = 'down' | 'up' | 'left' | 'right'

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
        if (current) pushLine(current)
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

    pushLine(current)
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
  const horizontalGrow = growDir === 'left' || growDir === 'right'

  let width = Math.max(Math.min(naturalWidth, baseW), minWidth)
  if (horizontalGrow && naturalWidth > width) {
    width = naturalWidth
  }
  let wrapped = wrapTextWithWidth(text, width, measure, lineHeight)
  let height = Math.max(wrapped.lines.length * lineHeight, minHeight)

  if (height <= baseH) {
    height = Math.min(height, baseH)
  } else if (growDir === 'left' || growDir === 'right') {
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
  if (width > baseW && (growDir === 'left' || growDir === 'right')) {
    offsetX = growDir === 'left' ? baseW - width : 0
  }
  if (height > baseH && (growDir === 'up' || growDir === 'down')) {
    offsetY = growDir === 'up' ? baseH - height : 0
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
