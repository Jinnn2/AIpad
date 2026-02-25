const BLOCK_MARKDOWN_RE =
  /(^|\n)\s{0,3}(#{1,6}\s+\S|[-*+]\s+(?:\[[ xX]\]\s+)?\S|\d+\.\s+\S|>\s+\S|```)/m

const INLINE_MARKDOWN_RE =
  /(\*\*[^*\n][^*\n]*\*\*|__[^_\n][^_\n]*__|!\[[^\]\n]*\]\([^)]+\)|\[[^\]\n]+\]\([^)]+\)|`[^`\n]+`)/m

export function looksLikeMarkdownText(value: string): boolean {
  if (!value) return false
  const source = String(value).replace(/\r\n?/g, '\n')
  if (BLOCK_MARKDOWN_RE.test(source)) return true
  return INLINE_MARKDOWN_RE.test(source)
}

export type MarkdownDisplayBlock =
  | { kind: 'blank' }
  | { kind: 'paragraph'; text: string; raw?: string }
  | { kind: 'heading'; text: string; level: number; raw?: string }
  | { kind: 'list-item'; text: string; indent: number; ordered: boolean; index?: number; checked?: boolean; raw?: string }
  | { kind: 'quote'; text: string; depth: number; raw?: string }

export type MarkdownInlineRun = {
  text: string
  bold?: boolean
  italic?: boolean
  code?: boolean
}

const stripInlineMarkdown = (line: string): string => {
  let text = line
  for (let i = 0; i < 4; i++) {
    const prev = text
    const next = prev
      .replace(/!\[([^\]\n]*)\]\([^)]+\)/g, '$1')
      .replace(/\[([^\]\n]+)\]\([^)]+\)/g, '$1')
      .replace(/`([^`\n]+)`/g, '$1')
      .replace(/\*\*([^*\n]+)\*\*/g, '$1')
      .replace(/__([^_\n]+)__/g, '$1')
      .replace(/\*([^*\n]+)\*/g, '$1')
      .replace(/_([^_\n]+)_/g, '$1')
      .replace(/~~([^~\n]+)~~/g, '$1')
    text = next
    if (next === prev) break
  }
  return text.replace(/\\([\\`*_{}\[\]()#+\-.!>])/g, '$1')
}

export function parseMarkdownDisplayBlocks(value: string): MarkdownDisplayBlock[] {
  if (!value) return []
  const source = String(value).replace(/\r\n?/g, '\n')
  const out: MarkdownDisplayBlock[] = []
  let inFence = false
  let paragraphBuf: string[] = []

  const flushParagraph = () => {
    if (!paragraphBuf.length) return
    const joined = paragraphBuf.join(' ').replace(/\s+/g, ' ').trim()
    paragraphBuf = []
    if (joined) out.push({ kind: 'paragraph', text: stripInlineMarkdown(joined), raw: joined })
  }

  for (const rawLine of source.split('\n')) {
    const line = rawLine.replace(/\t/g, '  ')
    const trimmed = line.trim()

    if (/^```/.test(trimmed)) {
      flushParagraph()
      inFence = !inFence
      continue
    }

    if (inFence) {
      flushParagraph()
      const codeText = line.replace(/^\s+/, '')
      out.push(codeText ? { kind: 'paragraph', text: stripInlineMarkdown(codeText) } : { kind: 'blank' })
      continue
    }

    if (!trimmed) {
      flushParagraph()
      if (!out.length || out[out.length - 1]?.kind !== 'blank') out.push({ kind: 'blank' })
      continue
    }

    let m = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/)
    if (m) {
      flushParagraph()
      out.push({
        kind: 'heading',
        level: Math.min(6, Math.max(1, (m[1] || '').length)),
        text: stripInlineMarkdown(m[2] || '').trim(),
        raw: String(m[2] || '').trim(),
      })
      continue
    }

    m = line.match(/^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$/)
    if (m) {
      flushParagraph()
      out.push({
        kind: 'list-item',
        ordered: false,
        checked: String(m[2] || '').toLowerCase() === 'x',
        indent: Math.min(4, Math.floor(((m[1] || '').length || 0) / 2)),
        text: stripInlineMarkdown(m[3] || '').trim(),
        raw: String(m[3] || '').trim(),
      })
      continue
    }

    m = line.match(/^(\s*)[-*+]\s+(.*)$/)
    if (m) {
      flushParagraph()
      out.push({
        kind: 'list-item',
        ordered: false,
        indent: Math.min(4, Math.floor(((m[1] || '').length || 0) / 2)),
        text: stripInlineMarkdown(m[2] || '').trim(),
        raw: String(m[2] || '').trim(),
      })
      continue
    }

    m = line.match(/^(\s*)(\d+)\.\s+(.*)$/)
    if (m) {
      flushParagraph()
      out.push({
        kind: 'list-item',
        ordered: true,
        index: Number(m[2]),
        indent: Math.min(4, Math.floor(((m[1] || '').length || 0) / 2)),
        text: stripInlineMarkdown(m[3] || '').trim(),
        raw: String(m[3] || '').trim(),
      })
      continue
    }

    m = line.match(/^(\s*)(>+)\s?(.*)$/)
    if (m) {
      flushParagraph()
      out.push({
        kind: 'quote',
        depth: Math.min(3, (m[2] || '').length || 1),
        text: stripInlineMarkdown(m[3] || '').trim(),
        raw: String(m[3] || '').trim(),
      })
      continue
    }

    paragraphBuf.push(line)
  }

  flushParagraph()
  while (out.length && out[out.length - 1]?.kind === 'blank') out.pop()
  return out
}

export function markdownToPlainText(value: string): string {
  if (!value) return ''
  const source = String(value).replace(/\r\n?/g, '\n')
  const out: string[] = []
  let inFence = false

  for (const rawLine of source.split('\n')) {
    const line = rawLine.replace(/\t/g, '  ')
    const trimmed = line.trim()
    if (/^```/.test(trimmed)) {
      inFence = !inFence
      continue
    }
    if (inFence) {
      out.push(stripInlineMarkdown(line))
      continue
    }

    let m = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/)
    if (m) {
      out.push(stripInlineMarkdown(m[2] || '').trim())
      continue
    }

    m = line.match(/^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$/)
    if (m) {
      const indent = ' '.repeat(Math.min((m[1] || '').length, 6))
      const mark = String(m[2] || ' ').toLowerCase() === 'x' ? '[x]' : '[ ]'
      out.push(`${indent}${mark} ${stripInlineMarkdown(m[3] || '').trim()}`)
      continue
    }

    m = line.match(/^(\s*)[-*+]\s+(.*)$/)
    if (m) {
      const indent = ' '.repeat(Math.min((m[1] || '').length, 6))
      out.push(`${indent}- ${stripInlineMarkdown(m[2] || '').trim()}`)
      continue
    }

    m = line.match(/^(\s*)(\d+)\.\s+(.*)$/)
    if (m) {
      const indent = ' '.repeat(Math.min((m[1] || '').length, 6))
      out.push(`${indent}${m[2]}. ${stripInlineMarkdown(m[3] || '').trim()}`)
      continue
    }

    m = line.match(/^\s*>\s?(.*)$/)
    if (m && trimmed.startsWith('>')) {
      out.push(`> ${stripInlineMarkdown(m[1] || '').trim()}`.trimEnd())
      continue
    }

    out.push(stripInlineMarkdown(line))
  }

  return out.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd()
}

export function hasInlineMarkdownStyle(value: string): boolean {
  if (!value) return false
  return INLINE_MARKDOWN_RE.test(String(value))
}

export function parseMarkdownInlineRuns(value: string): MarkdownInlineRun[] {
  const source = String(value ?? '')
  if (!source) return []
  const pattern =
    /!\[([^\]\n]*)\]\([^)]+\)|\[([^\]\n]+)\]\([^)]+\)|`([^`\n]+)`|\*\*([^*\n]+)\*\*|__([^_\n]+)__|\*([^*\n]+)\*|_([^_\n]+)_/g
  const runs: MarkdownInlineRun[] = []
  let cursor = 0

  const pushPlain = (txt: string) => {
    if (!txt) return
    const clean = txt.replace(/\\([\\`*_{}\[\]()#+\-.!>])/g, '$1')
    if (!clean) return
    const prev = runs[runs.length - 1]
    if (prev && !prev.bold && !prev.italic && !prev.code) {
      prev.text += clean
    } else {
      runs.push({ text: clean })
    }
  }

  for (let match = pattern.exec(source); match; match = pattern.exec(source)) {
    const idx = match.index ?? 0
    if (idx > cursor) pushPlain(source.slice(cursor, idx))
    let run: MarkdownInlineRun | null = null
    if (typeof match[1] === 'string') run = { text: match[1] }
    else if (typeof match[2] === 'string') run = { text: match[2] }
    else if (typeof match[3] === 'string') run = { text: match[3], code: true }
    else if (typeof match[4] === 'string' || typeof match[5] === 'string') run = { text: String(match[4] ?? match[5] ?? ''), bold: true }
    else if (typeof match[6] === 'string' || typeof match[7] === 'string') run = { text: String(match[6] ?? match[7] ?? ''), italic: true }
    if (run && run.text) {
      runs.push(run)
    }
    cursor = idx + String(match[0] ?? '').length
  }
  if (cursor < source.length) pushPlain(source.slice(cursor))

  return runs.length ? runs : [{ text: stripInlineMarkdown(source) }]
}

export function renderMarkdownToCanvasText(value: string): string {
  if (!value) return ''
  const source = String(value).replace(/\r\n?/g, '\n')
  return looksLikeMarkdownText(source) ? markdownToPlainText(source) : source
}
