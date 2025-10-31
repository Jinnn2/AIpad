import React, { useCallback, useMemo, useRef, useState } from 'react'
import {
  Stage,
  Layer,
  Group,
  Line as KLine,
  Rect as KRect,
  Ellipse as KEllipse,
  Text as KText,
  Label as KLabel,
  Tag as KTag,
} from 'react-konva'
import type { AIStrokePayload, AIStrokeV11, ColorName, PromptMode } from './ai/types'
import { normalizeAIStrokePayload, validateAIStrokePayload, COLORS } from './ai/normalize'
import type { ShapeDraft } from './ai/plan'
import { planDrafts } from './ai/plan'
import { chaikin, resampleEvenly, geomMaxDeviationFromChord, mergeCollinear, draftToAIStroke } from './ai/draw'
import { TopToolbar, SidePanel, BottomPanel, type AIFeedEntry } from './LineArtUI'
import { computeTextBoxLayout, DEFAULT_TEXTBOX_LINE_HEIGHT } from './textbox/layout'
/**
 * LineArtBoard renders a Konva-based workspace with:
 * - Top toolbar for grid/snap toggles, brush settings, import/export helpers.
 * - AI v1.1 plumbing: validate, normalize, plan, preview, accept, dismiss.
 * - Freehand pen smoothing, even resampling, and stacking for undo/redo.
 * - Vector eraser with radius-based masking that keeps history in sync.
 * - Export helpers for human strokes and minimal undo snapshots.
 *
 * Notes:
 * 1) The module passes current TypeScript checks, so it is safe to replace as a whole.
 * 2) Stage owns all pointer events; previews and committed shapes render on separate layers.
 */
/* ---------- 0) Types for AI protocol v1.1 ---------- */
/* ---------- 1) Validation & normalization (no external deps) ---------- */
const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v))
// ----- Zoom configuration -----
const ZOOM_MIN = 0.2
const ZOOM_MAX = 8
const ZOOM_STEP = 1.06 // zoom factor per wheel tick (>1 zooms in, <1 zooms out)
const FALLBACK_API_BASE = 'http://localhost:8000'
const API_BASE = (() => {
  try {
    const base = (import.meta as any)?.env?.VITE_API_BASE ?? ''
    if (!base) return ''
    return String(base).replace(/\/$/, '')
  } catch {
    return ''
  }
})()
const withBase = (base: string, path: string) => {
  if (!base) return path
  if (/^https?:/i.test(path)) return path
  return `${base}${path.startsWith('/') ? path : `/${path}`}`
}
const apiFetch = async (path: string, init?: RequestInit) => {
  const isAbsolute = typeof path === 'string' && /^https?:/i.test(path)
  const request = (url: string) => fetch(url, init)
  if (API_BASE) {
    const primary = withBase(API_BASE, path)
    try {
      return await request(primary)
    } catch (err) {
      if (!isAbsolute) {
        const fallback = withBase(FALLBACK_API_BASE, path)
        try {
          return await request(fallback)
        } catch (err2) {
          console.warn('[apiFetch] fallback failed', err2)
        }
      }
      throw err
    }
  }
  if (!isAbsolute) {
    const fallback = withBase(FALLBACK_API_BASE, path)
    return request(fallback)
  }
  return request(path)
}
const SIZE_TO_WIDTH: Record<'s'|'m'|'l'|'xl', number> = { s: 2, m: 4, l: 6, xl: 10 }
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
    default: return c
  }
}
const INPUT_BASE: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: 10,
  border: '1px solid #d1d5db',
  background: '#fff',
}
const BUTTON_BASE: React.CSSProperties = {
  padding: '6px 12px',
  borderRadius: 999,
  border: '1px solid #d1d5db',
  background: '#fff',
  cursor: 'pointer',
}
const TEXT_LINE_HEIGHT = DEFAULT_TEXTBOX_LINE_HEIGHT
const BLOCK_COLOR_PALETTE = [
  '#2563eb',
  '#ec4899',
  '#f97316',
  '#22c55e',
  '#0ea5e9',
  '#a855f7',
  '#f59e0b',
  '#ef4444',
  '#14b8a6',
  '#94a3b8',
] as const
const hexToRgba = (hex: string, alpha: number) => {
  const normalized = hex.replace('#', '')
  if (normalized.length !== 6) return `rgba(148, 163, 184, ${alpha})`
  const bigint = parseInt(normalized, 16)
  const r = (bigint >> 16) & 255
  const g = (bigint >> 8) & 255
  const b = bigint & 255
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
// Preview entries keep drafts grouped by payload id
type PreviewEntry = { payloadId: string; drafts: ShapeDraft[] }
type TextGrowDir = 'down' | 'up' | 'left' | 'right'
type TextSettings = {
  fontFamily: string
  fontSize: number
  fontWeight: string
  growDir: TextGrowDir
}
type TextEditorState = {
  id: string
  x: number
  y: number
  w: number
  h: number
  text: string
  summary: string
  fontFamily: string
  fontSize: number
  fontWeight: string
  growDir: TextGrowDir
  color: ColorName
  opacity: number
  isEditing: boolean
  originalShapeId?: string
  pendingCompletion?: string | null
  completing?: boolean
}
export default function LineArtBoard() {
  // Canvas size; swap to ResizeObserver for responsive layout
  const [size] = useState({ width: window.innerWidth, height: window.innerHeight })
  const askAIRef = useRef<null | (() => void)>(null)
  const stageRef = useRef<any>(null)
  // Top toolbar state
  const [showGrid, setShowGrid] = useState(true)
  const [snap, setSnap] = useState(true)
  // Toggle smoothing sharp turns into curves after mouseup
  const [curveTurns, setCurveTurns] = useState(true)
  // Committed shapes that have been accepted
  const [shapes, setShapes] = useState<ShapeDraft[]>([])
  const shapeById = useMemo(() => {
    const index = new Map<string, ShapeDraft>()
    for (const shape of shapes) {
      index.set(shape.id, shape)
    }
    return index
  }, [shapes])
  const shapesById = useMemo(() => {
    const map: Record<string, ShapeDraft> = {}
    for (const shape of shapes) map[shape.id] = shape
    return map
  }, [shapes])
  const clearCompletionPreview = useCallback((id: string | null | undefined) => {
    if (!id) return
    setCompletionPreviews(prev => {
      if (!(id in prev)) return prev
      const { [id]: _omit, ...rest } = prev
      return rest
    })
  }, [])
  // Prompt mode for AI requests
  const [mode, setMode] = useState<PromptMode>("full");
  // Brush configuration aligns with the AI protocol style definition
  const [brushSize, setBrushSize] = useState<'s'|'m'|'l'|'xl'>('m')
  const [brushColor, setBrushColor] = useState<ColorName>('black')
  const currentBrush = useMemo(() => ({
    tool: 'pen' as const,
    style: { size: brushSize, color: brushColor as ColorName, opacity: 1 },
    meta: { author: 'human' } as Record<string, any>,
  }), [brushSize, brushColor])
  // Hint text forwarded to /suggest
  const [hint, setHint] = useState<string>('Work as a noting assistant to draw or write.')
  // AI generation scale caps point count and informs upload density
  const [aiScale, setAiScale] = useState<number>(16) // adjustable 4-64, defaults to 16
  // Live drawing state with raw float coordinates (world space)
  const [isDrawing, setIsDrawing] = useState(false)
  const [rawPoints, setRawPoints] = useState<number[]>([])  // [x0,y0,x1,y1,...] world coordinates
  // Stack of human strokes for erasing and undo/redo
  type DrawStackEntry = { ai: AIStrokeV11; draft: ShapeDraft }
  const [drawStack, setDrawStack] = useState<DrawStackEntry[]>([])
  // -------- Tool modes: pen / eraser / ellipse / hand / text --------
  const [toolMode, setToolMode] = useState<'pen' | 'eraser' | 'ellipse' | 'hand' | 'text' | 'select'>('pen')
  const [eraserRadius, setEraserRadius] = useState<number>(14) // pixels
  const [boxDraft, setBoxDraft] = useState<ShapeDraft | null>(null)
const [textSettings, setTextSettings] = useState<TextSettings>({
    fontFamily: 'sans-serif',
    fontSize: 18,
    fontWeight: '400',
    growDir: 'down',
  })
  const [textEditor, setTextEditor] = useState<TextEditorState | null>(null)
  const [selectedShapeId, setSelectedShapeId] = useState<string | null>(null)
  const [completionPreviews, setCompletionPreviews] = useState<Record<string, string>>({})
  const updateTextSettings = useCallback((patch: Partial<TextSettings>) => {
    setTextSettings((prev) => ({ ...prev, ...patch }))
  }, [])
  // Visual cursor for the eraser radius
  const [eraserCursor, setEraserCursor] = useState<{x:number;y:number}|null>(null)
  // Only push history once per erase gesture (pointer down -> up)
  const eraseGestureStarted = useRef(false)
  const selectDragRef = useRef<{ id: string; offsetX: number; offsetY: number; startX: number; startY: number; moved: boolean } | null>(null)
  // Viewport transform for the infinite canvas
  const [view, setView] = useState<{x:number; y:number; scale:number}>({ x: 0, y: 0, scale: 1 })
  const [isPanning, setIsPanning] = useState(false)
  // Wheel zoom keeps the cursor anchored in world space
  const onWheelZoom = useCallback((e: any) => {
    // Konva-proxied native wheel event
    const evt: WheelEvent = e?.evt
    if (!evt) return
    // Prevent page scroll and browser zoom
    evt.preventDefault()
    // Optional: allow Ctrl+wheel to fall back to browser zoom
    // if (evt.ctrlKey) return
    const stage = e.target.getStage?.()
    const ptr = stage?.getPointerPosition?.()
    if (!ptr) return
    // Current scale and new target direction
    const oldScale = view.scale
    // Wheel delta decides zoom in/out
    const direction = evt.deltaY > 0 ? -1 : 1
    const scaleBy = direction > 0 ? ZOOM_STEP : 1 / ZOOM_STEP
    let newScale = oldScale * scaleBy
    newScale = clamp(newScale, ZOOM_MIN, ZOOM_MAX)
    // Mouse coordinates translated back to world space before zoom
    const worldX = (ptr.x - view.x) / oldScale
    const worldY = (ptr.y - view.y) / oldScale
    // Shift view so the cursor points to the same world coordinate post-zoom
    const newX = ptr.x - worldX * newScale
    const newY = ptr.y - worldY * newScale
    setView(v => ({ ...v, x: newX, y: newY, scale: newScale }))
  }, [view])
  // Convert screen coordinates (mouse) to world coordinates
  const screenToWorld = useCallback((sx:number, sy:number) => {
    return { x: (sx - view.x) / view.scale, y: (sy - view.y) / view.scale }
  }, [view])
  const worldToScreen = useCallback((wx:number, wy:number) => {
    return { x: wx * view.scale + view.x, y: wy * view.scale + view.y }
  }, [view])
  // ----- Snapshot stage canvas to JPEG/PNG Base64 (max edge size configurable) -----
  const snapshotCanvas = useCallback(async (
    maxSize = 768,
    mime: "image/jpeg" | "image/png" = "image/jpeg",
    quality = 0.6
  ): Promise<{ data: string|null; w: number; h: number; mime: string }> => {
    const stage = stageRef.current
    if (!stage) return { data: null, w: 0, h: 0, mime }
    // Raw dataURL generated by Konva (renders stage into its own <canvas>)
    const srcUri: string = stage.toDataURL({ pixelRatio: 1 })
    // Downscale via an offscreen <canvas> before encoding
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
  // Store AI previews grouped by payload id
  const [previews, setPreviews] = useState<Record<string, PreviewEntry>>({})
  const previewEntries = useMemo(() => Object.values(previews), [previews])
  const activeEditTargets = useMemo(() => {
    const ids = new Set<string>()
    for (const entry of previewEntries) {
      for (const draft of entry.drafts) {
        if (draft.kind === 'edit') {
          const meta = draft.meta ?? {}
          const targetId = (draft.targetId ?? meta.targetId ?? meta.target ?? meta.id) as string | undefined
          if (targetId) ids.add(String(targetId))
        }
      }
    }
    return ids
  }, [previewEntries])
  const [currentPayloadId, setCurrentPayloadId] = useState<string | null>(null)
  // AI feed keeps the latest 50 suggestion entries
  const [aiFeed, setAiFeed] = useState<AIFeedEntry[]>([])
  // Session identifiers from backend; lastSentIndex tracks delta uploads
  const [sid, setSid] = useState<string | null>(null)
  const [visionVersion, setVisionVersion] = useState<number>(2.0)
  const lastSentIndexRef = useRef<number>(0)
  // Debounce timer handle for session sync
  const syncTimerRef = useRef<number | null>(null)
  // Helper for three-decimal rounding to shrink payloads (backend also clamps)
  const round3 = (v:number) => Math.round(v * 1000) / 1000
  // Pack drawStack.ai (absolute coordinates) into protocol-friendly strokes
  const packAllStrokes = useCallback(() => {
    return drawStack.map(e => ({
      ...e.ai,
      points: e.ai.points.map(([x,y,t,p]) => [round3(x), round3(y), t, p]) as any,
    }))
  }, [drawStack])
  // -------- Undo/redo snapshot stacks --------
  type Snapshot = { shapes: ShapeDraft[]; drawStack: DrawStackEntry[] }
  const [past, setPast] = useState<Snapshot[]>([])
  const [future, setFuture] = useState<Snapshot[]>([])
  const pushHistory = useCallback((snap?: Snapshot) => {
    setPast(p => [...p, snap ?? { shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }])
    setFuture([]) // Clear redo branch after a new action
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
  // ----- Auto-sync drawStack to backend session every 300ms -----
  const syncSession = useCallback(async (curSid: string) => {
    try {
      const strokes = packAllStrokes()
      const res = await apiFetch('/session/sync', {
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
  // ----- Auto-complete toggle & countdown (5s) -----
  const [autoComplete, setAutoComplete] = useState<boolean>(false)
  const [autoCountdown, setAutoCountdown] = useState<number|null>(null)
  const [autoMaintain, setAutoMaintain] = useState<boolean>(false)
  const [autoMaintainPending, setAutoMaintainPending] = useState<boolean>(false)
  const [graphSnapshot, setGraphSnapshot] = useState<GraphSnapshot | null>(null)
  const [graphInspectorVisible, setGraphInspectorVisible] = useState<boolean>(false)
  const [promoteGroupPending, setPromoteGroupPending] = useState<string | null>(null)
  const blockColorMapRef = useRef<Record<string, string>>({})
  const blockColorMap = useMemo(() => {
    const blocks = graphSnapshot?.blocks ?? []
    if (blocks.length === 0) {
      blockColorMapRef.current = {}
      return {}
    }
    const existing = blockColorMapRef.current
    const used = new Set<string>()
    const next: Record<string, string> = {}
    for (const block of blocks) {
      const color = existing[block.blockId]
      if (color) {
        next[block.blockId] = color
        used.add(color)
      }
    }
    let paletteIndex = 0
    for (const block of blocks) {
      if (next[block.blockId]) continue
      let candidate = BLOCK_COLOR_PALETTE[paletteIndex % BLOCK_COLOR_PALETTE.length]
      while (used.has(candidate)) {
        paletteIndex += 1
        candidate = BLOCK_COLOR_PALETTE[paletteIndex % BLOCK_COLOR_PALETTE.length]
      }
      next[block.blockId] = candidate
      used.add(candidate)
      paletteIndex += 1
    }
    blockColorMapRef.current = next
    return next
  }, [graphSnapshot?.blocks])
  const autoTimerRef = useRef<number | ReturnType<typeof setTimeout> | null>(null)
  const autoTickerRef = useRef<number | ReturnType<typeof setInterval> | null>(null)
  const graphPollRef = useRef<number | null>(null)
  const hasActivePreview = useMemo(() => {
    // Previews exist when at least one AI payload is staged
    return Object.keys(previews || {}).length > 0
  }, [previews])
  const clearAutoTimer = useCallback(() => {
    if (autoTimerRef.current) { clearTimeout(autoTimerRef.current as any); autoTimerRef.current = null }
    if (autoTickerRef.current) { clearInterval(autoTickerRef.current as any); autoTickerRef.current = null }
    setAutoCountdown(null)
  }, [])
  const toggleGrid = useCallback(() => setShowGrid((value) => !value), [setShowGrid])
  const toggleSnap = useCallback(() => setSnap((value) => !value), [setSnap])
  const toggleCurveTurns = useCallback(() => setCurveTurns((value) => !value), [setCurveTurns])
  const handleAutoCompleteToggle = useCallback((enabled: boolean) => {
    setAutoComplete(enabled)
    clearAutoTimer()
  }, [clearAutoTimer])
  const fetchGraphSnapshot = useCallback(async (currentSid: string) => {
    try {
      const res = await apiFetch(`/graph/state?sid=${encodeURIComponent(currentSid)}`)
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`graph state failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = (await res.json()) as GraphSnapshot
      setGraphSnapshot({
        blocks: Array.isArray(data?.blocks) ? data.blocks : [],
        fragments: Array.isArray(data?.fragments) ? data.fragments : [],
        groups: Array.isArray(data?.groups) ? data.groups : [],
      })
    } catch (err) {
      console.warn('[graph] snapshot error:', err)
    }
  }, [])
  const updateAutoMaintain = useCallback(async (nextEnabled: boolean, opts?: { silent?: boolean }) => {
    const quiet = opts?.silent ?? false
    if (!quiet && autoMaintainPending) return
    try {
      if (!quiet) setAutoMaintainPending(true)
      let curSid = sid
      if (nextEnabled) {
        if (!curSid) {
          const initRes = await apiFetch('/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'light_helper', init_goal: hint }),
          })
          if (!initRes.ok) {
            const txt = await initRes.text().catch(() => '')
            throw new Error(`init session failed: ${initRes.status} ${initRes.statusText}${txt ? `\n${txt}` : ''}`)
          }
          const j0 = await initRes.json()
          curSid = String(j0.sid)
          setSid(curSid)
          lastSentIndexRef.current = 0
        }
      } else {
        if (!curSid) {
          setAutoMaintain(false)
          return
        }
      }
      if (!curSid) return
      const payload: any = { sid: curSid, enabled: nextEnabled }
      if (nextEnabled) {
        payload.canvas_size = [size.width, size.height]
        payload.strokes = packAllStrokes()
      }
      const res = await apiFetch('/graph/auto-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`graph auto-mode failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = await res.json().catch(() => ({}))
      const enabled = Boolean((data as any)?.enabled)
      if (enabled && nextEnabled) {
        setAutoMaintain(true)
        if (curSid) await fetchGraphSnapshot(curSid)
      } else {
        setAutoMaintain(false)
        setGraphSnapshot(null)
      }
    } catch (err: any) {
      console.warn('[auto-maintain] error:', err)
      if (!quiet) {
        alert('自动维护切换失败：\n' + (err?.message || String(err)))
      }
      if (!nextEnabled) {
        setAutoMaintain(false)
      }
    } finally {
      if (!quiet) setAutoMaintainPending(false)
      if (!nextEnabled) {
        setAutoMaintain(false)
        setGraphSnapshot(null)
      }
    }
  }, [autoMaintainPending, sid, size.width, size.height, packAllStrokes, hint, fetchGraphSnapshot])
  const handleAutoMaintainToggle = useCallback(() => {
    void updateAutoMaintain(!autoMaintain)
  }, [autoMaintain, updateAutoMaintain])
  const graphBlockCards = useMemo<GraphBlockCard[]>(() => {
    if (!graphSnapshot) return []
    const fragments = graphSnapshot.fragments ?? []
    const fragmentMap = new Map<string, (typeof fragments)[number]>()
    for (const frag of fragments) {
      fragmentMap.set(frag.id, frag)
    }
    const resolveShapeBBox = (shape?: ShapeDraft | null): [number, number, number, number] | null => {
      if (!shape) return null
      const baseX = Number(shape.x) || 0
      const baseY = Number(shape.y) || 0
      if (Number.isFinite(shape.w) && Number.isFinite(shape.h)) {
        const w = Math.max(1, Number(shape.w))
        const h = Math.max(1, Number(shape.h))
        return [baseX, baseY, baseX + w, baseY + h]
      }
      if (shape.points && shape.points.length > 0) {
        let minX = Number.POSITIVE_INFINITY
        let minY = Number.POSITIVE_INFINITY
        let maxX = Number.NEGATIVE_INFINITY
        let maxY = Number.NEGATIVE_INFINITY
        for (const pt of shape.points) {
          const px = baseX + (Number(pt.x) || 0)
          const py = baseY + (Number(pt.y) || 0)
          if (px < minX) minX = px
          if (py < minY) minY = py
          if (px > maxX) maxX = px
          if (py > maxY) maxY = py
        }
        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
          return null
        }
        return [minX, minY, maxX, maxY]
      }
      return null
    }
    const mergeBBox = (
      current: [number, number, number, number] | null,
      next: [number, number, number, number] | null | undefined,
    ): [number, number, number, number] | null => {
      if (!next) return current
      if (!current) return [...next] as [number, number, number, number]
      const [cx0, cy0, cx1, cy1] = current
      const [nx0, ny0, nx1, ny1] = next
      return [
        Math.min(cx0, nx0),
        Math.min(cy0, ny0),
        Math.max(cx1, nx1),
        Math.max(cy1, ny1),
      ]
    }
    return (graphSnapshot.blocks ?? []).map((block) => {
      const candidateIds = new Set<string>()
      for (const id of block.contents ?? []) candidateIds.add(id)
      for (const frag of fragments) {
        if (frag.blockId === block.blockId) candidateIds.add(frag.id)
      }
      const blockColor = blockColorMap[block.blockId] ?? '#94a3b8'
      const entries: GraphBlockCardFragment[] = []
      let blockBBox: [number, number, number, number] | null = null
      candidateIds.forEach((fragId) => {
        const frag = fragmentMap.get(fragId)
        if (!frag) return
        const lowerType = String(frag.type || '').toLowerCase()
        const shape = shapeById.get(frag.id) ?? null
        const shapeBBox = resolveShapeBBox(shape)
        const fragBBox = frag.bbox && frag.bbox.length === 4 ? frag.bbox as [number, number, number, number] : shapeBBox
        const rawText = (frag.text ?? shape?.summary ?? shape?.text ?? '').toString().trim()
        const summary =
          rawText.length > 120 ? `${rawText.slice(0, 120)}…` : (rawText || '(无摘要)')
        entries.push({
          id: frag.id,
          type: lowerType || 'unknown',
          text: summary,
          bbox: fragBBox ?? null,
        })
        blockBBox = mergeBBox(blockBBox, fragBBox ?? shapeBBox ?? null)
      })
      return {
        blockId: block.blockId,
        label: block.label,
        summary: block.summary,
        updatedAt: block.updatedAt,
        color: blockColor,
        fragments: entries,
        bbox: blockBBox,
      }
    })
  }, [graphSnapshot, blockColorMap, shapeById])
  const focusOnBBox = useCallback((bbox: [number, number, number, number] | null | undefined) => {
    if (!bbox) return
    const [x0, y0, x1, y1] = bbox
    if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) return
    setView((prev) => {
      const width = Math.max(40, x1 - x0)
      const height = Math.max(40, y1 - y0)
      const margin = 180
      const scaleX = (size.width - margin) / width
      const scaleY = (size.height - margin) / height
      const fitScale = clamp(Math.min(scaleX, scaleY), ZOOM_MIN, ZOOM_MAX)
      const currentScale = prev.scale
      let nextScale = clamp(fitScale, ZOOM_MIN, ZOOM_MAX)
      if (fitScale > currentScale) {
        nextScale = clamp(Math.min(fitScale, currentScale * 1.8), ZOOM_MIN, ZOOM_MAX)
      }
      const centerX = x0 + width / 2
      const centerY = y0 + height / 2
      const newX = size.width / 2 - centerX * nextScale
      const newY = size.height / 2 - centerY * nextScale
      return {
        x: newX,
        y: newY,
        scale: nextScale,
      }
    })
  }, [size.height, size.width])
  const focusOnFragment = useCallback((fragmentId: string) => {
    if (!fragmentId) return
    for (const block of graphBlockCards) {
      const frag = block.fragments.find((f) => f.id === fragmentId)
      if (frag) {
        focusOnBBox(frag.bbox ?? block.bbox ?? null)
        return
      }
    }
  }, [graphBlockCards, focusOnBBox])
  const focusOnBlock = useCallback((blockId: string) => {
    if (!blockId) return
    const target = graphBlockCards.find((block) => block.blockId === blockId)
    focusOnBBox(target?.bbox ?? null)
  }, [graphBlockCards, focusOnBBox])
  const showGraphHighlights = graphInspectorVisible && autoMaintain && graphBlockCards.some(
    (block) => (block.fragments && block.fragments.length > 0) || !!block.bbox,
  )
  const promoteGroup = useCallback(async (groupId: string) => {
    if (!sid) {
      alert('需要先初始化会话（点击 Ask AI 一次即可）');
      return
    }
    setPromoteGroupPending(groupId)
    try {
      const res = await apiFetch('/graph/promote-group', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sid, group_id: groupId }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error('promote group failed: ' + res.status + ' ' + res.statusText + (txt ? '\n' + txt : ''))
      }
      await fetchGraphSnapshot(sid)
    } catch (err: any) {
      console.warn('[graph] promote error:', err)
      alert('Promote group 失败:\n' + (err?.message || String(err)))
    } finally {
      setPromoteGroupPending((prev) => (prev === groupId ? null : prev))
    }
  }, [sid, fetchGraphSnapshot])
  const toggleGraphInspector = useCallback(() => {
    setGraphInspectorVisible((prev) => !prev)
  }, [])
  const cycleMode = useCallback(() => {
    setMode((current) => (current === 'light' ? 'full' : current === 'full' ? 'vision' : 'light'))
  }, [setMode])
  const canUndo = past.length > 0
  const canRedo = future.length > 0
const stageCursor = toolMode === 'hand'
  ? (isPanning ? 'grabbing' : 'grab')
  : toolMode === 'select'
    ? 'pointer'
    : 'default'
  // Export/import committed shapes as JSON
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
  // Export human strokes as AI v1.1 payload (useful for /suggest testing)
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
  // Manual QA helper: stash raw AI JSON into localStorage
  const applyAIStub = useCallback(() => {
    const raw = prompt('Paste AI suggestions JSON (v1.1)')
    if (!raw) return
    try {
      const payload = JSON.parse(raw) as AIStrokePayload
      localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload))
      alert('Saved to localStorage.ai_suggestions_v1')
    } catch { alert('Invalid JSON') }
  }, [])
  // Drop previews when the committed shape count changes
  React.useEffect(() => {
    if (Object.keys(previews).length > 0) {
      setPreviews({})
      setCurrentPayloadId(null)
    }
  }, [shapes.length])
  // Preview pipeline: localStorage -> validate -> normalize -> plan -> store
  const previewAI = useCallback(() => {
    const raw = localStorage.getItem('ai_suggestions_v1')
    if (!raw) { alert('No ai_suggestions_v1 in localStorage'); return }
    try {
      const obj = JSON.parse(raw) as AIStrokePayload
      const items = (obj.strokes||[]).map(s=>({ id: s.id, desc: (s.meta as any)?.desc }))
      setAiFeed(prev=>([{ payloadId: 'local_'+Date.now().toString(36), time: Date.now(), items }, ...prev].slice(0,50)))
      const v = validateAIStrokePayload(obj)
      if (!v.ok || !v.payload) { alert('Invalid payload: ' + v.errors.join('; ')); return }
      const norm = normalizeAIStrokePayload(v.payload)
      const drafts = planDrafts(norm)
      setPreviews(prev => ({ ...prev, [norm.payloadId]: { payloadId: norm.payloadId, drafts } }))
      setCurrentPayloadId(norm.payloadId)
      clearAutoTimer()
      alert(`Preview created: ${drafts.length} shapes\nPayloadId: ${norm.payloadId}`)
    } catch { alert('Invalid JSON in localStorage') }
  }, [clearAutoTimer])
  const noteUserAction = useCallback((opts?: { forceStart?: boolean }) => {
    const forceStart = !!opts?.forceStart
    // Only handles countdown start/reset; preview cleanup happens elsewhere
    // Start when enabled and preview absent (or explicitly forced)
    if (autoComplete && (!hasActivePreview || forceStart)) {
      clearAutoTimer()
      setAutoCountdown(5)
      // Update visible countdown every second
      autoTickerRef.current = setInterval(() => {
        setAutoCountdown((sec) => (sec == null ? null : Math.max(0, sec - 1)))
      }, 1000)
      // Trigger askAI after 5 seconds
      autoTimerRef.current = setTimeout(() => {
        clearAutoTimer()
        // Equivalent to pressing the Ask AI button
        askAIRef.current && askAIRef.current()
      }, 5000)
    } else {
      // Otherwise ensure all timers are cleared
      clearAutoTimer()
    }
  }, [autoComplete, hasActivePreview, clearAutoTimer])
  const updateTextEditorState = useCallback((patch: Partial<TextEditorState>) => {
    setTextEditor((prev) => (prev ? { ...prev, ...patch } : prev))
  }, [])
  const triggerCompletion = useCallback(async (targetId: string, baseText: string) => {
    if (!baseText.trim()) return
    setTextEditor(prev => (prev && prev.id === targetId ? { ...prev, completing: true, pendingCompletion: null } : prev))
    try {
      const res = await apiFetch('/completion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: baseText }),
      })
      if (!res.ok) {
        const errText = await res.text().catch(() => 'completion failed')
        throw new Error(errText)
      }
      const data = await res.json()
      const completionText = String(data?.completion ?? '').trim()
      if (!completionText) throw new Error('empty completion')
      setCompletionPreviews(prev => ({ ...prev, [targetId]: completionText }))
      setTextEditor(prev => (prev && prev.id === targetId
        ? { ...prev, text: prev.text + completionText, pendingCompletion: completionText, completing: false }
        : prev))
    } catch (err) {
      console.error(err)
      setTextEditor(prev => (prev && prev.id === targetId ? { ...prev, completing: false } : prev))
      alert('补全失败，请稍后重试')
    }
  }, [])
  const handleEditorTextChange = useCallback((value: string) => {
    if (!textEditor) return
    const hasTrigger = value.includes(':::')
    const sanitized = hasTrigger ? value.replace(/:::/g, '') : value
    clearCompletionPreview(textEditor.id)
    setTextEditor(prev => (prev && prev.id === textEditor.id ? { ...prev, text: sanitized, pendingCompletion: null } : prev))
    if (hasTrigger) triggerCompletion(textEditor.id, sanitized)
  }, [textEditor, clearCompletionPreview, triggerCompletion])
  const cancelTextEditor = useCallback(() => {
    const targetId = textEditor?.id
    setTextEditor((prev) => {
      if (prev?.isEditing) {
        setSelectedShapeId(prev.id)
      }
      return null
    })
    if (targetId) clearCompletionPreview(targetId)
  }, [textEditor, setSelectedShapeId, clearCompletionPreview])
  const commitTextEditor = useCallback(() => {
    if (!textEditor) return
    const content = textEditor.text.replace(/\s+$/g, '')
    if (!content.trim()) {
      alert('文字内容不能为空')
      return
    }
    const summary = textEditor.summary.trim().slice(0, 30)
    const fallbackWidth = Math.max(240, textEditor.fontSize * 10)
    const fallbackHeight = Math.max(160, textEditor.fontSize * 4)
    const baseWidth = textEditor.w > 0 ? Math.max(textEditor.w, 80) : fallbackWidth
    const baseHeight = textEditor.h > 0 ? Math.max(textEditor.h, Math.round(textEditor.fontSize * 1.6)) : fallbackHeight
    const layout = computeTextBoxLayout({
      text: content,
      fontFamily: textEditor.fontFamily,
      fontSize: textEditor.fontSize,
      fontWeight: textEditor.fontWeight,
      baseWidth,
      baseHeight,
      growDir: textEditor.growDir,
      padding: 0,
      lineHeight: TEXT_LINE_HEIGHT,
    })
    const posX = textEditor.x + layout.offsetX
    const posY = textEditor.y + layout.offsetY
    const actualLineHeight = textEditor.fontSize * layout.lineHeight
    const heightPadding = Math.min(actualLineHeight * 0.35, 16)
    const paddedHeight = layout.height + heightPadding
    const sharedMeta = {
      author: 'human',
      text: content,
      summary,
      fontFamily: textEditor.fontFamily,
      fontWeight: textEditor.fontWeight,
      fontSize: textEditor.fontSize,
      growDir: textEditor.growDir,
      baseWidth: layout.baseWidth,
      baseHeight: layout.baseHeight,
      configuredWidth: baseWidth,
      configuredHeight: baseHeight,
      lineHeight: layout.lineHeight,
      padding: layout.padding,
      contentWidth: layout.contentWidth,
      contentHeight: layout.contentHeight,
      lineCount: layout.lineCount,
      renderedText: layout.renderedText,
    }
    const draft: ShapeDraft = {
      id: textEditor.id,
      kind: 'text',
      x: posX,
      y: posY,
      w: layout.width,
      h: paddedHeight,
      text: content,
      summary,
      style: { size: 'm', color: textEditor.color, opacity: textEditor.opacity },
      meta: { ...sharedMeta },
    }
    const aiStroke: AIStrokeV11 = {
      id: textEditor.id,
      tool: 'text',
      points: [
        [posX, posY],
        [posX + layout.width, posY + paddedHeight],
      ],
      style: { size: 'm', color: textEditor.color, opacity: textEditor.opacity },
      meta: { ...sharedMeta },
    }
    pushHistory()
    if (textEditor.isEditing) {
      setShapes((prev) => prev.map((s) => (s.id === draft.id ? draft : s)))
      setDrawStack((prev) => {
        let found = false
        const next = prev.map((entry) => {
          if (entry.draft.id !== draft.id) return entry
          found = true
          return { ai: aiStroke, draft }
        })
        return found ? next : prev
      })
    } else {
      setShapes((prev) => [...prev, draft])
      setDrawStack((prev) => [...prev, { ai: aiStroke, draft }])
    }
    setTextEditor(null)
    clearCompletionPreview(textEditor.id)
    updateTextSettings({
      fontFamily: textEditor.fontFamily,
      fontSize: textEditor.fontSize,
      fontWeight: textEditor.fontWeight,
      growDir: textEditor.growDir,
    })
    noteUserAction({ forceStart: true })
  }, [textEditor, pushHistory, setShapes, setDrawStack, updateTextSettings, noteUserAction, computeTextBoxLayout, setSelectedShapeId, clearCompletionPreview])
const openTextEditor = useCallback((params: {
    id: string
    x: number
    y: number
    w: number
    h: number
    color: ColorName
    opacity?: number
    text?: string
    summary?: string
    fontFamily?: string
    fontSize?: number
    fontWeight?: string
    growDir?: TextGrowDir
    editing?: boolean
  }) => {
    const fontSize = params.fontSize ?? textSettings.fontSize
    const fallbackWidth = Math.max(240, fontSize * 10)
    const fallbackHeight = Math.max(160, fontSize * 4)
    const baseWidth = params.w > 0 ? Math.max(params.w, 80) : fallbackWidth
    const baseHeight = params.h > 0 ? Math.max(params.h, Math.round(fontSize * 1.6)) : fallbackHeight
    setTextEditor({
      id: params.id,
      x: params.x,
      y: params.y,
      w: baseWidth,
      h: baseHeight,
      text: params.text ?? '',
      summary: params.summary ?? '',
      fontFamily: params.fontFamily ?? textSettings.fontFamily,
      fontSize,
      fontWeight: params.fontWeight ?? textSettings.fontWeight,
      growDir: params.growDir ?? textSettings.growDir,
      color: params.color,
      opacity: params.opacity ?? 1,
      isEditing: !!params.editing,
      originalShapeId: params.editing ? params.id : undefined,
      pendingCompletion: completionPreviews[params.id] ?? null,
      completing: false,
    })
  }, [textSettings, completionPreviews])
  const openEditorForShape = useCallback((shape: ShapeDraft) => {
    if (shape.kind !== 'text') return
    const meta = shape.meta ?? {}
    openTextEditor({
      id: shape.id,
      x: shape.x,
      y: shape.y,
      w: shape.w ?? 240,
      h: shape.h ?? 160,
      color: (shape.style?.color ?? 'black') as ColorName,
      opacity: shape.style?.opacity ?? 1,
      text: shape.text ?? (meta.text as string) ?? '',
      summary: shape.summary ?? (meta.summary as string) ?? '',
      fontFamily: (meta.fontFamily as string) ?? textSettings.fontFamily,
      fontSize: Number(meta.fontSize ?? textSettings.fontSize) || textSettings.fontSize,
      fontWeight: (meta.fontWeight as string) ?? textSettings.fontWeight,
      growDir: (meta.growDir as TextGrowDir) ?? textSettings.growDir,
      editing: true,
    })
  }, [openTextEditor, textSettings])
  // ===== Ask AI: call backend and populate previews =====
  const askAI = useCallback(async () => {
    try {
      // 1) Ensure session exists
      let curSid = sid
      if (!curSid) {
        const r0 = await apiFetch('/session/init', {
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
      // 2) Build delta strokes (new segments only) with a light simplification pass
      const from = lastSentIndexRef.current
      const deltaStrokes = drawStack.slice(from).map(e => {
        const s = e.ai
        const xy = (s.points || []).map(p => [p[0], p[1]] as [number, number])
        const slim = mergeCollinear(xy, 0.01)
        return { ...s, points: slim.map(([x, y]) => [x, y] as [number, number]) }
      })
      lastSentIndexRef.current = drawStack.length
      // Vision mode: capture canvas snapshot if needed
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
      // 3) Build request (include viewport to help backend validation)
      const snapshot = packAllStrokes()
      const baseReq = {
        sid: curSid!,
        canvas: { viewport: [0, 0, size.width, size.height] as [number, number, number, number] },
        delta: { strokes: deltaStrokes },
        context: { version: 1, intent: 'complete', strokes: snapshot },
        hint,
        gen_scale: aiScale,
        mode, // key: one of the prompt modes
        vision_version: visionVersion,
        ...(mode === "vision" ? {
          image_data,
          image_mime,
          snapshot_size,
        } : {})
      }
      const doPost = async (payload: any) =>
        apiFetch('/suggest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      // === Vision 2.0 two-phase flow ===
      if (mode === "vision" && visionVersion >= 2) {
        // Step 1: image analysis without stroke context
        const req1 = { ...baseReq, seq: 1, context: { version: 1, intent: 'hint', strokes: [] } }
        let res1 = await doPost(req1)
        if (!res1.ok) {
          const t = await res1.text().catch(()=> '')
          throw new Error(`Vision 2.0 step1 failed: ${res1.status} ${res1.statusText}\n${t}`)
        }
        const data1 = await res1.json()
        const v2 = data1?.vision2 || {}
        const instruction: string = (v2.instruction || '').toString()
        // Fall back to hint when server returns no instruction
        const inst = instruction || hint || 'Make the single best next stroke.'
        // Step 2: feed instruction back into the full completion flow
        const req2 = {
          sid: curSid!,
          // Only include required fields to avoid sending image data again
          canvas: { viewport: [0, 0, size.width, size.height] as [number, number, number, number] },
          delta: { strokes: deltaStrokes },
          // Preserve full stroke context
          context: { version: 1, intent: 'complete', strokes: snapshot },
          // Pass Step-1 analysis plus final instruction downstream
          instruction_text: JSON.stringify({
            analysis: (v2.analysis || '').toString(),
            instruction: inst
          }),
          // Keep mode=vision so backend takes the Step-2 branch without another image upload
          mode: "vision",
          vision_version: visionVersion,
          seq: 2,
          // Reuse existing parameters such as hint/gen_scale
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
      // === Legacy flow (Vision 1.0 / full / light) ===
      let res = await doPost({ ...baseReq, sid: curSid! })
      if (!res.ok) {
        // If session expired, re-initialize once then retry
        const txt = await res.text().catch(()=>'')
        if (res.status === 404 && /session not found/i.test(txt)) {
          const r1 = await apiFetch('/session/init', {
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
        // Prefer JSON detail (422/400); otherwise fall back to plain text
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
      // 4) Parse response payload and stage previews
      const data = await res.json()
      if (data?.usage?.new_sid) setSid(String(data.usage.new_sid))
      const payload = data?.payload as AIStrokePayload | undefined
      if (!payload) throw new Error('No payload in response')
      // Log feed entries and raw payload text for debugging
      localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload))
      const items = (payload.strokes || []).map(s => ({ id: s.id, desc: (s.meta as any)?.desc }))
      setAiFeed(prev => ([{ payloadId: 'srv_'+Date.now().toString(36), time: Date.now(), items }, ...prev].slice(0, 50)))
      if (data?.usage?.raw_text) localStorage.setItem('ai_last_raw', String(data.usage.raw_text))
      // Validate, normalize, plan drafts, then store them
      const v = validateAIStrokePayload(payload)
      if (!v.ok || !v.payload) throw new Error('Invalid AI payload: ' + v.errors.join('; '))
      const norm = normalizeAIStrokePayload(v.payload)
      const drafts = planDrafts(norm)   // Already supports poly/line/pen
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
  React.useEffect(() => {
    if (mode !== 'full' && autoMaintain && !autoMaintainPending) {
      void updateAutoMaintain(false, { silent: true })
    }
  }, [mode, autoMaintain, autoMaintainPending, updateAutoMaintain])
  React.useEffect(() => {
    if (!autoMaintain || !sid) {
        setPromoteGroupPending(null)
      if (graphPollRef.current) {
        window.clearInterval(graphPollRef.current)
        graphPollRef.current = null
      }
      if (!autoMaintain) {
        setGraphSnapshot(null)
        setGraphInspectorVisible(false)
      }
      return
    }
    void fetchGraphSnapshot(sid)
    if (graphPollRef.current) window.clearInterval(graphPollRef.current)
    graphPollRef.current = window.setInterval(() => {
      void fetchGraphSnapshot(sid)
    }, 3500) as unknown as number
    return () => {
      if (graphPollRef.current) {
        window.clearInterval(graphPollRef.current)
        graphPollRef.current = null
      }
    }
  }, [autoMaintain, sid, fetchGraphSnapshot])
  // Infinite grid that follows the camera viewport
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
    const lines: React.ReactElement[] = []
    for (let x = xStart; x <= xEnd; x += STEP) {
      lines.push(<KLine key={'gx'+x} points={[x, yStart, x, yEnd]} stroke="#eee" strokeWidth={1 / view.scale} listening={false} />)
    }
    for (let y = yStart; y <= yEnd; y += STEP) {
      lines.push(<KLine key={'gy'+y} points={[xStart, y, xEnd, y]} stroke="#eee" strokeWidth={1 / view.scale} listening={false} />)
    }
    return <Group listening={false}>{lines}</Group>
  }
  // Draft -> Konva node renderer
  const DraftNode: React.FC<{ d: ShapeDraft; preview?: boolean; selected?: boolean; editHighlight?: boolean; completionText?: string | null }> = ({ d, preview, selected, editHighlight, completionText }) => {
    const stroke = colorToStroke(d.style?.color ?? 'black')
    const strokeWidth = SIZE_TO_WIDTH[(d.style?.size ?? 'm')]
    const opacity = preview ? Math.min(0.35, (d.style?.opacity ?? 1)) : (d.style?.opacity ?? 1)
    switch (d.kind) {
      case 'pen':
      case 'polyline': {
        const pts = (d.points ?? []).flatMap(p => [d.x + p.x, d.y + p.y])
        // Smooth rendering when meta.curve is true
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
        if (rx === 0 && ry === 0) return null  // Skip rendering when ellipse is too small
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
        // Keep polygon vertices intact; render as a closed shape
        const pts = (d.points ?? []).flatMap(p => [d.x + p.x, d.y + p.y])
        return <KLine points={pts} closed stroke={stroke} strokeWidth={strokeWidth} opacity={opacity} lineJoin="round" />
      }
      case 'text': {
        const fillColor = colorToStroke(d.style?.color ?? 'black')
        const fontSize = (d.meta?.fontSize ?? d.meta?.fontsize ?? d.meta?.font_size ?? 16) as number
        const fontFamily = (d.meta?.fontFamily ?? 'sans-serif') as string
        const fontStyle = (d.meta?.fontWeight === 'bold' || d.meta?.fontWeight === '700')
          ? 'bold'
          : 'normal'
        const boxW = d.w ?? 160
        const boxH = d.h ?? 80
        const lineHeight = typeof d.meta?.lineHeight === 'number' && Number.isFinite(d.meta.lineHeight)
          ? (d.meta.lineHeight as number)
          : TEXT_LINE_HEIGHT
        const hasCompletion = !!completionText
        const isHighlighted = !!editHighlight
        let borderColor = preview ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.4)'
        let borderDash: number[] | undefined = preview ? [4,4] : undefined
        let fillColorOverlay: string | undefined
        if (isHighlighted) {
          borderColor = '#ffb74d'
          borderDash = [2, 2]
          fillColorOverlay = 'rgba(255,183,77,0.12)'
        } else if (hasCompletion) {
          borderColor = '#2563eb'
          borderDash = [6, 4]
          fillColorOverlay = 'rgba(37,99,235,0.08)'
        } else if (selected) {
          borderColor = '#4aa3ff'
          borderDash = [8,4]
        }
        return (
          <Group listening={false}>
            <KRect
              x={d.x}
              y={d.y}
              width={boxW}
              height={boxH}
              stroke={borderColor}
              strokeWidth={1 / view.scale}
              dash={borderDash}
              fill={fillColorOverlay}
              opacity={0.6}
            />
            <KText
              x={d.x}
              y={d.y}
              width={boxW}
              height={boxH}
              text={(d.meta?.renderedText as string) ?? d.text ?? ''}
              fontFamily={fontFamily}
              fontSize={fontSize}
              fontStyle={fontStyle}
              fill={fillColor}
              opacity={preview ? Math.min(0.35, d.style?.opacity ?? 1) : (d.style?.opacity ?? 1)}
              align="left"
              verticalAlign="top"
              listening={false}
              wrap="char"
              lineHeight={lineHeight}
            />
            {completionText && (
              <KText
                x={d.x}
                y={d.y + boxH + 6}
                width={boxW}
                text={completionText}
                fontFamily={fontFamily}
                fontSize={Math.max(fontSize * 0.9, 12)}
                fontStyle="italic"
                fill="#2563eb"
                align="left"
                wrap="word"
                listening={false}
              />
            )}
          </Group>
        )
      }
      case 'edit':
        return null
      case 'erase':
        return <KText x={d.x} y={d.y} text="[erase]" fill="#999" opacity={opacity} />
      default:
        return null
    }
  }
  // Grid snapping (display-only helper)
  const GRID_STEP = 32
  const snapPoint = useCallback((x: number, y: number) => {
    // Display-only snapping; do not apply when persisting/uploads
    return snap
      ? [Math.round(x / GRID_STEP) * GRID_STEP, Math.round(y / GRID_STEP) * GRID_STEP]
      : [x, y]
  }, [snap])
  // Approximate closure if endpoints are within tolerance
  const isClosedPath = useCallback((pts: Array<[number,number]>)=>{
    if (pts.length < 3) return false
    const [x0,y0] = pts[0]
    const [xn,yn] = pts[pts.length-1]
    const tol = snap ? GRID_STEP * 0.5 : 3 // Snap enabled uses a wider tolerance
    return Math.hypot(xn - x0, yn - y0) <= tol
  }, [snap])
  // Optionally snap preview stroke arrays [x0,y0,x1,y1,...]
  const snapPointsIfNeeded = useCallback((pts: number[]) => {
    if (!snap) return pts
    const out: number[] = []
    for (let i = 0; i < pts.length; i += 2) {
      const [sx, sy] = snapPoint(pts[i], pts[i+1])
      out.push(sx, sy)
    }
    return out
  }, [snap, snapPoint])
  // ----- Whole-stroke erasing helpers -----
  // Shortest distance from a point to a segment
  const distPointToSegment = (px:number, py:number, x1:number, y1:number, x2:number, y2:number) => {
    const A = px - x1, B = py - y1, C = x2 - x1, D = y2 - y1
    const dot = A*C + B*D
    const len = C*C + D*D
    const t = len ? Math.max(0, Math.min(1, dot / len)) : 0
    const qx = x1 + t*C, qy = y1 + t*D
    return Math.hypot(px - qx, py - qy)
  }
  // Minimum distance from a polyline to a point (absolute coords)
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
  // Hit test succeeds when any segment is within the erase radius
  const hitStrokeByCircle = (d: ShapeDraft, cx:number, cy:number, r:number) => {
    // Works for pen / line / polyline / poly drafts
    if (!d.points || d.points.length < 2) return false
    const absPts: Array<[number,number]> = d.points.map(p => [d.x + p.x, d.y + p.y])
    return polylineMinDistToPoint(absPts, cx, cy) <= r
  }
  const hitTextBoxByCircle = (d: ShapeDraft, cx:number, cy:number, r:number) => {
    const w = d.w ?? 0
    const h = d.h ?? 0
    const x0 = Math.min(d.x, d.x + w)
    const x1 = Math.max(d.x, d.x + w)
    const y0 = Math.min(d.y, d.y + h)
    const y1 = Math.max(d.y, d.y + h)
    return cx >= x0 - r && cx <= x1 + r && cy >= y0 - r && cy <= y1 + r
  }
  const findTextShapeAtPoint = useCallback((px:number, py:number) => {
    for (let i = shapes.length - 1; i >= 0; i--) {
      const d = shapes[i]
      if (d.kind !== 'text') continue
      const w = d.w ?? 0
      const h = d.h ?? 0
      const x0 = Math.min(d.x, d.x + w)
      const x1 = Math.max(d.x, d.x + w)
      const y0 = Math.min(d.y, d.y + h)
      const y1 = Math.max(d.y, d.y + h)
      if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return d
    }
    return null
  }, [shapes])
const moveTextShape = useCallback((id: string, nextX: number, nextY: number) => {
    const target = shapes.find((s) => s.id === id && s.kind === 'text')
    if (!target) return
    const updated: ShapeDraft = { ...target, x: nextX, y: nextY }
    setShapes(prev => prev.map(s => (s.id === id ? updated : s)))
    const width = updated.w ?? 0
    const height = updated.h ?? 0
    setDrawStack(prev => prev.map(entry => {
      if (entry.draft.id !== id) return entry
      const points: [number, number, number?, number?][] = [
        [nextX, nextY],
        [nextX + width, nextY + height],
      ]
      const ai: AIStrokeV11 = {
        ...entry.ai,
        points,
      }
      return { ai, draft: updated }
    }))
  }, [shapes, setShapes, setDrawStack])
  const applyEditDraftToState = useCallback((draft: ShapeDraft, baseShapes: ShapeDraft[], baseStack: DrawStackEntry[]) => {
    if (draft.kind !== 'edit') return { shapes: baseShapes, stack: baseStack }
    const meta = draft.meta ?? {}
    const targetIdRaw = meta.targetId ?? draft.targetId ?? meta.target ?? meta.id
    if (!targetIdRaw) return { shapes: baseShapes, stack: baseStack }
    const targetId = String(targetIdRaw)
    const index = baseShapes.findIndex(s => s.id === targetId && s.kind === 'text')
    if (index === -1) return { shapes: baseShapes, stack: baseStack }
    const target = baseShapes[index]
    const currentMeta = target.meta ?? {}
    const content = String(meta.content ?? draft.text ?? target.text ?? '')
    const summary = String(meta.summary ?? target.summary ?? '')
    const fontFamily = String(meta.fontFamily ?? currentMeta.fontFamily ?? 'sans-serif')
    const fontWeight = String(meta.fontWeight ?? currentMeta.fontWeight ?? '400')
    const fontSize = Number(meta.fontSize ?? currentMeta.fontSize ?? 16) || 16
    const growDir = (meta.growDir as TextGrowDir) ?? (currentMeta.growDir as TextGrowDir) ?? 'down'
    const padding = Number(meta.padding ?? currentMeta.padding ?? 0)
    const baseWidth = Number(meta.configuredWidth ?? currentMeta.configuredWidth ?? target.w ?? 240)
    const baseHeight = Number(meta.configuredHeight ?? currentMeta.configuredHeight ?? target.h ?? 160)
    const rawLineHeight = typeof meta.lineHeight === 'number'
      ? Number(meta.lineHeight)
      : typeof currentMeta.lineHeight === 'number'
        ? Number(currentMeta.lineHeight)
        : TEXT_LINE_HEIGHT
    const layout = computeTextBoxLayout({
      text: content,
      fontFamily,
      fontSize,
      fontWeight,
      baseWidth,
      baseHeight,
      growDir,
      padding,
      lineHeight: rawLineHeight,
    })
    const posX = target.x + layout.offsetX
    const posY = target.y + layout.offsetY
    const actualLineHeight = fontSize * layout.lineHeight
    const heightPadding = Math.min(actualLineHeight * 0.35, 16)
    const paddedHeight = layout.height + heightPadding
    const updatedMeta = {
      ...currentMeta,
      ...meta,
      text: content,
      summary,
      fontFamily,
      fontWeight,
      fontSize,
      growDir,
      baseWidth: layout.baseWidth,
      baseHeight: layout.baseHeight,
      configuredWidth: baseWidth,
      configuredHeight: baseHeight,
      lineHeight: layout.lineHeight,
      padding: layout.padding,
      contentWidth: layout.contentWidth,
      contentHeight: layout.contentHeight,
      lineCount: layout.lineCount,
      lastOperation: meta.operation ?? draft.operation,
      renderedText: layout.renderedText,
    }
    const updatedShape: ShapeDraft = {
      ...target,
      x: posX,
      y: posY,
      w: layout.width,
      h: paddedHeight,
      text: content,
      summary,
      meta: updatedMeta,
    }
    const nextShapes = [...baseShapes]
    nextShapes[index] = updatedShape
    const width = layout.width ?? 0
    const nextStack = baseStack.map(entry => {
      if (entry.draft.id !== targetId) return entry
      const ai: AIStrokeV11 = {
        ...entry.ai,
        points: [
          [posX, posY],
          [posX + width, posY + paddedHeight],
        ],
        meta: { ...(entry.ai.meta ?? {}), ...updatedMeta },
      }
      return { ai, draft: updatedShape }
    })
    return { shapes: nextShapes, stack: nextStack }
  }, [computeTextBoxLayout])
  const acceptAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    const entry = previews[currentPayloadId]
    if (!entry) { alert('Preview not found'); return }
    pushHistory()
    let nextShapes = [...shapes]
    let nextDrawStack = [...drawStack]
    for (const draft of entry.drafts) {
      if (draft.kind === 'edit') {
        const result = applyEditDraftToState(draft, nextShapes, nextDrawStack)
        nextShapes = result.shapes
        nextDrawStack = result.stack
      } else {
        nextShapes = [...nextShapes, draft]
        const ai = draftToAIStroke(draft)
        if (ai) nextDrawStack = [...nextDrawStack, { ai, draft }]
      }
    }
    setShapes(nextShapes)
    setDrawStack(nextDrawStack)
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    setCompletionPreviews(prev => {
      const next = { ...prev }
      for (const draft of entry.drafts) {
        if (draft.kind !== 'edit') continue
        const meta = draft.meta ?? {}
        const targetId = draft.targetId ?? meta.targetId ?? meta.target ?? meta.id
        if (targetId) delete next[String(targetId)]
      }
      return next
    })
    noteUserAction({ forceStart: true })
  }, [currentPayloadId, previews, pushHistory, noteUserAction, shapes, drawStack, applyEditDraftToState])
  const dismissAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    clearAutoTimer()
  }, [currentPayloadId, clearAutoTimer])
  const buildEditPreviewNode = useCallback((draft: ShapeDraft, key: string) => {
    const meta = draft.meta ?? {}
    const targetId = (meta.targetId ?? draft.targetId) as string | undefined
    const target = targetId ? shapesById[targetId] : undefined
    const baseX = target?.x ?? draft.x ?? 0
    const baseY = target?.y ?? draft.y ?? 0
    const baseWidth = target?.w ?? draft.w ?? 220
    const operator = String(meta.operation ?? draft.operation ?? '编辑')
    const content = String(meta.content ?? draft.text ?? '')
    const message = `Agent Suggest: (${operator}) : ${content}`
    const overlayWidth = Math.max(baseWidth ?? 220, 220)
    const lineCount = Math.max(message.split(/\r?\n/).length, 1)
    const overlayHeight = Math.min(lineCount * 18 + 20, 260)
    const overlayY = Math.max(baseY - overlayHeight - 8, baseY - overlayHeight - 8)
    return (
      <Group key={key} listening={false}>
        <KRect
          x={baseX}
          y={overlayY}
          width={overlayWidth}
          height={overlayHeight}
          cornerRadius={8}
          fill="rgba(74,163,255,0.12)"
          stroke="rgba(74,163,255,0.6)"
          strokeWidth={1 / view.scale}
          dash={[6, 4]}
        />
        <KText
          x={baseX + 8}
          y={overlayY + 6}
          width={overlayWidth - 16}
          text={message}
          fontFamily="sans-serif"
          fontSize={14}
          fill="#1d4ed8"
          wrap="word"
          listening={false}
        />
      </Group>
    )
  }, [shapesById, view.scale])
  // Apply whole-stroke erasure (one history push per gesture)
  const eraseWholeStrokesAt = (cx:number, cy:number, radius:number) => {
    // --- Local helpers dedicated to ellipse hit detection ---
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
      pts.push(pts[0]) // Close loop
      return pts
    }
    const hitEllipseByCircle = (d: ShapeDraft, px: number, py: number, r: number) => {
      // Normalize bounding box ordering
      const w = d.w ?? 0, h = d.h ?? 0
      const x0 = Math.min(d.x, d.x + w), x1 = Math.max(d.x, d.x + w)
      const y0 = Math.min(d.y, d.y + h), y1 = Math.max(d.y, d.y + h)
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
      const rx = Math.abs(x1 - x0) / 2, ry = Math.abs(y1 - y0) / 2
      if (rx < 0.5 && ry < 0.5) return false
      // Quick reject: outside the bounding box expanded by radius
      if (px < x0 - r || px > x1 + r || py < y0 - r || py > y1 + r) return false
      // Precise pass: discretize the ellipse boundary and measure distances
      const pts = ellipseToPolyline(cx, cy, rx, ry, 48)
      for (let i = 1; i < pts.length; i++) {
        const [ax, ay] = pts[i - 1]
        const [bx, by] = pts[i]
        if (distancePointToSegment(px, py, ax, ay, bx, by) <= r) return true
      }
      return false
    }
    // --- End local helpers ---
    if (!eraseGestureStarted.current) {
      pushHistory()
      eraseGestureStarted.current = true
    }
    const removed = new Set<string>()
    const kept: ShapeDraft[] = []
    for (const d of shapes) {
      let hit = false
      if (d.kind === 'ellipse') {
        // Ellipses use custom hit testing; others reuse the generic path
        hit = hitEllipseByCircle(d, cx, cy, radius)
      } else if (d.kind === 'text') {
        hit = hitTextBoxByCircle(d, cx, cy, radius)
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
  // Pointer handlers branch by toolMode
  const onMouseDown = useCallback((e: any) => {
    if (textEditor) return
    if (toolMode === 'hand') return
    const pos = e.target.getStage()?.getPointerPosition()
    if (!pos) return
    if (toolMode === 'select') {
      const wpt = screenToWorld(pos.x, pos.y)
      const target = findTextShapeAtPoint(wpt.x, wpt.y)
      if (!target) {
        setSelectedShapeId(null)
        selectDragRef.current = null
        return
      }
      noteUserAction()
      setSelectedShapeId(target.id)
      selectDragRef.current = {
        id: target.id,
        offsetX: wpt.x - target.x,
        offsetY: wpt.y - target.y,
        startX: target.x,
        startY: target.y,
        moved: false,
      }
      return
    }
    if (toolMode === 'pen') {
      setIsDrawing(true)
      // Snap on stores grid-aligned integers; off stores raw floats
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
    } else if (toolMode === 'text') {
      setIsDrawing(true)
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snap ? snapPoint(wpt.x, wpt.y) : [wpt.x, wpt.y]
      const id = `text_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
      setBoxDraft({
        id,
        kind: 'text',
        x: sx,
        y: sy,
        w: 0,
        h: 0,
        text: '',
        summary: '',
        style: { size: 'm', color: brushColor, opacity: 1 },
        meta: {
          author: 'human',
          fontFamily: textSettings.fontFamily,
          fontWeight: textSettings.fontWeight,
          fontSize: textSettings.fontSize,
          growDir: textSettings.growDir,
        },
      })
    } else { // eraser
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })
      eraseGestureStarted.current = false // Start a new eraser gesture
      eraseWholeStrokesAt(sx, sy, eraserRadius) // Attempt initial erase immediately
    }
  }, [snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld, brushSize, brushColor, textSettings, textEditor, findTextShapeAtPoint, noteUserAction])
  const onMouseMove = useCallback((e: any) => {
    if (textEditor) return
    if (toolMode === 'hand') return
    const pos = e.target.getStage()?.getPointerPosition()
    if (!pos) return
    if (toolMode === 'select') {
      const drag = selectDragRef.current
      if (!drag) return
      const wpt = screenToWorld(pos.x, pos.y)
      const nextX = wpt.x - drag.offsetX
      const nextY = wpt.y - drag.offsetY
      if (!drag.moved) {
        const dist = Math.hypot(nextX - drag.startX, nextY - drag.startY)
        if (dist > 0.5) {
          drag.moved = true
          pushHistory()
        } else {
          return
        }
      }
      moveTextShape(drag.id, nextX, nextY)
      return
    }
    if (toolMode === 'pen') {
      if (!isDrawing) return
      setRawPoints(prev => {
        const n = prev.length
        const wpt = screenToWorld(pos.x, pos.y)
        // Snap on appends snapped points; off appends raw world-space floats
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
    } else if (toolMode === 'text') {
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
      // Eraser keeps snapping for more reliable hits
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })           // Show eraser cursor indicator
      eraseWholeStrokesAt(sx, sy, eraserRadius)    // Continue erasing whole strokes
    }
  }, [isDrawing, boxDraft, snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld, textEditor, moveTextShape, pushHistory])
  const onMouseUp = useCallback(() => {
    if (textEditor) return
    if (toolMode === 'hand') return
    if (toolMode === 'select') {
      const drag = selectDragRef.current
      const targetId = drag?.id ?? selectedShapeId
      selectDragRef.current = null
      if (drag && drag.moved) {
        return
      }
      if (targetId) {
        const shape = shapes.find((s) => s.id === targetId && s.kind === 'text') as ShapeDraft | undefined
        if (shape) openEditorForShape(shape)
      }
      return
    }
    if (toolMode === 'pen') {
      if (!isDrawing) return
      setIsDrawing(false)
      if (rawPoints.length < 4) { setRawPoints([]); return }
      // Convert [x0,y0,x1,y1,...] into [[x,y], ...]
      const absPts: Array<[number, number]> = []
      for (let i = 0; i < rawPoints.length; i += 2) absPts.push([rawPoints[i], rawPoints[i+1]])
      // Remove redundant collinear segments first
      let basePts = mergeCollinear(absPts, 0.01)
      // --- A) Closed shapes: always treat as polygons ---
      if (isClosedPath(basePts)) {
        // Ensure closed paths repeat the first point
        if (basePts.length >= 2) {
          const [x0,y0] = basePts[0]
          const [xn,yn] = basePts[basePts.length-1]
          if (x0 !== xn || y0 !== yn) basePts = [...basePts, [x0,y0]]
        }
        // Further trim collinear redundancy while keeping turns
        const closedPts = mergeCollinear(basePts, 0.0)
        // Record history snapshot
        pushHistory()
        // Build polygon draft and matching AI stroke
        let minX=Infinity, minY=Infinity
        for (const [x,y] of closedPts){ if (x<minX) minX=x; if (y<minY) minY=y }
        const local = closedPts.map(([x,y])=>({x:x-minX,y:y-minY}))
        const id = `poly_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const draft: ShapeDraft = { id, kind:'poly', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta} }
        const aiStroke: AIStrokeV11 = { id, tool:'poly', points: closedPts.map(([x,y])=>[x,y] as [number,number]), style: currentBrush.style, meta:{ author:'human' } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
        return
      }
      // --- B) Open paths: optionally convert to curves ---
      let displayPts: Array<[number,number]>
      if (curveTurns) {
        // Curve: Chaikin smoothing plus even resampling for smooth feel
        displayPts = resampleEvenly(chaikin(basePts, 2), 3)
      } else {
        // Polyline: keep sharp corners (tension=0)
        displayPts = basePts
      }
      // Detect straight lines using max deviation from the chord
      const LINEAR_EPS = 1.2
      const isLine = geomMaxDeviationFromChord(displayPts) <= LINEAR_EPS
      // Record history snapshot
      pushHistory()
      // Build draft and AI stroke
      if (isLine && displayPts.length >= 2) {
        const p0 = displayPts[0], pn = displayPts[displayPts.length-1]
        const id = `line_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const minX = Math.min(p0[0], pn[0]), minY = Math.min(p0[1], pn[1])
        const local = [{x:p0[0]-minX, y:p0[1]-minY}, {x:pn[0]-minX, y:pn[1]-minY}]
        const draft: ShapeDraft = { id, kind:'line', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta} }
        const aiStroke: AIStrokeV11 = { id, tool:'line', points:[p0, pn], style: currentBrush.style, meta:{ author:'human' } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
      } else {
        // Non-lines render as pen strokes; store curve flag in meta for tension
        let minX=Infinity, minY=Infinity
        for (const [x,y] of displayPts){ if (x<minX) minX=x; if (y<minY) minY=y }
        const local = displayPts.map(([x,y])=>({x:x-minX,y:y-minY}))
        const id = `pen_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,6)}`
        const draft: ShapeDraft = { id, kind:'pen', x:minX, y:minY, points:local, style: currentBrush.style, meta:{...currentBrush.meta, curve: curveTurns} }
        const aiStroke: AIStrokeV11 = { id, tool:'pen', points: displayPts.map(([x,y])=>[x,y] as [number,number]), style: currentBrush.style, meta:{ author:'human', curve: curveTurns } }
        setDrawStack(prev => [...prev, { ai: aiStroke, draft }])
        setShapes(prev => [...prev, draft])
        setRawPoints([])
      }
    } else if (toolMode === 'ellipse') {
      if (!isDrawing || !boxDraft) return
      setIsDrawing(false)
      // Normalize diagonal corner coordinates (absolute space)
      const x0 = Math.min(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const y0 = Math.min(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      const x1 = Math.max(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const y1 = Math.max(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      // Guard against degenerate zero-sized shapes
      if (Math.abs(x1 - x0) < 1 && Math.abs(y1 - y0) < 1) { setBoxDraft(null); return }
      // Record history snapshot
      pushHistory()
      // Build draft and AI stroke
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
    } else if (toolMode === 'text') {
      if (!isDrawing || !boxDraft) { setIsDrawing(false); setBoxDraft(null); return }
      setIsDrawing(false)
      const bx0 = Math.min(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const by0 = Math.min(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      const bx1 = Math.max(boxDraft.x, boxDraft.x + (boxDraft.w ?? 0))
      const by1 = Math.max(boxDraft.y, boxDraft.y + (boxDraft.h ?? 0))
      let width = bx1 - bx0
      let height = by1 - by0
      if (width <= 1 && height <= 1) {
        width = 240
        height = 160
      }
      const meta = boxDraft.meta ?? {}
      const styleColor = (boxDraft.style?.color ?? brushColor) as ColorName
      const opacity = boxDraft.style?.opacity ?? 1
      setBoxDraft(null)
      openTextEditor({
        id: boxDraft.id,
        x: bx0,
        y: by0,
        w: Math.max(width, 32),
        h: Math.max(height, 32),
        color: styleColor,
        opacity,
        text: (boxDraft.text ?? '') as string,
        summary: (boxDraft.summary ?? '') as string,
        fontFamily: (meta as any).fontFamily,
        fontSize: (meta as any).fontSize,
        fontWeight: (meta as any).fontWeight,
        growDir: (meta as any).growDir,
      })
    } else {
      // Finish eraser gesture
      setEraserCursor(null)
      eraseGestureStarted.current = false
    }
  }, [isDrawing, rawPoints, boxDraft, snap, snapPoint, toolMode, curveTurns, currentBrush, pushHistory, setShapes, setDrawStack, brushColor, openTextEditor, textEditor, openEditorForShape, shapes, selectedShapeId])
  // --- Enter shortcut: auto-accept preview when focus is outside inputs ---
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
      // Cancel any in-progress drawing preview
      setIsDrawing(false)
      // Clear transient boxDraft/preview state
      setBoxDraft?.(null as any)
    }
  }, [toolMode])
  React.useEffect(() => {
    if (toolMode !== 'select') {
      setSelectedShapeId(null)
      selectDragRef.current = null
    }
  }, [toolMode])
  React.useEffect(() => {
    if (selectedShapeId && !shapes.some(s => s.id === selectedShapeId)) {
      setSelectedShapeId(null)
    }
  }, [selectedShapeId, shapes])
  React.useEffect(() => {
    if (!textEditor) return
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        ev.preventDefault()
        cancelTextEditor()
      } else if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) {
        ev.preventDefault()
        commitTextEditor()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [textEditor, cancelTextEditor, commitTextEditor])
  const textEditorScreen = textEditor ? worldToScreen(textEditor.x, textEditor.y) : null
  const textEditorSize = textEditor ? {
    width: Math.max(textEditor.w * view.scale, 260),
    height: Math.max(textEditor.h * view.scale, 200),
  } : null
// ===== Stage binds camera (x/y/scale); hand mode enables dragging =====
  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow:'hidden' }}>
      <TopToolbar
        showGrid={showGrid}
        snap={snap}
        curveTurns={curveTurns}
        onToggleGrid={toggleGrid}
        onToggleSnap={toggleSnap}
        onToggleCurve={toggleCurveTurns}
        onAskAI={askAI}
        onAcceptAI={acceptAI}
        onDismissAI={dismissAI}
      />
      <SidePanel
        toolMode={toolMode}
        onToolModeChange={setToolMode}
        eraserRadius={eraserRadius}
        onEraserRadiusChange={setEraserRadius}
        brushSize={brushSize}
        onBrushSizeChange={setBrushSize}
        brushColor={brushColor}
        onBrushColorChange={setBrushColor}
        aiScale={aiScale}
        onAiScaleChange={setAiScale}
        autoComplete={autoComplete}
        onAutoCompleteToggle={handleAutoCompleteToggle}
        autoCountdown={autoCountdown}
        hasActivePreview={hasActivePreview}
        canUndo={canUndo}
        canRedo={canRedo}
        onUndo={undo}
        onRedo={redo}
        onExportJSON={exportJSON}
        onImportJSON={importJSON}
        fileInputRef={fileRef}
        onExportAI={exportHumanStrokesAI}
        onApplyAIStub={applyAIStub}
        onPreviewAI={previewAI}
        promptMode={mode}
        visionVersion={visionVersion}
        onVisionVersionChange={setVisionVersion}
        textSettings={textSettings}
        onTextSettingsChange={updateTextSettings}
        onToggleGraphInspector={toggleGraphInspector}
        graphInspectorActive={graphInspectorVisible}
      />
      {textEditor && textEditorScreen && textEditorSize && (
        <div
          style={{
            position: 'absolute',
            left: textEditorScreen.x,
            top: textEditorScreen.y,
            width: textEditorSize.width,
            minWidth: 260,
            maxWidth: 420,
            background: 'rgba(255,255,255,0.95)',
            border: '1px solid #d1d5db',
            borderRadius: 14,
            padding: 14,
            zIndex: 2000,
            boxShadow: '0 14px 36px rgba(15,23,42,0.18)',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            backdropFilter: 'blur(6px)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1f2937' }}>
              {textEditor.isEditing ? 'Edit text box' : 'Create text box'}
            </div>
            <button
              onClick={cancelTextEditor}
              style={{ border: 'none', background: 'transparent', fontSize: 16, cursor: 'pointer', color: '#6b7280' }}
              title="Cancel"
            >
              ×
            </button>
          </div>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 12, color: '#4b5563' }}>Summary</span>
            <input
              type="text"
              value={textEditor.summary}
              onChange={(e) => updateTextEditorState({ summary: e.target.value.slice(0, 30) })}
              style={{ ...INPUT_BASE, width: '100%' }}
            />
            <span style={{ alignSelf: 'flex-end', fontSize: 10, color: '#9ca3af' }}>
              {textEditor.summary.length}/30
            </span>
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            <label style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Font family</span>
              <select
                value={textEditor.fontFamily}
                onChange={(e) => updateTextEditorState({ fontFamily: e.target.value })}
                style={{ ...INPUT_BASE, width: '100%' }}
              >
                {['sans-serif', 'serif', 'monospace', 'cursive'].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </label>
            <label style={{ width: 90, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Font size</span>
              <input
                type="number"
                min={8}
                max={96}
                value={Math.round(textEditor.fontSize)}
                onChange={(e) => {
                  const next = Math.max(8, Math.min(96, Number(e.target.value) || textEditor.fontSize))
                  updateTextEditorState({ fontSize: next })
                }}
                style={{ ...INPUT_BASE, width: '100%' }}
              />
            </label>
            <label style={{ width: 90, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Weight</span>
              <select
                value={textEditor.fontWeight}
                onChange={(e) => updateTextEditorState({ fontWeight: e.target.value })}
                style={{ ...INPUT_BASE, width: '100%' }}
              >
                {['300', '400', '500', '600', '700'].map(w => (
                  <option key={w} value={w}>{w}</option>
                ))}
              </select>
            </label>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <label style={{ width: 90, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Width</span>
              <input
                type="number"
                min={32}
                max={1600}
                value={Math.round(textEditor.w)}
                onChange={(e) => {
                  const next = Math.max(32, Math.min(1600, Number(e.target.value) || textEditor.w))
                  updateTextEditorState({ w: next })
                }}
                style={{ ...INPUT_BASE, width: '100%' }}
              />
            </label>
            <label style={{ width: 90, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Height</span>
              <input
                type="number"
                min={32}
                max={1600}
                value={Math.round(textEditor.h)}
                onChange={(e) => {
                  const next = Math.max(32, Math.min(1600, Number(e.target.value) || textEditor.h))
                  updateTextEditorState({ h: next })
                }}
                style={{ ...INPUT_BASE, width: '100%' }}
              />
            </label>
            <label style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#4b5563' }}>Grow</span>
              <select
                value={textEditor.growDir}
                onChange={(e) => updateTextEditorState({ growDir: e.target.value as TextGrowDir })}
                style={{ ...INPUT_BASE, width: '100%' }}
              >
                {(['down', 'right', 'up', 'left'] as const).map(dir => (
                  <option key={dir} value={dir}>{dir}</option>
                ))}
              </select>
            </label>
          </div>
          <div>
            <span style={{ fontSize: 12, color: '#4b5563', display: 'block', marginBottom: 4 }}>Color</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 6 }}>
              {COLORS.map(c => (
                <button
                  key={c}
                  title={c}
                  onClick={() => updateTextEditorState({ color: c as ColorName })}
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: 6,
                    border: `2px solid ${textEditor.color === c ? '#4aa3ff' : '#e5e7eb'}`,
                    background: c === 'white' ? '#fff' : c.replace('light-', 'light'),
                    cursor: 'pointer',
                  }}
                />
              ))}
            </div>
          </div>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 12, color: '#4b5563' }}>Content (输入 ::: 触发补全，Ctrl/Cmd + Enter 保存)</span>
            <textarea
              value={textEditor.text}
              onChange={(e) => handleEditorTextChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault()
                  commitTextEditor()
                }
              }}
              rows={Math.max(6, Math.round(textEditorSize.height / 40))}
              style={{
                width: '100%',
                minHeight: 160,
                resize: 'vertical',
                padding: '8px 10px',
                borderRadius: 10,
                border: '1px solid #d1d5db',
                fontFamily: textEditor.fontFamily,
                fontSize: textEditor.fontSize,
                lineHeight: TEXT_LINE_HEIGHT,
              }}
            />
          </label>
          {textEditor.completing && (
            <div style={{ fontSize: 12, color: '#2563eb' }}>补全中…</div>
          )}
          {textEditor.pendingCompletion && !textEditor.completing && (
            <div
              style={{
                fontSize: 12,
                fontStyle: 'italic',
                color: '#2563eb',
                background: 'rgba(37,99,235,0.08)',
                padding: '6px 8px',
                borderRadius: 8,
              }}
            >
              {textEditor.pendingCompletion}
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button
              onClick={cancelTextEditor}
              style={{ ...BUTTON_BASE, borderColor: '#d1d5db', background: '#f9fafb' }}
            >
              Cancel
            </button>
            <button
              onClick={commitTextEditor}
              style={{ ...BUTTON_BASE, borderColor: '#4aa3ff', background: 'rgba(74,163,255,0.15)', color: '#1d4ed8' }}
            >
              Save
            </button>
          </div>
        </div>
      )}
      {/* Konva stage spans the viewport; side panel floats above */}
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
        onMouseDown={(e)=>{ /* User begins an action that may modify content */
          noteUserAction()
          onMouseDown(e)
        }}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{ cursor: stageCursor }}
      >
        {/* Optional grid */}
        <Layer listening={false}>{showGrid && <Grid />}</Layer>
        {/* Committed shapes */}
        <Layer>
          {shapes.map(d => {
            const completionText = completionPreviews[d.id]
            return (
              <DraftNode
                key={'s:'+d.id}
                d={d}
                selected={selectedShapeId === d.id}
                editHighlight={activeEditTargets.has(d.id)}
                completionText={completionText}
              />
            )
          })}
          {/* Live drawing preview (not yet in shapes) */}
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
          {isDrawing && boxDraft && (toolMode==='ellipse' || toolMode==='text') && (
            <DraftNode
              d={boxDraft}
              completionText={
                textEditor && textEditor.id === boxDraft.id
                  ? textEditor.pendingCompletion ?? completionPreviews[textEditor.id] ?? null
                  : null
              }
            />
          )}
          {/* Eraser radius indicator (rounded rect avoids extra import) */}
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
        {showGraphHighlights && (
          <Layer listening={false}>
            {graphBlockCards.map((block) => {
              const fragments = block.fragments?.filter((frag) => frag.type === 'text' && frag.bbox) ?? []
              const hasBlockBox = !!block.bbox
              if (!fragments.length && !hasBlockBox) return null
              const overlays = fragments.map((frag) => {
                if (!frag.bbox) return null
                const [fx0, fy0, fx1, fy1] = frag.bbox
                const width = Math.max(4, fx1 - fx0)
                const height = Math.max(4, fy1 - fy0)
                return (
                  <KRect
                    key={frag.id}
                    x={fx0}
                    y={fy0}
                    width={width}
                    height={height}
                    fill={hexToRgba(block.color, 0.2)}
                    listening={false}
                    cornerRadius={8}
                  />
                )
              })
              let bboxNode: React.ReactNode = null
              if (block.bbox) {
                const [bx0, by0, bx1, by1] = block.bbox
                const bw = Math.max(16, bx1 - bx0)
                const bh = Math.max(16, by1 - by0)
                const tentativeLabelY = by0 - 28
                const labelY = tentativeLabelY >= 12 ? tentativeLabelY : by0 + 8
                bboxNode = (
                  <>
                    <KRect
                      x={bx0}
                      y={by0}
                      width={bw}
                      height={bh}
                      stroke={block.color}
                      strokeWidth={1.6}
                      dash={[10, 6]}
                      listening={false}
                      cornerRadius={14}
                      opacity={0.9}
                    />
                    <KLabel x={bx0} y={labelY} listening={false}>
                      <KTag
                        fill={hexToRgba(block.color, 0.55)}
                        stroke={block.color}
                        lineJoin="round"
                        cornerRadius={6}
                        shadowColor={block.color}
                        shadowBlur={6}
                        shadowOpacity={0.18}
                      />
                      <KText
                        text={block.label || block.blockId}
                        fontSize={12}
                        fill="#0f172a"
                        padding={6}
                      />
                    </KLabel>
                  </>
                )
              }
              return (
                <Group key={`block-highlight-${block.blockId}`} listening={false}>
                  {overlays}
                  {bboxNode}
                </Group>
              )
            })}
          </Layer>
        )}
        {/* AI previews */}
        <Layer>
          {previewEntries.map(entry => (
            <Group key={'p:'+entry.payloadId} listening={false} name="ai-candidate" id={entry.payloadId}>
              {entry.drafts.map(d => {
                if (d.kind === 'edit') {
                  return buildEditPreviewNode(d, `edit:${entry.payloadId}:${d.id}`)
                }
                return <DraftNode key={'pd:'+entry.payloadId+':'+d.id} d={d} preview />
              })}
            </Group>
          ))}
        </Layer>
      </Stage>
      <BottomPanel
        hint={hint}
        onHintChange={setHint}
        onSubmit={askAI}
        mode={mode}
        onModeCycle={cycleMode}
        aiFeed={aiFeed}
        showAutoMaintain={mode === 'full'}
        autoMaintainEnabled={autoMaintain}
        autoMaintainPending={autoMaintainPending}
        onToggleAutoMaintain={handleAutoMaintainToggle}
        graphInspectorActive={graphInspectorVisible}
        viewportHeight={size.height}
        graphBlocksDetailed={graphBlockCards}
        onFragmentFocus={focusOnFragment}
        onBlockFocus={focusOnBlock}
        graphBlocks={(graphSnapshot?.blocks ?? []).map(b => ({
          blockId: b.blockId,
          label: b.label,
          summary: b.summary,
          updatedAt: b.updatedAt,
        }))}
      />
      {graphInspectorVisible && (
        <div
          style={{
            position: 'absolute',
            top: 92,
            right: 370,
            width: 400,
            maxHeight: size.height - 160,
            overflow: 'auto',
            padding: 16,
            borderRadius: 16,
            background: 'rgba(15,23,42,0.75)',
            color: '#e2e8f0',
            boxShadow: '0 12px 36px rgba(15,23,42,0.45)',
            zIndex: 1200,
            backdropFilter: 'blur(10px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ fontSize: 16, fontWeight: 600 }}>Graph Inspector</div>
            <button
              onClick={toggleGraphInspector}
              style={{
                border: 'none',
                background: 'rgba(255,255,255,0.15)',
                color: '#f1f5f9',
                borderRadius: 999,
                padding: '4px 10px',
                cursor: 'pointer',
              }}
            >
              Close
            </button>
          </div>
          {!autoMaintain && (
            <div style={{ fontSize: 13, color: '#cbd5f5' }}>
              启用 Auto Maintain 后即可看到实时知识图更新。
            </div>
          )}
          {autoMaintain && (
            <>
              <div style={{ fontSize: 12, color: '#a5b4fc', marginBottom: 8 }}>
                Blocks: {graphSnapshot?.blocks?.length ?? 0} · Fragments: {graphSnapshot?.fragments?.length ?? 0}
                {graphSnapshot?.groups && graphSnapshot.groups.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 12, color: '#fca5a5', marginBottom: 6 }}>
                      Pending Groups ({graphSnapshot.groups.length})
                    </div>
                    {graphSnapshot.groups.map((group) => (
                      <div
                        key={group.groupId}
                        style={{
                          border: '1px dashed rgba(251,191,36,0.65)',
                          borderRadius: 10,
                          padding: '10px 12px',
                          marginBottom: 8,
                          background: 'rgba(251,191,36,0.12)',
                        }}
                      >
                        <div style={{ fontSize: 12, fontWeight: 600, color: '#fde68a' }}>
                          {group.groupId} · {group.state} · size {group.size}
                        </div>
                        <div style={{ fontSize: 11, color: '#fef3c7', marginTop: 4 }}>
                          touchCount: {group.touchCount} · need LLM review: {group.needLLMReview ? 'yes' : 'no'}
                        </div>
                        <div style={{ fontSize: 10, color: '#fcd34d', marginTop: 4 }}>
                          members: {group.members.slice(0, 4).join(', ')}
                          {group.members.length > 4 ? ' …' : ''}
                        </div>
                        <button
                          onClick={() => promoteGroup(group.groupId)}
                          disabled={promoteGroupPending === group.groupId}
                          style={{
                            marginTop: 8,
                            border: '1px solid rgba(251,191,36,0.8)',
                            background: promoteGroupPending === group.groupId ? 'rgba(251,191,36,0.2)' : 'rgba(251,191,36,0.35)',
                            color: '#78350f',
                            padding: '6px 10px',
                            borderRadius: 999,
                            cursor: promoteGroupPending === group.groupId ? 'wait' : 'pointer',
                          }}
                        >
                          {promoteGroupPending === group.groupId ? 'Promoting…' : 'Promote to Block'}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {(!graphSnapshot || graphSnapshot.blocks.length === 0) ? (
                <div style={{ fontSize: 13, color: '#cbd5f5' }}>
                  暂无已命名的语义块。尝试添加文本或等待 LLM 汇总。
                </div>
              ) : (
                graphSnapshot.blocks.map((block) => (
                  <div
                    key={block.blockId}
                    style={{
                      background: 'rgba(59,130,246,0.12)',
                      borderRadius: 12,
                      padding: 12,
                      marginBottom: 10,
                      border: '1px solid rgba(96,165,250,0.35)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span
                          style={{
                            width: 12,
                            height: 12,
                            borderRadius: '50%',
                            background: blockColorMap[block.blockId] ?? '#38bdf8',
                            boxShadow: `0 0 8px ${hexToRgba(blockColorMap[block.blockId] ?? '#38bdf8', 0.55)}`,
                          }}
                        />
                        <span style={{ fontSize: 13, fontWeight: 600, color: '#bfdbfe' }}>
                          {block.label || block.blockId}
                        </span>
                      </div>
                      <span style={{ fontSize: 11, color: '#93c5fd' }}>
                        {(block.contents?.length ?? 0)} fragments
                      </span>
                    </div>
                    <div style={{ fontSize: 12, lineHeight: 1.5, color: '#e2e8f0' }}>
                      {block.summary || '（暂无摘要）'}
                    </div>
                    {block.contents?.length ? (
                      <div style={{ fontSize: 11, color: '#c4b5fd', marginTop: 6 }}>
                        Fragments: {block.contents.length}
                      </div>
                    ) : null}
                    {block.relationships?.length ? (
                      <div style={{ fontSize: 11, color: '#cbd5f5', marginTop: 6 }}>
                        Links: {block.relationships.map(rel => `${rel.type}→${rel.target}`).join(', ')}
                      </div>
                    ) : null}
                    {block.updatedAt && (
                      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6 }}>
                        {new Date(block.updatedAt).toLocaleString()}
                      </div>
                    )}
                  </div>
                ))
              )}
              {graphSnapshot && graphSnapshot.fragments.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 12, color: '#a5b4fc', marginBottom: 6 }}>Recent Fragments</div>
                  {graphSnapshot.fragments.slice(-6).reverse().map((frag) => (
                    <div
                      key={frag.id}
                      style={{
                        fontSize: 11,
                        color: '#e0f2fe',
                        padding: '6px 8px',
                        borderRadius: 8,
                        background: 'rgba(14,165,233,0.12)',
                        marginBottom: 4,
                      }}
                    >
                      <div><strong>{frag.type}</strong> · {frag.id}</div>
                      {frag.text && <div style={{ marginTop: 2 }}>{frag.text}</div>}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

type GraphBlock = {
  blockId: string
  label: string
  summary: string
  contents: string[]
  relationships: Array<{ target: string; type: string; score: number }>
  updatedAt?: string
}

type GraphGroup = {
  groupId: string
  size: number
  state: string
  needLLMReview?: boolean
  members: string[]
  touchCount: number
  updatedAt?: string
}

type GraphBlockCardFragment = {
  id: string
  type: string
  text: string
  bbox: [number, number, number, number] | null
}

type GraphBlockCard = {
  blockId: string
  label: string
  summary: string
  updatedAt?: string
  color: string
  fragments: GraphBlockCardFragment[]
  bbox: [number, number, number, number] | null
}

type GraphSnapshot = {
  blocks: GraphBlock[]
  fragments: Array<{
    id: string
    type: string
    bbox?: [number, number, number, number] | null
    text?: string | null
    timestamp?: string | null
    blockId?: string | null
    blockLabel?: string | null
  }>
  groups: GraphGroup[]
}

