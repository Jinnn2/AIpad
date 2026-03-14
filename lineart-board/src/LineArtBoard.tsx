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
  Arrow as KArrow,
} from 'react-konva'
import type { AIStrokePayload, AIStrokeV11, ColorName, PromptMode } from './ai/types'
import { normalizeAIStrokePayload, validateAIStrokePayload, COLORS } from './ai/normalize'
import type { ShapeDraft } from './ai/plan'
import { planDrafts } from './ai/plan'
import { chaikin, resampleEvenly, geomMaxDeviationFromChord, mergeCollinear, draftToAIStroke } from './ai/draw'
import { TopToolbar, SidePanel, BottomPanel, SettingsButton, AIFeedSidebar, GraphBlocksDrawer, type AIFeedEntry } from './LineArtUI'
import { computeTextBoxLayout, DEFAULT_TEXTBOX_LINE_HEIGHT, measureTextWidth } from './textbox/layout'
import { hasInlineMarkdownStyle, looksLikeMarkdownText, parseMarkdownDisplayBlocks, parseMarkdownInlineRuns, renderMarkdownToCanvasText } from './textbox/markdown'
import {
  addAcceptedSuggestion,
  addDismissRecord,
  addPreviewRecord,
  addRequestCompleted,
  addRequestFailed,
  addRequestSent,
  bboxIntersects,
  computeShapeBBox,
  createExperimentRun,
  endExperimentRun,
  estimateDraftUsableUnits,
  extractUsageUpdate,
  mergeBBox,
  normalizePhaseId,
  refreshAcceptedSuggestionMutations,
  shapeToSnapshot,
  summarizeExperimentRun,
  updateExperimentPhase,
  type BBox,
  type ExperimentRun,
} from './experiment'
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
const readViteNumber = (key: string, fallback: number, min?: number, max?: number) => {
  try {
    const raw = (import.meta as any)?.env?.[key]
    if (raw === undefined || raw === null || raw === '') return fallback
    let value = Number(raw)
    if (!Number.isFinite(value)) return fallback
    if (typeof min === 'number') value = Math.max(min, value)
    if (typeof max === 'number') value = Math.min(max, value)
    return value
  } catch {
    return fallback
  }
}
const VITE_LLM_MODEL_DEFAULT = (() => {
  try {
    return String((import.meta as any)?.env?.VITE_OPENAI_MODEL ?? '').trim()
  } catch {
    return ''
  }
})()
const VITE_LLM_TEMPERATURE_DEFAULT = readViteNumber('VITE_OPENAI_TEMPERATURE', 0.4, 0, 2)
const VITE_LLM_TOP_P_DEFAULT = readViteNumber('VITE_OPENAI_TOP_P', 0.95, 0, 1)
const VITE_LLM_MAX_TOKENS_DEFAULT = Math.round(readViteNumber('VITE_OPENAI_MAX_TOKENS', 10240, 256, 32768))
type GroupPromoteMode = 'heuristic' | 'hybrid' | 'llm'
type VisionImageMode = 'off' | 'auto' | 'always'
const normalizeGroupPromoteMode = (value: unknown): GroupPromoteMode => {
  const token = String(value ?? '').trim().toLowerCase()
  if (token === 'hybrid' || token === 'llm') return token
  return 'heuristic'
}
const normalizeVisionImageMode = (value: unknown): VisionImageMode => {
  const token = String(value ?? '').trim().toLowerCase()
  if (token === 'off' || token === 'always') return token
  return 'auto'
}
const VITE_GROUP_PROMOTE_MODE_DEFAULT: GroupPromoteMode = (() => {
  try {
    return normalizeGroupPromoteMode((import.meta as any)?.env?.VITE_GRAPH_AGENT_GROUP_PROMOTE_MODE ?? 'heuristic')
  } catch {
    return 'heuristic'
  }
})()
const VITE_VISION_IMAGE_MODE_DEFAULT: VisionImageMode = (() => {
  try {
    return normalizeVisionImageMode((import.meta as any)?.env?.VITE_GRAPH_VISION_IMAGE_MODE ?? 'auto')
  } catch {
    return 'auto'
  }
})()
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
const apiUrl = (path: string) => {
  if (/^https?:/i.test(path)) return path
  if (API_BASE) return withBase(API_BASE, path)
  return withBase(FALLBACK_API_BASE, path)
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
  padding: '7px 10px',
  borderRadius: 10,
  border: '1px solid rgba(148, 163, 184, 0.42)',
  background: 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.94))',
  color: '#0f172a',
  boxShadow: '0 1px 0 rgba(255,255,255,0.9) inset, 0 1px 2px rgba(15,23,42,0.04)',
  outline: 'none',
}
const BUTTON_BASE: React.CSSProperties = {
  padding: '6px 12px',
  borderRadius: 999,
  border: '1px solid rgba(148, 163, 184, 0.34)',
  background: 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
  color: '#0f172a',
  fontWeight: 600,
  boxShadow: '0 1px 0 rgba(255,255,255,0.92) inset, 0 6px 14px rgba(15,23,42,0.06)',
  cursor: 'pointer',
}
const TEXT_LINE_HEIGHT = DEFAULT_TEXTBOX_LINE_HEIGHT
type TextStylePreset = {
  id: 'body' | 'subtitle' | 'title'
  label: string
  fontSize: number
  fontWeight: string
  color: ColorName
}
const TEXT_STYLE_PRESETS: TextStylePreset[] = [
  { id: 'body', label: '正文', fontSize: 18, fontWeight: '400', color: 'black' },
  { id: 'subtitle', label: '副标题', fontSize: 24, fontWeight: '600', color: 'blue' },
  { id: 'title', label: '标题', fontSize: 32, fontWeight: '700', color: 'red' },
]
type TextRole = 'body' | 'subtitle' | 'title'
const parseFontWeight = (raw: string | number | null | undefined) => {
  if (typeof raw === 'number' && Number.isFinite(raw)) return Math.round(raw)
  const token = String(raw ?? '').trim().toLowerCase()
  if (!token) return 400
  if (/^\d+$/.test(token)) return Number(token)
  if (token === 'bold' || token === 'heavy') return 700
  if (token === 'semibold' || token === 'demibold') return 600
  return 400
}
const inferTextRole = (fontSize: number, fontWeight: string): TextRole => {
  const normalizedSize = Number(fontSize) || 16
  const normalizedWeight = parseFontWeight(fontWeight)
  if (normalizedSize >= 30 || (normalizedSize >= 28 && normalizedWeight >= 700)) {
    return 'title'
  }
  if (normalizedSize >= 22 && normalizedWeight >= 600) {
    return 'subtitle'
  }
  return 'body'
}
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
const EXPERIMENT_STORAGE_KEY = 'aipad_experiment_run_v1'
const EXPERIMENT_WIDGET_STORAGE_KEY = 'aipad_experiment_widget_v1'
// Preview entries keep drafts grouped by payload id
type PreviewEntry = {
  payloadId: string
  drafts: ShapeDraft[]
  requestId?: string | null
  phaseId?: string
  promptTokens?: number
  activeBlockIds?: string[]
  planTargetBlockIds?: string[]
}
type TextGrowDir = 'down' | 'up' | 'left' | 'right' | 'right-down'
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
type ProjectListItem = {
  projectId: string
  name: string
  createdAt?: string | null
  updatedAt?: string | null
  lastSavedAt?: string | null
  commitCount?: number
  currentPreviewUpdatedAt?: string | null
  currentPreview?: Record<string, any> | null
  stats?: Record<string, any>
}
type ProjectSnapshotItem = {
  snapshotId: string
  createdAt?: string | null
  note?: string | null
  mime?: string | null
  width?: number | null
  height?: number | null
  imageUrl?: string | null
  bbox?: any
}
type ProjectCurrentPreviewItem = {
  updatedAt?: string | null
  note?: string | null
  mime?: string | null
  width?: number | null
  height?: number | null
  bbox?: any
  imageUrl?: string | null
}
type ProjectCommitItem = {
  commitId: string
  createdAt?: string | null
  message?: string | null
  mime?: string | null
  width?: number | null
  height?: number | null
  bbox?: any
  imageUrl?: string | null
}
type ProjectDetail = {
  projectId: string
  meta?: Record<string, any>
  current?: ProjectCurrentPreviewItem | null
  commits: ProjectCommitItem[]
  legacySnapshots?: ProjectSnapshotItem[]
  snapshots: ProjectSnapshotItem[]
}
type ProjectContextMenuState =
  | {
      kind: 'project'
      projectId: string
      x: number
      y: number
    }
  | {
      kind: 'commit'
      projectId: string
      commitId: string
      x: number
      y: number
    }
type ExperimentWidgetState = {
  open: boolean
  x: number
  y: number
}
export default function LineArtBoard() {
  // Canvas size; swap to ResizeObserver for responsive layout
  const [size] = useState({ width: window.innerWidth, height: window.innerHeight })
  const askAIRef = useRef<null | (() => void)>(null)
  const askInFlightRef = useRef(false)
const stageRef = useRef<any>(null)
const gridLayerRef = useRef<any>(null)
const rootRef = useRef<HTMLDivElement | null>(null)
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
  const mergeWorldBBoxes = useCallback((
    a: [number, number, number, number] | null,
    b: [number, number, number, number] | null,
  ): [number, number, number, number] | null => {
    if (!b) return a
    if (!a) return [...b] as [number, number, number, number]
    const [ax0, ay0, ax1, ay1] = a
    const [bx0, by0, bx1, by1] = b
    return [
      Math.min(ax0, bx0),
      Math.min(ay0, by0),
      Math.max(ax1, bx1),
      Math.max(ay1, by1),
    ]
  }, [])
  const computeStrokeBBox = useCallback((strokes: AIStrokeV11[] | undefined | null) => {
    if (!strokes || strokes.length === 0) return null
    let minX = Number.POSITIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY
    let hit = false
    for (const stroke of strokes) {
      if (!stroke || stroke.tool === 'text') continue
      for (const point of stroke.points || []) {
        if (!point || point.length < 2) continue
        const px = Number(point[0])
        const py = Number(point[1])
        if (!Number.isFinite(px) || !Number.isFinite(py)) continue
        hit = true
        if (px < minX) minX = px
        if (py < minY) minY = py
        if (px > maxX) maxX = px
        if (py > maxY) maxY = py
      }
    }
    if (!hit) return null
    return [minX, minY, maxX, maxY] as [number, number, number, number]
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
  // When enabled, encourage LLM to add explanatory sketches/diagrams to existing content.
  const [preferExplanatoryDrawing, setPreferExplanatoryDrawing] = useState<boolean>(false)
  // Runtime LLM controls (editable while app is running).
  const [llmModel, setLlmModel] = useState<string>(VITE_LLM_MODEL_DEFAULT)
  const [llmTemperature, setLlmTemperature] = useState<number>(VITE_LLM_TEMPERATURE_DEFAULT)
  const [llmTopP, setLlmTopP] = useState<number>(VITE_LLM_TOP_P_DEFAULT)
  const [llmMaxTokens, setLlmMaxTokens] = useState<number>(VITE_LLM_MAX_TOKENS_DEFAULT)
  const [groupPromoteMode, setGroupPromoteMode] = useState<GroupPromoteMode>(VITE_GROUP_PROMOTE_MODE_DEFAULT)
  const [visionImageMode, setVisionImageMode] = useState<VisionImageMode>(VITE_VISION_IMAGE_MODE_DEFAULT)
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
    growDir: 'right-down',
  })
  const [textEditor, setTextEditor] = useState<TextEditorState | null>(null)
  const [selectedShapeId, setSelectedShapeId] = useState<string | null>(null)
  const [selectDeleteDragActive, setSelectDeleteDragActive] = useState(false)
  const [selectDeleteHover, setSelectDeleteHover] = useState(false)
  const [completionPreviews, setCompletionPreviews] = useState<Record<string, string>>({})
  const updateTextSettings = useCallback((patch: Partial<TextSettings>) => {
    setTextSettings((prev) => ({ ...prev, ...patch }))
  }, [])
  const resetRuntimeLLMSettings = useCallback(() => {
    setLlmModel(VITE_LLM_MODEL_DEFAULT)
    setLlmTemperature(VITE_LLM_TEMPERATURE_DEFAULT)
    setLlmTopP(VITE_LLM_TOP_P_DEFAULT)
    setLlmMaxTokens(VITE_LLM_MAX_TOKENS_DEFAULT)
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
  // ----- Snapshot stage canvas to JPEG/PNG Base64 (supports cropped capture) -----
  type SnapshotOptions = {
    bbox?: [number, number, number, number]
    padding?: number
    hideGrid?: boolean
    background?: string | null
  }

  const snapshotCanvas = useCallback(async (
    maxSize = 768,
    mime: "image/jpeg" | "image/png" = "image/jpeg",
    quality = 0.6,
    options?: SnapshotOptions,
  ): Promise<{ data: string | null; w: number; h: number; mime: string }> => {
    const stage = stageRef.current
    if (!stage) return { data: null, w: 0, h: 0, mime }
    const toScreenRect = (bbox: [number, number, number, number], padding = 0) => {
      const [x0, y0, x1, y1] = bbox
      const minX = Math.min(x0, x1) - padding
      const minY = Math.min(y0, y1) - padding
      const maxX = Math.max(x0, x1) + padding
      const maxY = Math.max(y0, y1) + padding
      const width = Math.max(4, maxX - minX)
      const height = Math.max(4, maxY - minY)
      const screenX = minX * view.scale + view.x
      const screenY = minY * view.scale + view.y
      const screenW = Math.max(16, width * view.scale)
      const screenH = Math.max(16, height * view.scale)
      return { x: screenX, y: screenY, width: screenW, height: screenH }
    }
    const cropRect = options?.bbox ? toScreenRect(options.bbox, options.padding ?? 0) : null
    const gridLayer = gridLayerRef.current
    const shouldHideGrid = options?.hideGrid ?? false
    let previousGridVisible: boolean | undefined
    if (shouldHideGrid && gridLayer && typeof gridLayer.visible === 'function') {
      previousGridVisible = gridLayer.visible()
      if (previousGridVisible) {
        gridLayer.visible(false)
        stage.draw()
      }
    }
    try {
      const pixelRatio = Math.min(2, window.devicePixelRatio || 1)
      const rawCanvas: HTMLCanvasElement = cropRect
        ? stage.toCanvas({
            x: cropRect.x,
            y: cropRect.y,
            width: cropRect.width,
            height: cropRect.height,
            pixelRatio,
          })
        : stage.toCanvas({ pixelRatio })
      const srcWidth = rawCanvas.width
      const srcHeight = rawCanvas.height
      if (!srcWidth || !srcHeight) {
        return { data: null, w: 0, h: 0, mime }
      }
      const scale = Math.min(1, maxSize / Math.max(srcWidth, srcHeight))
      const targetWidth = Math.round(srcWidth * scale)
      const targetHeight = Math.round(srcHeight * scale)
      const targetCanvas = document.createElement("canvas")
      targetCanvas.width = targetWidth
      targetCanvas.height = targetHeight
      const ctx = targetCanvas.getContext("2d")
      if (!ctx) return { data: null, w: 0, h: 0, mime }
      const background = options?.background !== undefined
        ? options.background
        : (mime === "image/jpeg" ? "#ffffff" : null)
      if (background) {
        ctx.fillStyle = background
        ctx.fillRect(0, 0, targetWidth, targetHeight)
      }
      ctx.drawImage(rawCanvas, 0, 0, targetWidth, targetHeight)
      const outUri = targetCanvas.toDataURL(mime, quality)
      const base64 = outUri.split(",")[1] || null
      return { data: base64, w: targetWidth, h: targetHeight, mime }
    } finally {
      if (shouldHideGrid && gridLayer && typeof gridLayer.visible === 'function' && previousGridVisible !== undefined) {
        gridLayer.visible(previousGridVisible)
        stage.draw()
      }
    }
  }, [view])
  type GraphSnapshotUpload = {
    bbox: [number, number, number, number]
    width: number
    height: number
    mime: string
    data: string
  }
  const captureGraphSnapshotPayload = useCallback(async (
    bbox: [number, number, number, number],
    opts?: { maxSize?: number; mime?: "image/jpeg" | "image/png"; quality?: number; padding?: number; background?: string | null },
  ): Promise<GraphSnapshotUpload | null> => {
    const snap = await snapshotCanvas(opts?.maxSize ?? 1024, opts?.mime ?? "image/jpeg", opts?.quality ?? 0.72, {
      bbox,
      padding: opts?.padding ?? 72,
      hideGrid: true,
      background: opts?.background,
    })
    if (!snap.data) return null
    return {
      bbox,
      width: snap.w,
      height: snap.h,
      mime: snap.mime,
      data: snap.data,
    }
  }, [snapshotCanvas])
  const captureProjectSaveSnapshotPayload = useCallback(async (): Promise<GraphSnapshotUpload | null> => {
    const snap = await snapshotCanvas(1400, "image/jpeg", 0.82, {
      hideGrid: true,
      background: "#ffffff",
    })
    if (!snap.data) return null
    const worldLeft = (-view.x) / view.scale
    const worldTop = (-view.y) / view.scale
    const worldWidth = size.width / view.scale
    const worldHeight = size.height / view.scale
    return {
      bbox: [worldLeft, worldTop, worldLeft + worldWidth, worldTop + worldHeight],
      width: snap.w,
      height: snap.h,
      mime: snap.mime,
      data: snap.data,
    }
  }, [snapshotCanvas, view.x, view.y, view.scale, size.width, size.height])
  // Store AI previews grouped by payload id
  const [previews, setPreviews] = useState<Record<string, PreviewEntry>>({})
  const lastCommittedShapeCountRef = useRef(shapes.length)
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
  const [aiFeedSidebarOpen, setAiFeedSidebarOpen] = useState<boolean>(false)
  const [projectManagerOpen, setProjectManagerOpen] = useState<boolean>(false)
  const [projectManagerBusy, setProjectManagerBusy] = useState<boolean>(false)
  const [projectManagerError, setProjectManagerError] = useState<string>('')
  const [projectNameDraft, setProjectNameDraft] = useState<string>('')
  const [projectPromptOpen, setProjectPromptOpen] = useState<boolean>(false)
  const [projectPromptNameDraft, setProjectPromptNameDraft] = useState<string>('')
  const [projectPromptSubmitting, setProjectPromptSubmitting] = useState<boolean>(false)
  const [projectList, setProjectList] = useState<ProjectListItem[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string>('')
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null)
  const [currentProjectName, setCurrentProjectName] = useState<string | null>(null)
  const [projectDetail, setProjectDetail] = useState<ProjectDetail | null>(null)
  const [projectActionPending, setProjectActionPending] = useState<'create' | 'save' | 'commit' | 'open' | 'checkout' | 'delete-project' | 'delete-commit' | 'current-snapshot' | null>(null)
  const [projectSavePending, setProjectSavePending] = useState<boolean>(false)
  const [projectSaveFlash, setProjectSaveFlash] = useState<boolean>(false)
  const projectSaveFlashTimerRef = useRef<number | null>(null)
  const [projectCommitMessageDraft, setProjectCommitMessageDraft] = useState<string>('')
  const [projectCurrentPreviewDirty, setProjectCurrentPreviewDirty] = useState<boolean>(false)
  const [projectContextMenuState, setProjectContextMenuState] = useState<ProjectContextMenuState | null>(null)
  const [plannerNextStepHint, setPlannerNextStepHint] = useState<string>('')
  const [experimentPhaseId, setExperimentPhaseId] = useState<string>('phase-1')
  const [experimentRun, setExperimentRun] = useState<ExperimentRun | null>(() => {
    try {
      const raw = localStorage.getItem(EXPERIMENT_STORAGE_KEY)
      if (!raw) return null
      const parsed = JSON.parse(raw) as ExperimentRun
      return parsed && typeof parsed === 'object' ? parsed : null
    } catch {
      return null
    }
  })
  const [experimentWidget, setExperimentWidget] = useState<ExperimentWidgetState>(() => {
    try {
      const raw = localStorage.getItem(EXPERIMENT_WIDGET_STORAGE_KEY)
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<ExperimentWidgetState>
        if (typeof parsed?.x === 'number' && typeof parsed?.y === 'number') {
          return {
            open: parsed.open !== false,
            x: parsed.x,
            y: parsed.y,
          }
        }
      }
    } catch {}
    return {
      open: true,
      x: Math.max(12, window.innerWidth - 360 - 72),
      y: 76,
    }
  })
  const experimentRunRef = useRef<ExperimentRun | null>(experimentRun)
  const experimentDragRef = useRef<{
    pointerId: number
    startX: number
    startY: number
    baseX: number
    baseY: number
    moved: boolean
  } | null>(null)
  const experimentDragSuppressClickRef = useRef(false)
  const commitExperimentRun = useCallback((updater: (current: ExperimentRun) => ExperimentRun) => {
    const current = experimentRunRef.current
    if (!current || current.endedAt) return null
    const next = updater(current)
    experimentRunRef.current = next
    setExperimentRun(next)
    return next
  }, [])
  const replaceExperimentRun = useCallback((next: ExperimentRun | null) => {
    experimentRunRef.current = next
    setExperimentRun(next)
  }, [])
  React.useEffect(() => {
    experimentRunRef.current = experimentRun
  }, [experimentRun])
  React.useEffect(() => {
    if (experimentRun?.currentPhaseId) {
      setExperimentPhaseId(experimentRun.currentPhaseId)
    }
  }, [])
  React.useEffect(() => {
    try {
      if (!experimentRun) {
        localStorage.removeItem(EXPERIMENT_STORAGE_KEY)
      } else {
        localStorage.setItem(EXPERIMENT_STORAGE_KEY, JSON.stringify(experimentRun))
      }
    } catch {}
  }, [experimentRun])
  React.useEffect(() => {
    try {
      localStorage.setItem(EXPERIMENT_WIDGET_STORAGE_KEY, JSON.stringify(experimentWidget))
    } catch {}
  }, [experimentWidget])
  // Session identifiers from backend; lastSentIndex tracks delta uploads
  const [sid, setSid] = useState<string | null>(null)
  const [visionVersion, setVisionVersion] = useState<number>(2.0)
  const lastSentIndexRef = useRef<number>(0)
  // Debounce timer handle for session sync
  const syncTimerRef = useRef<number | null>(null)
  const projectAutoSnapshotTimerRef = useRef<number | null>(null)
  const projectSkipNextDirtyRef = useRef(false)
  const graphCaptureBBoxRef = useRef<[number, number, number, number] | null>(null)
  const graphKnownStrokeIdsRef = useRef<Set<string>>(new Set())
  const autoMaintainRef = useRef(false)
  // Helper for three-decimal rounding to shrink payloads (backend also clamps)
  const round3 = (v:number) => Math.round(v * 1000) / 1000
  // Pack drawStack.ai (absolute coordinates) into protocol-friendly strokes
  const packAllStrokes = useCallback((): AIStrokeV11[] => {
    return drawStack.map((entry) => {
      const nextPoints = (entry.ai.points || []).map(([x, y, t, p]) => [
        round3(x),
        round3(y),
        t,
        p,
      ]) as Array<[number, number, number?, number?]>
      return {
        ...entry.ai,
        points: nextPoints,
      }
    })
  }, [drawStack])
  // -------- Undo/redo snapshot stacks --------
  type Snapshot = { shapes: ShapeDraft[]; drawStack: DrawStackEntry[] }
  const HISTORY_LIMIT = 30
  const [past, setPast] = useState<Snapshot[]>([])
  const [future, setFuture] = useState<Snapshot[]>([])
  const pushHistory = useCallback((snap?: Snapshot) => {
    const nextSnap = snap ?? { shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }
    setPast(p => {
      const next = [...p, nextSnap]
      return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next
    })
    setFuture([]) // Clear redo branch after a new action
  }, [shapes, drawStack])
  const undo = useCallback(() => {
    setPast(p => {
      if (p.length === 0) return p
      const last = p[p.length - 1]
      setFuture(f => {
        const next = [{ shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }, ...f]
        return next.length > HISTORY_LIMIT ? next.slice(0, HISTORY_LIMIT) : next
      })
      setShapes(JSON.parse(JSON.stringify(last.shapes)))
      setDrawStack(JSON.parse(JSON.stringify(last.drawStack)))
      return p.slice(0, -1)
    })
  }, [shapes, drawStack])
  const redo = useCallback(() => {
    setFuture(f => {
      if (f.length === 0) return f
      const head = f[0]
      setPast(p => {
        const next = [...p, { shapes: JSON.parse(JSON.stringify(shapes)), drawStack: JSON.parse(JSON.stringify(drawStack)) }]
        return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next
      })
      setShapes(JSON.parse(JSON.stringify(head.shapes)))
      setDrawStack(JSON.parse(JSON.stringify(head.drawStack)))
      return f.slice(1)
    })
  }, [shapes, drawStack])
  const [suspendSessionSync, setSuspendSessionSync] = useState<boolean>(false)
  React.useEffect(() => {
    if (toolMode === 'select') return
    setSuspendSessionSync(false)
  }, [toolMode])
  // ----- Auto-sync drawStack to backend session (debounced) -----
  const syncSession = useCallback(async (curSid: string) => {
    try {
      const strokes = packAllStrokes()
      let graphSnapshotPayload: GraphSnapshotUpload | null = null
      const pendingBBox = autoMaintainRef.current ? graphCaptureBBoxRef.current : null
      if (pendingBBox) {
        graphCaptureBBoxRef.current = null
        try {
          graphSnapshotPayload = await captureGraphSnapshotPayload(pendingBBox)
        } catch (err) {
          console.warn('[graph] snapshot capture failed:', err)
        }
      }
      const body: Record<string, unknown> = { sid: curSid, strokes }
      body.vision_image_mode = visionImageMode
      if (graphSnapshotPayload) {
        body.graph_snapshot = graphSnapshotPayload
      }
      const res = await apiFetch('/session/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) console.warn('session/sync failed', await res.text())
    } catch (err) {
      console.warn('session/sync error', err)
    }
  }, [packAllStrokes, captureGraphSnapshotPayload, visionImageMode])
  React.useEffect(() => {
    if (!sid) return
    if (syncTimerRef.current) window.clearTimeout(syncTimerRef.current)
    if (suspendSessionSync) return
    syncTimerRef.current = window.setTimeout(() => {
      syncSession(sid)
    }, 1000) as unknown as number
    return () => {
      if (syncTimerRef.current) {
        window.clearTimeout(syncTimerRef.current)
        syncTimerRef.current = null
      }
    }
  }, [drawStack, sid, syncSession, suspendSessionSync])
  // ----- Auto-complete toggle & countdown (5s) -----
  const [autoComplete, setAutoComplete] = useState<boolean>(false)
  const [autoCountdown, setAutoCountdown] = useState<number|null>(null)
  const [autoMaintain, setAutoMaintain] = useState<boolean>(false)
  const [autoMaintainPending, setAutoMaintainPending] = useState<boolean>(false)
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false)
  const [graphBlocksDrawerOpen, setGraphBlocksDrawerOpen] = useState<boolean>(false)
  const [graphSnapshot, setGraphSnapshot] = useState<GraphSnapshot | null>(null)
  const [graphInspectorVisible, setGraphInspectorVisible] = useState<boolean>(false)
  const [hoveredGraphBlockId, setHoveredGraphBlockId] = useState<string | null>(null)
  const [hoveredGraphFragmentId, setHoveredGraphFragmentId] = useState<string | null>(null)
  const [graphBlockSelectionMode, setGraphBlockSelectionMode] = useState<boolean>(false)
  const [graphSelectionRectScreen, setGraphSelectionRectScreen] = useState<{
    x0: number
    y0: number
    x1: number
    y1: number
  } | null>(null)
  const graphSelectionDragRef = useRef<{ pointerId: number; x0: number; y0: number } | null>(null)
  const graphSelectionRectRafRef = useRef<number | null>(null)
  const graphSelectionRectPendingRef = useRef<{
    x0: number
    y0: number
    x1: number
    y1: number
  } | null>(null)
  const [graphSelectionDragging, setGraphSelectionDragging] = useState<boolean>(false)
  const [graphSelectedFragmentIds, setGraphSelectedFragmentIds] = useState<string[]>([])
  const [graphSelectionActionPending, setGraphSelectionActionPending] = useState<'create_block' | 'assign_block' | null>(null)
  const [graphSelectionTargetBlockId, setGraphSelectionTargetBlockId] = useState<string>('')
  const bottomPanelHeight = useMemo(() => {
    const expandedHeight = Math.max(size.height * 0.5, 360)
    return graphInspectorVisible
      ? Math.min(expandedHeight, size.height - 120)
      : 220
  }, [graphInspectorVisible, size.height])
  const selectDeleteZone = useMemo(() => {
    const width = Math.max(220, Math.min(360, Math.round(size.width * 0.28)))
    const height = 76
    const bottom = 14 + bottomPanelHeight + 12
    const left = Math.max(16, Math.round((size.width - width) / 2))
    const top = Math.max(16, size.height - bottom - height)
    return { left, top, width, height }
  }, [size.width, size.height, bottomPanelHeight])
  const isPointInSelectDeleteZone = useCallback((screenX: number, screenY: number) => {
    return (
      screenX >= selectDeleteZone.left &&
      screenX <= selectDeleteZone.left + selectDeleteZone.width &&
      screenY >= selectDeleteZone.top &&
      screenY <= selectDeleteZone.top + selectDeleteZone.height
    )
  }, [selectDeleteZone])
  React.useEffect(() => {
    autoMaintainRef.current = autoMaintain
    if (!autoMaintain) {
      graphCaptureBBoxRef.current = null
      setPlannerNextStepHint('')
    }
  }, [autoMaintain])
  React.useEffect(() => {
    const prevKnown = graphKnownStrokeIdsRef.current
    const nextKnown = new Set<string>()
    const newlyAdded: AIStrokeV11[] = []
    drawStack.forEach((entry, idx) => {
      const stroke = entry.ai
      if (!stroke) return
      const strokeId = stroke.id ? String(stroke.id) : `stroke-${idx}`
      nextKnown.add(strokeId)
      if (!prevKnown.has(strokeId)) {
        newlyAdded.push(stroke)
      }
    })
    graphKnownStrokeIdsRef.current = nextKnown
    if (!autoMaintain || newlyAdded.length === 0) return
    const bbox = computeStrokeBBox(newlyAdded)
    if (!bbox) return
    graphCaptureBBoxRef.current = mergeWorldBBoxes(graphCaptureBBoxRef.current, bbox)
  }, [drawStack, autoMaintain, computeStrokeBBox, mergeWorldBBoxes])
  React.useEffect(() => {
    if (!graphInspectorVisible) {
      setHoveredGraphBlockId(null)
      setHoveredGraphFragmentId(null)
      setGraphBlocksDrawerOpen(false)
      setGraphBlockSelectionMode(false)
      setGraphSelectionRectScreen(null)
      setGraphSelectedFragmentIds([])
      graphSelectionDragRef.current = null
      graphSelectionRectPendingRef.current = null
      setGraphSelectionDragging(false)
      if (graphSelectionRectRafRef.current != null) {
        window.cancelAnimationFrame(graphSelectionRectRafRef.current)
        graphSelectionRectRafRef.current = null
      }
    }
  }, [graphInspectorVisible])
  React.useEffect(() => {
    if (autoMaintain) return
    setGraphBlocksDrawerOpen(false)
    setGraphBlockSelectionMode(false)
    setGraphSelectionRectScreen(null)
    setGraphSelectedFragmentIds([])
    graphSelectionDragRef.current = null
    graphSelectionRectPendingRef.current = null
    setGraphSelectionDragging(false)
    if (graphSelectionRectRafRef.current != null) {
      window.cancelAnimationFrame(graphSelectionRectRafRef.current)
      graphSelectionRectRafRef.current = null
    }
  }, [autoMaintain])
  React.useEffect(() => {
    if (graphBlockSelectionMode) return
    graphSelectionDragRef.current = null
    graphSelectionRectPendingRef.current = null
    setGraphSelectionDragging(false)
    if (graphSelectionRectRafRef.current != null) {
      window.cancelAnimationFrame(graphSelectionRectRafRef.current)
      graphSelectionRectRafRef.current = null
    }
  }, [graphBlockSelectionMode])
  React.useEffect(() => {
    return () => {
      if (graphSelectionRectRafRef.current != null) {
        window.cancelAnimationFrame(graphSelectionRectRafRef.current)
        graphSelectionRectRafRef.current = null
      }
    }
  }, [])
  React.useEffect(() => {
    if (graphInspectorVisible && autoMaintain) {
      setGraphBlocksDrawerOpen(true)
    }
  }, [graphInspectorVisible, autoMaintain])
  const [promoteGroupPending, setPromoteGroupPending] = useState<string | null>(null)
  const [promoteVisionGroupPending, setPromoteVisionGroupPending] = useState<string | null>(null)
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
  React.useEffect(() => {
    if (autoMaintainPending) {
      clearAutoTimer()
    }
  }, [autoMaintainPending, clearAutoTimer])
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
        visionPendingGroups: Array.isArray((data as any)?.visionPendingGroups) ? (data as any).visionPendingGroups : [],
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
      payload.vision_image_mode = visionImageMode
      if (nextEnabled) {
        payload.canvas_size = [size.width, size.height]
        const allStrokes = packAllStrokes()
        payload.strokes = allStrokes
        const bbox = computeStrokeBBox(allStrokes)
        if (bbox) {
          try {
            const snapped = await captureGraphSnapshotPayload(bbox, { maxSize: 1400 })
            if (snapped) {
              payload.graph_snapshot = snapped
            }
          } catch (err) {
            console.warn('[graph] initial snapshot failed:', err)
          }
        }
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
        const strokePayloads = (payload.strokes as AIStrokeV11[] | undefined) ?? []
        graphKnownStrokeIdsRef.current = new Set(strokePayloads.map((s) => String(s.id)))
        graphCaptureBBoxRef.current = null
        if (curSid) await fetchGraphSnapshot(curSid)
      } else {
        setAutoMaintain(false)
        setGraphSnapshot(null)
        graphCaptureBBoxRef.current = null
      }
    } catch (err: any) {
      console.warn('[auto-maintain] error:', err)
      if (!quiet) {
        alert('Automatic maintenance switchover failed:\n' + (err?.message || String(err)))
      }
      if (!nextEnabled) {
        setAutoMaintain(false)
      }
    } finally {
      if (!quiet) setAutoMaintainPending(false)
      if (!nextEnabled) {
        setAutoMaintain(false)
        setGraphSnapshot(null)
        graphCaptureBBoxRef.current = null
      }
    }
  }, [
    autoMaintainPending,
    sid,
    size.width,
    size.height,
    packAllStrokes,
    hint,
    fetchGraphSnapshot,
    computeStrokeBBox,
    captureGraphSnapshotPayload,
    visionImageMode,
  ])
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
      const positionBBox = Array.isArray(block.position) && block.position.length === 4
        ? (block.position as [number, number, number, number])
        : null
      candidateIds.forEach((fragId) => {
        const frag = fragmentMap.get(fragId)
        if (!frag) return
        const lowerType = String(frag.type || '').toLowerCase()
        const shape = shapeById.get(frag.id) ?? null
        const shapeBBox = resolveShapeBBox(shape)
        const fragBBox =
          frag.bbox && frag.bbox.length === 4
            ? (frag.bbox as [number, number, number, number])
            : shapeBBox
        const rawText = (frag.text ?? shape?.summary ?? shape?.text ?? '').toString().trim()
        const summary =
          rawText.length > 120 ? `${rawText.slice(0, 120)}¡­` : rawText || '(ÎÞÕªÒª)'
        entries.push({
          id: frag.id,
          type: lowerType || 'unknown',
          text: summary,
          bbox: fragBBox ?? null,
        })
        blockBBox = mergeBBox(blockBBox, fragBBox ?? shapeBBox ?? null)
      })
      blockBBox = mergeBBox(blockBBox, positionBBox)
      const center = blockBBox
        ? { x: (blockBBox[0] + blockBBox[2]) / 2, y: (blockBBox[1] + blockBBox[3]) / 2 }
        : positionBBox
          ? { x: (positionBBox[0] + positionBBox[2]) / 2, y: (positionBBox[1] + positionBBox[3]) / 2 }
          : null
      const relations = Array.isArray(block.relationships)
        ? block.relationships
            .filter((rel) => rel && rel.target)
            .map((rel) => ({
              target: String(rel.target),
              type: rel.type ? String(rel.type) : 'related',
              score: typeof rel.score === 'number' ? rel.score : undefined,
            }))
        : []
      return {
        blockId: block.blockId,
        label: block.label,
        summary: block.summary,
        updatedAt: block.updatedAt,
        color: blockColor,
        fragments: entries,
        bbox: blockBBox,
        center,
        relationships: relations,
      }
    })
  }, [graphSnapshot, blockColorMap, shapeById])
  const experimentPanelWidth = useMemo(() => Math.min(360, Math.max(300, size.width * 0.24)), [size.width])
  const experimentChipWidth = 188
  const clampExperimentWidgetPosition = useCallback((x: number, y: number, open: boolean) => {
    const width = open ? experimentPanelWidth : experimentChipWidth
    const clampedX = clamp(x, 12, Math.max(12, size.width - width - 12))
    const clampedY = clamp(y, 76, Math.max(76, size.height - 56))
    return { x: clampedX, y: clampedY }
  }, [experimentPanelWidth, size.height, size.width])
  const experimentActive = Boolean(experimentRun && !experimentRun.endedAt)
  const experimentSummary = useMemo(() => (
    experimentRun ? summarizeExperimentRun(experimentRun, shapes) : null
  ), [experimentRun, shapes])
  React.useEffect(() => {
    setExperimentWidget((current) => {
      const nextPos = clampExperimentWidgetPosition(current.x, current.y, current.open)
      if (nextPos.x === current.x && nextPos.y === current.y) return current
      return { ...current, ...nextPos }
    })
  }, [clampExperimentWidgetPosition])
  React.useEffect(() => {
    if (!experimentActive) return
    commitExperimentRun((current) => refreshAcceptedSuggestionMutations(current, shapes))
  }, [experimentActive, commitExperimentRun, shapes])
  const handleExperimentPhaseChange = useCallback((value: string) => {
    const normalized = normalizePhaseId(value)
    setExperimentPhaseId(normalized)
    if (!experimentActive) return
    commitExperimentRun((current) => updateExperimentPhase(current, normalized))
  }, [commitExperimentRun, experimentActive])
  const startExperiment = useCallback(() => {
    const phaseId = normalizePhaseId(experimentPhaseId)
    setExperimentPhaseId(phaseId)
    replaceExperimentRun(createExperimentRun(phaseId, sid))
  }, [experimentPhaseId, replaceExperimentRun, sid])
  const endExperiment = useCallback(() => {
    if (!experimentRunRef.current || experimentRunRef.current.endedAt) return
    replaceExperimentRun(endExperimentRun(experimentRunRef.current))
  }, [replaceExperimentRun])
  const exportExperiment = useCallback(() => {
    if (!experimentRunRef.current) return
    const summary = summarizeExperimentRun(experimentRunRef.current, shapes)
    const payload = {
      exportedAt: new Date().toISOString(),
      run: experimentRunRef.current,
      summary,
      graphBlockCount: graphBlockCards.length,
      currentShapeCount: shapes.length,
      config: {
        editThreshold: experimentRunRef.current.editThreshold,
        textUnitChars: 20,
      },
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(blob)
    anchor.download = `aipad-experiment-${experimentRunRef.current.runId}.json`
    anchor.click()
    URL.revokeObjectURL(anchor.href)
  }, [graphBlockCards.length, shapes])
  const openExperimentWidget = useCallback(() => {
    setExperimentWidget((current) => {
      const nextPos = clampExperimentWidgetPosition(current.x, current.y, true)
      return { ...current, open: true, ...nextPos }
    })
  }, [clampExperimentWidgetPosition])
  const closeExperimentWidget = useCallback(() => {
    setExperimentWidget((current) => ({ ...current, open: false }))
  }, [])
  const onExperimentChipPointerDown = useCallback((ev: React.PointerEvent<HTMLButtonElement>) => {
    experimentDragRef.current = {
      pointerId: ev.pointerId,
      startX: ev.clientX,
      startY: ev.clientY,
      baseX: experimentWidget.x,
      baseY: experimentWidget.y,
      moved: false,
    }
    experimentDragSuppressClickRef.current = false
    try { ev.currentTarget.setPointerCapture(ev.pointerId) } catch {}
  }, [experimentWidget.x, experimentWidget.y])
  const onExperimentChipPointerMove = useCallback((ev: React.PointerEvent<HTMLButtonElement>) => {
    const drag = experimentDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    const dx = ev.clientX - drag.startX
    const dy = ev.clientY - drag.startY
    if (!drag.moved && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) {
      drag.moved = true
      experimentDragSuppressClickRef.current = true
    }
    if (!drag.moved) return
    const nextPos = clampExperimentWidgetPosition(drag.baseX + dx, drag.baseY + dy, false)
    setExperimentWidget((current) => ({ ...current, ...nextPos }))
  }, [clampExperimentWidgetPosition])
  const onExperimentChipPointerUp = useCallback((ev: React.PointerEvent<HTMLButtonElement>) => {
    const drag = experimentDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    experimentDragRef.current = null
    try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch {}
  }, [])
  const onExperimentChipPointerCancel = useCallback((ev: React.PointerEvent<HTMLButtonElement>) => {
    const drag = experimentDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    experimentDragRef.current = null
    try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch {}
  }, [])
  const recordExperimentRequestSent = useCallback((requestMode: PromptMode) => {
    if (!experimentActive) return null
    let requestId: string | null = null
    commitExperimentRun((current) => {
      const recorded = addRequestSent(current, {
        phaseId: experimentPhaseId,
        sessionId: sid,
        requestMode,
      })
      requestId = recorded.requestId
      return recorded.run
    })
    return requestId
  }, [commitExperimentRun, experimentActive, experimentPhaseId, sid])
  const recordExperimentRequestCompleted = useCallback((requestId: string | null, rawUsage: any) => {
    if (!requestId || !experimentActive) return
    const update = extractUsageUpdate(rawUsage)
    commitExperimentRun((current) => addRequestCompleted(current, requestId, update))
  }, [commitExperimentRun, experimentActive])
  const recordExperimentRequestFailed = useCallback((requestId: string | null, errorMessage: string) => {
    if (!requestId || !experimentActive) return
    commitExperimentRun((current) => addRequestFailed(current, requestId, errorMessage))
  }, [commitExperimentRun, experimentActive])
  const recordExperimentPreview = useCallback((payloadId: string, requestId: string | null, phaseId: string) => {
    if (!experimentActive) return
    commitExperimentRun((current) => addPreviewRecord(current, { payloadId, requestId, phaseId }))
  }, [commitExperimentRun, experimentActive])
  const recordExperimentDismiss = useCallback((payloadId: string, requestId: string | null, phaseId: string, reason?: string | null) => {
    if (!experimentActive) return
    commitExperimentRun((current) => addDismissRecord(current, { payloadId, requestId, phaseId, reason }))
  }, [commitExperimentRun, experimentActive])
  const graphSelectedFragments = useMemo(() => {
    const selected = new Set(graphSelectedFragmentIds)
    if (!selected.size) return [] as Array<{
      id: string
      type: string
      text?: string | null
      bbox: [number, number, number, number]
      blockId?: string | null
      blockLabel?: string | null
    }>
    const source = graphSnapshot?.fragments ?? []
    return source
      .filter((frag) => selected.has(String(frag.id)))
      .map((frag) => {
        const bbox = Array.isArray(frag.bbox) && frag.bbox.length === 4
          ? (frag.bbox as [number, number, number, number])
          : null
        if (!bbox) return null
        return {
          id: String(frag.id),
          type: String(frag.type || 'unknown'),
          text: frag.text ?? null,
          bbox,
          blockId: frag.blockId ?? null,
          blockLabel: frag.blockLabel ?? null,
        }
      })
      .filter((frag): frag is NonNullable<typeof frag> => Boolean(frag))
  }, [graphSnapshot?.fragments, graphSelectedFragmentIds])
  React.useEffect(() => {
    const blocks = graphSnapshot?.blocks ?? []
    if (!blocks.length) {
      setGraphSelectionTargetBlockId('')
      return
    }
    setGraphSelectionTargetBlockId((prev) => (
      prev && blocks.some((block) => block.blockId === prev)
        ? prev
        : String(blocks[0].blockId)
    ))
  }, [graphSnapshot?.blocks])
  React.useEffect(() => {
    const validIds = new Set((graphSnapshot?.fragments ?? []).map((frag) => String(frag.id)))
    setGraphSelectedFragmentIds((prev) => {
      const next = prev.filter((fid) => validIds.has(fid))
      return next.length === prev.length ? prev : next
    })
  }, [graphSnapshot?.fragments])
  const selectGraphFragmentsByScreenRect = useCallback((rect: { x0: number; y0: number; x1: number; y1: number }) => {
    const left = Math.min(rect.x0, rect.x1)
    const right = Math.max(rect.x0, rect.x1)
    const top = Math.min(rect.y0, rect.y1)
    const bottom = Math.max(rect.y0, rect.y1)
    const width = right - left
    const height = bottom - top
    if (width < 4 || height < 4) {
      return
    }
    const p0 = screenToWorld(left, top)
    const p1 = screenToWorld(right, bottom)
    const wx0 = Math.min(p0.x, p1.x)
    const wy0 = Math.min(p0.y, p1.y)
    const wx1 = Math.max(p0.x, p1.x)
    const wy1 = Math.max(p0.y, p1.y)
    const hits = (graphSnapshot?.fragments ?? [])
      .filter((frag) => Array.isArray(frag.bbox) && frag.bbox.length === 4)
      .filter((frag) => {
        const [fx0, fy0, fx1, fy1] = frag.bbox as [number, number, number, number]
        const ix = Math.max(0, Math.min(wx1, fx1) - Math.max(wx0, fx0))
        const iy = Math.max(0, Math.min(wy1, fy1) - Math.max(wy0, fy0))
        if (ix <= 0 || iy <= 0) return false
        const interArea = ix * iy
        const fragArea = Math.max(1, Math.abs((fx1 - fx0) * (fy1 - fy0)))
        return interArea >= 16 || interArea / fragArea >= 0.08
      })
      .map((frag) => String(frag.id))
    setGraphSelectedFragmentIds(hits)
  }, [graphSnapshot?.fragments, screenToWorld])
  const clipPointToRect = useCallback(
    (
      origin: { x: number; y: number },
      bbox: [number, number, number, number] | null,
      toward: { x: number; y: number },
    ) => {
      if (!bbox) return origin
      const dx = toward.x - origin.x
      const dy = toward.y - origin.y
      const len = Math.hypot(dx, dy)
      if (!Number.isFinite(len) || len < 1e-3) return origin
      const [x0, y0, x1, y1] = bbox
      const ratios: number[] = []
      if (dx > 0) ratios.push((x1 - origin.x) / dx)
      if (dx < 0) ratios.push((x0 - origin.x) / dx)
      if (dy > 0) ratios.push((y1 - origin.y) / dy)
      if (dy < 0) ratios.push((y0 - origin.y) / dy)
      const positive = ratios.filter((value) => Number.isFinite(value) && value > 0)
      if (!positive.length) return origin
      const t = Math.min(...positive)
      const margin = Math.min(12 / len, t * 0.25)
      const adjusted = Math.max(0, t - margin)
      return {
        x: origin.x + dx * adjusted,
        y: origin.y + dy * adjusted,
      }
    },
    [],
  )
  const graphRelationshipEdges = useMemo(() => {
    const edges: Array<{
      key: string
      points: [number, number, number, number]
      color: string
      label: string
      labelPos: { x: number; y: number }
      highlighted: boolean
    }> = []
    const centerMap = new Map<
      string,
      { center: { x: number; y: number } | null; bbox: [number, number, number, number] | null; color: string }
    >()
    graphBlockCards.forEach((card) => {
      centerMap.set(card.blockId, { center: card.center, bbox: card.bbox, color: card.color })
    })
    const seen = new Set<string>()
    for (const card of graphBlockCards) {
      if (!card.center || !card.relationships.length) continue
      const sourceInfo = centerMap.get(card.blockId)
      if (!sourceInfo || !sourceInfo.center) continue
      for (const rel of card.relationships) {
        const targetInfo = centerMap.get(rel.target)
        if (!targetInfo || !targetInfo.center) continue
        const key = `${card.blockId}->${rel.target}:${rel.type}`
        const reverseKey = `${rel.target}->${card.blockId}:${rel.type}`
        if (seen.has(key) || seen.has(reverseKey)) continue
        seen.add(key)
        const startPoint = clipPointToRect(sourceInfo.center, sourceInfo.bbox, targetInfo.center)
        const endPoint = clipPointToRect(targetInfo.center, targetInfo.bbox, sourceInfo.center)
        const dx = endPoint.x - startPoint.x
        const dy = endPoint.y - startPoint.y
        const len = Math.hypot(dx, dy)
        if (!Number.isFinite(len) || len < 24) continue
        const midX = startPoint.x + dx * 0.5
        const midY = startPoint.y + dy * 0.5
        const offset = Math.min(28, len * 0.12)
        const norm = len || 1
        const labelPos = {
          x: midX + (-dy / norm) * offset,
          y: midY + (dx / norm) * offset,
        }
        const typeLabelRaw = String(rel.type || '')
        const typeLabel = typeLabelRaw
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (s) => s.toUpperCase())
        const label =
          Number.isFinite(rel.score) && rel.score !== undefined && rel.score !== null
            ? `${typeLabel} (${Number(rel.score).toFixed(2)})`
            : typeLabel
        const highlighted = hoveredGraphBlockId
          ? card.blockId === hoveredGraphBlockId || rel.target === hoveredGraphBlockId
          : false
        edges.push({
          key,
          points: [startPoint.x, startPoint.y, endPoint.x, endPoint.y],
          color: card.color,
          label,
          labelPos,
          highlighted,
        })
      }
    }
    return edges
  }, [graphBlockCards, clipPointToRect, hoveredGraphBlockId])
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
  const handleBlockHover = useCallback((blockId: string | null) => {
    const nextBlock = blockId ?? null
    setHoveredGraphBlockId(nextBlock)
    if (!nextBlock) {
      setHoveredGraphFragmentId(null)
    }
  }, [])
  const handleFragmentHover = useCallback((fragmentId: string | null, blockId: string | null) => {
    setHoveredGraphFragmentId(fragmentId ?? null)
    if (blockId) {
      setHoveredGraphBlockId(blockId)
    }
  }, [])
  const showGraphHighlights = graphInspectorVisible && autoMaintain && (
    graphBlockCards.some((block) => (block.fragments && block.fragments.length > 0) || !!block.bbox)
    || graphRelationshipEdges.length > 0
    || ((graphSnapshot?.visionPendingGroups?.length ?? 0) > 0)
    || graphSelectedFragmentIds.length > 0
  )
  const promoteGroup = useCallback(async (groupId: string) => {
    if (!sid) {
      alert('Need to initialize session first');
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
      alert('Promote group Failed:\n' + (err?.message || String(err)))
    } finally {
      setPromoteGroupPending((prev) => (prev === groupId ? null : prev))
    }
  }, [sid, fetchGraphSnapshot])
  const promoteVisionPendingGroup = useCallback(async (groupId: string) => {
    if (!sid) {
      alert('Need to initialize session first')
      return
    }
    setPromoteVisionGroupPending(groupId)
    try {
      let graphSnapshotPayload: GraphSnapshotUpload | null = null
      const pendingGroup = (graphSnapshot?.visionPendingGroups ?? []).find((g) => String(g.groupId) === String(groupId))
      const pendingBBox = Array.isArray(pendingGroup?.bbox) && pendingGroup!.bbox.length === 4
        ? pendingGroup!.bbox as [number, number, number, number]
        : null
      if (pendingBBox) {
        try {
          graphSnapshotPayload = await captureGraphSnapshotPayload(pendingBBox, {
            maxSize: 1400,
            mime: "image/png",
            quality: 1.0,
            padding: 96,
            background: "#ffffff",
          })
        } catch (err) {
          console.warn('[graph] promote pending vision snapshot capture failed:', err)
        }
      }
      const res = await apiFetch('/graph/promote-vision-group', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sid,
          group_id: groupId,
          ...(graphSnapshotPayload ? { graph_snapshot: graphSnapshotPayload } : {}),
        }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error('promote pending vision group failed: ' + res.status + ' ' + res.statusText + (txt ? '\n' + txt : ''))
      }
      await fetchGraphSnapshot(sid)
    } catch (err: any) {
      console.warn('[graph] promote pending vision group error:', err)
      alert('Promote pending vision group failed:\n' + (err?.message || String(err)))
    } finally {
      setPromoteVisionGroupPending((prev) => (prev === groupId ? null : prev))
    }
  }, [sid, fetchGraphSnapshot, graphSnapshot?.visionPendingGroups, captureGraphSnapshotPayload])
  const toggleGraphInspector = useCallback(() => {
    setGraphInspectorVisible((prev) => !prev)
  }, [])
  const applyGraphSelectionBlockAction = useCallback(async (
    action: 'create_block' | 'assign_block',
  ) => {
    if (!sid) {
      alert('Need to initialize session first')
      return
    }
    if (!autoMaintain) {
      alert('Enable Auto Maintain first')
      return
    }
    const fragmentIds = [...graphSelectedFragmentIds]
    if (!fragmentIds.length) {
      alert('No fragments selected')
      return
    }
    let labelHint: string | undefined
    let targetBlockId: string | undefined
    if (action === 'create_block') {
      const raw = window.prompt('New block label (optional)', '')
      if (raw === null) return
      const compact = raw.trim()
      labelHint = compact || undefined
    } else {
      targetBlockId = (graphSelectionTargetBlockId || '').trim()
      if (!targetBlockId) {
        alert('Select a target block first')
        return
      }
    }
    setGraphSelectionActionPending(action)
    try {
      const res = await apiFetch('/graph/selection-block-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sid,
          action,
          fragment_ids: fragmentIds,
          ...(targetBlockId ? { target_block_id: targetBlockId } : {}),
          ...(labelHint ? { label_hint: labelHint } : {}),
          focus_after: true,
        }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`selection block action failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      await fetchGraphSnapshot(sid)
    } catch (err: any) {
      console.warn('[graph] selection block action error:', err)
      alert('Selection block action failed:\n' + (err?.message || String(err)))
    } finally {
      setGraphSelectionActionPending((prev) => (prev === action ? null : prev))
    }
  }, [sid, autoMaintain, graphSelectedFragmentIds, graphSelectionTargetBlockId, fetchGraphSnapshot])
  const graphSelectionOverlayActive = graphInspectorVisible && autoMaintain && graphBlockSelectionMode
  const scheduleGraphSelectionRectScreen = useCallback((nextRect: {
    x0: number
    y0: number
    x1: number
    y1: number
  }) => {
    graphSelectionRectPendingRef.current = nextRect
    if (graphSelectionRectRafRef.current != null) return
    graphSelectionRectRafRef.current = window.requestAnimationFrame(() => {
      graphSelectionRectRafRef.current = null
      const pending = graphSelectionRectPendingRef.current
      if (!pending) return
      setGraphSelectionRectScreen(pending)
    })
  }, [])
  const graphSelectionRectNormalized = useMemo(() => {
    if (!graphSelectionRectScreen) return null
    const left = Math.min(graphSelectionRectScreen.x0, graphSelectionRectScreen.x1)
    const top = Math.min(graphSelectionRectScreen.y0, graphSelectionRectScreen.y1)
    const width = Math.abs(graphSelectionRectScreen.x1 - graphSelectionRectScreen.x0)
    const height = Math.abs(graphSelectionRectScreen.y1 - graphSelectionRectScreen.y0)
    return { left, top, width, height }
  }, [graphSelectionRectScreen])
  const onGraphSelectionPointerDown = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    if (!graphSelectionOverlayActive) return
    const rect = rootRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    if (graphSelectionRectRafRef.current != null) {
      window.cancelAnimationFrame(graphSelectionRectRafRef.current)
      graphSelectionRectRafRef.current = null
    }
    graphSelectionRectPendingRef.current = null
    graphSelectionDragRef.current = { pointerId: ev.pointerId, x0: x, y0: y }
    setGraphSelectionDragging(true)
    setGraphSelectionRectScreen({ x0: x, y0: y, x1: x, y1: y })
    try { ev.currentTarget.setPointerCapture(ev.pointerId) } catch {}
    ev.preventDefault()
  }, [graphSelectionOverlayActive])
  const onGraphSelectionPointerMove = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    const drag = graphSelectionDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    const rect = rootRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    scheduleGraphSelectionRectScreen({ x0: drag.x0, y0: drag.y0, x1: x, y1: y })
    ev.preventDefault()
  }, [scheduleGraphSelectionRectScreen])
  const onGraphSelectionPointerUp = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    const drag = graphSelectionDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    const rect = rootRef.current?.getBoundingClientRect()
    graphSelectionDragRef.current = null
    graphSelectionRectPendingRef.current = null
    setGraphSelectionDragging(false)
    if (graphSelectionRectRafRef.current != null) {
      window.cancelAnimationFrame(graphSelectionRectRafRef.current)
      graphSelectionRectRafRef.current = null
    }
    if (!rect) {
      setGraphSelectionRectScreen(null)
      return
    }
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    const finalRect = { x0: drag.x0, y0: drag.y0, x1: x, y1: y }
    setGraphSelectionRectScreen(finalRect)
    selectGraphFragmentsByScreenRect(finalRect)
    try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch {}
    ev.preventDefault()
  }, [selectGraphFragmentsByScreenRect])
  const onGraphSelectionPointerCancel = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    const drag = graphSelectionDragRef.current
    if (!drag || drag.pointerId !== ev.pointerId) return
    graphSelectionDragRef.current = null
    graphSelectionRectPendingRef.current = null
    setGraphSelectionDragging(false)
    if (graphSelectionRectRafRef.current != null) {
      window.cancelAnimationFrame(graphSelectionRectRafRef.current)
      graphSelectionRectRafRef.current = null
    }
    try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch {}
  }, [])
  const pendingVisionGroupOverlays = useMemo(() => {
    const groups = graphSnapshot?.visionPendingGroups ?? []
    if (!graphInspectorVisible || !autoMaintain || groups.length === 0) return []
    return groups
      .map((group) => {
        const bbox = Array.isArray(group.bbox) && group.bbox.length === 4
          ? group.bbox as [number, number, number, number]
          : null
        if (!bbox) return null
        const [x0, y0, x1, y1] = bbox
        const cx = (x0 + x1) / 2
        const cy = (y0 + y1) / 2
        const screen = worldToScreen(cx, cy)
        const onScreen = screen.x >= -80 && screen.x <= size.width + 80 && screen.y >= -40 && screen.y <= size.height + 40
        if (!onScreen) return null
        return {
          groupId: group.groupId,
          bbox,
          count: Number(group.count || 0),
          eligible: Boolean(group.eligible),
          readyReason: group.readyReason ?? null,
          centerWorld: { x: cx, y: cy },
          centerScreen: screen,
        }
      })
      .filter((item): item is NonNullable<typeof item> => Boolean(item))
  }, [graphSnapshot?.visionPendingGroups, graphInspectorVisible, autoMaintain, worldToScreen, size.width, size.height])
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
  const importQueueRef = useRef<Array<{ draft: ShapeDraft; stroke: AIStrokeV11 }>>([])
  const importTimerRef = useRef<number | null>(null)
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
    const prevShapeCount = lastCommittedShapeCountRef.current
    lastCommittedShapeCountRef.current = shapes.length
    if (prevShapeCount === shapes.length) return
    if (Object.keys(previews).length > 0) {
      if (currentPayloadId && previews[currentPayloadId]) {
        const entry = previews[currentPayloadId]
        recordExperimentDismiss(entry.payloadId, entry.requestId ?? null, entry.phaseId ?? experimentPhaseId, 'canvas_mutation')
      }
      setPreviews({})
      setCurrentPayloadId(null)
    }
  }, [shapes.length, previews, currentPayloadId, recordExperimentDismiss, experimentPhaseId])
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
    if (autoComplete && !autoMaintainPending && !askInFlightRef.current && (!hasActivePreview || forceStart)) {
      clearAutoTimer()
      setAutoCountdown(5)
      // Update visible countdown every second
      autoTickerRef.current = setInterval(() => {
        setAutoCountdown((sec) => (sec == null ? null : Math.max(0, sec - 1)))
      }, 1000)
      // Trigger askAI after 5 seconds
      autoTimerRef.current = setTimeout(() => {
        if (askInFlightRef.current || autoMaintainPending) return
        clearAutoTimer()
        // Equivalent to pressing the Ask AI button
        askAIRef.current && askAIRef.current()
      }, 5000)
    } else {
      // Otherwise ensure all timers are cleared
      clearAutoTimer()
    }
  }, [autoComplete, hasActivePreview, clearAutoTimer, autoMaintainPending])

  const deleteShapeById = useCallback((targetId: string, opts?: { skipHistory?: boolean }) => {
    if (!targetId) return
    if (!opts?.skipHistory) pushHistory()
    setShapes(prev => prev.filter(s => s.id !== targetId))
    setDrawStack(prev => prev.filter(entry => entry.draft.id !== targetId))
    setSelectedShapeId(prev => (prev === targetId ? null : prev))
    noteUserAction({ forceStart: true })
  }, [pushHistory, setShapes, setDrawStack, noteUserAction])
  const deleteSelectedShape = useCallback(() => {
    if (!selectedShapeId) return
    deleteShapeById(selectedShapeId)
  }, [selectedShapeId, deleteShapeById])

  const clearImportQueue = useCallback(() => {
    if (importTimerRef.current) {
      window.clearTimeout(importTimerRef.current)
      importTimerRef.current = null
    }
    importQueueRef.current = []
  }, [])

  const processImportQueue = useCallback(() => {
    const next = importQueueRef.current.shift()
    if (!next) {
      importTimerRef.current = null
      return
    }
    setShapes((prev) => [...prev, next.draft])
    setDrawStack((prev) => [...prev, { ai: next.stroke, draft: next.draft }])
    noteUserAction({ forceStart: true })
    if (importQueueRef.current.length > 0) {
      importTimerRef.current = window.setTimeout(processImportQueue, 4000) as unknown as number
    } else {
      importTimerRef.current = null
    }
  }, [noteUserAction])

  const startDelayedImport = useCallback((entries: Array<{ draft: ShapeDraft; stroke: AIStrokeV11 }>) => {
    clearImportQueue()
    graphKnownStrokeIdsRef.current = new Set()
    graphCaptureBBoxRef.current = null
    setShapes([])
    setDrawStack([])
    importQueueRef.current = [...entries]
    processImportQueue()
  }, [clearImportQueue, processImportQueue])

  React.useEffect(() => {
    return () => {
      clearImportQueue()
    }
  }, [clearImportQueue])

  const importJSON = useCallback(async (file: File) => {
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const strokes = extractStrokesFromImport(data)
      if (strokes.length === 0) {
        alert('No strokes found in JSON file')
        return
      }
      const entries = buildImportEntriesFromStrokes(strokes)
      if (!entries.length) {
        alert('No renderable shapes found in JSON file')
        return
      }
      if (autoMaintain) {
        startDelayedImport(entries)
        alert(`Queued ${entries.length} items for delayed import (4s interval).`)
      } else {
        clearImportQueue()
        setShapes(entries.map((entry) => entry.draft))
        setDrawStack(entries.map((entry) => ({ ai: entry.stroke, draft: entry.draft })))
        noteUserAction({ forceStart: true })
        alert(`Imported ${entries.length} items.`)
      }
    } catch (err: any) {
      console.warn('[import json] failed:', err)
      alert('Invalid JSON:\n' + (err?.message || String(err)))
    }
  }, [autoMaintain, startDelayedImport, clearImportQueue, noteUserAction])
  const refreshProjectList = useCallback(async () => {
    const res = await apiFetch('/projects')
    if (!res.ok) {
      const txt = await res.text().catch(() => '')
      throw new Error(`projects failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
    }
    const data = await res.json().catch(() => ({}))
    const next = Array.isArray(data?.projects) ? data.projects as ProjectListItem[] : []
    setProjectList(next)
    setCurrentProjectName((prev) => {
      if (!currentProjectId) return prev
      const hit = next.find((p) => p.projectId === currentProjectId)
      return hit ? (hit.name || hit.projectId) : prev
    })
    setSelectedProjectId((prev) => {
      if (prev && next.some((p) => p.projectId === prev)) return prev
      const currentProject = next.find((p) => p.projectId === (currentProjectId || projectDetail?.projectId || ''))
      if (currentProject) return currentProject.projectId
      return next[0]?.projectId || ''
    })
    return next
  }, [projectDetail?.projectId, currentProjectId])
  const refreshProjectDetail = useCallback(async (projectId: string) => {
    const pid = String(projectId || '').trim()
    if (!pid) {
      setProjectDetail(null)
      return null
    }
    const res = await apiFetch(`/project/detail?project_id=${encodeURIComponent(pid)}`)
    if (!res.ok) {
      const txt = await res.text().catch(() => '')
      throw new Error(`project detail failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
    }
    const data = await res.json().catch(() => ({}))
    const snapshots = Array.isArray(data?.snapshots) ? data.snapshots as ProjectSnapshotItem[] : []
    const legacySnapshots = Array.isArray(data?.legacySnapshots) ? data.legacySnapshots as ProjectSnapshotItem[] : snapshots
    const commits = Array.isArray(data?.commits) ? data.commits as ProjectCommitItem[] : []
    const current = (data?.current && typeof data.current === 'object') ? (data.current as ProjectCurrentPreviewItem) : null
    const detail: ProjectDetail = {
      projectId: String(data?.projectId || pid),
      meta: (data?.meta && typeof data.meta === 'object') ? data.meta : {},
      current,
      commits,
      legacySnapshots,
      snapshots,
    }
    setProjectDetail(detail)
    return detail
  }, [])
  const openProjectManager = useCallback(async () => {
    setProjectManagerOpen(true)
    setProjectManagerBusy(true)
    setProjectManagerError('')
    try {
      const list = await refreshProjectList()
      const bound = (projectList.find((p) => p.projectId === (currentProjectId || projectDetail?.projectId || ''))?.projectId)
      const nextId = selectedProjectId || bound || list?.[0]?.projectId || ''
      if (nextId) {
        setSelectedProjectId(nextId)
        await refreshProjectDetail(nextId)
      } else {
        setProjectDetail(null)
      }
    } catch (err: any) {
      console.warn('[project] open manager failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectManagerBusy(false)
    }
  }, [refreshProjectList, refreshProjectDetail, selectedProjectId, projectList, projectDetail?.projectId, currentProjectId])
  React.useEffect(() => {
    if (!projectManagerOpen) return
    if (!selectedProjectId) return
    let alive = true
    ;(async () => {
      try {
        const detail = await refreshProjectDetail(selectedProjectId)
        if (!alive) return
        if (detail) setProjectManagerError('')
      } catch (err: any) {
        if (!alive) return
        console.warn('[project] detail refresh failed:', err)
        setProjectManagerError(err?.message || String(err))
      }
    })()
    return () => { alive = false }
  }, [projectManagerOpen, selectedProjectId, refreshProjectDetail])
  const initSessionForProjectAction = useCallback(async () => {
    let curSid = sid
    if (curSid) return curSid
    const initRes = await apiFetch('/session/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'light_helper', init_goal: hint }),
    })
    if (!initRes.ok) {
      const txt = await initRes.text().catch(() => '')
      throw new Error(`init session failed: ${initRes.status} ${initRes.statusText}${txt ? `\n${txt}` : ''}`)
    }
    const j = await initRes.json().catch(() => ({}))
    curSid = String(j?.sid || '')
    if (!curSid) throw new Error('invalid session id from /session/init')
    setSid(curSid)
    lastSentIndexRef.current = 0
    return curSid
  }, [sid, hint])
  const createProjectAndBind = useCallback(async (name?: string) => {
    const curSid = await initSessionForProjectAction()
    const res = await apiFetch('/project/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sid: curSid,
        name: (name || '').trim() || undefined,
      }),
    })
    if (!res.ok) {
      const txt = await res.text().catch(() => '')
      throw new Error(`project create failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
    }
    const data = await res.json().catch(() => ({}))
    const pid = String(data?.projectId || '')
    const pname = String(data?.meta?.name || name || pid || '')
    if (pid) {
      setCurrentProjectId(pid)
      setCurrentProjectName(pname || pid)
      setSelectedProjectId(pid)
    }
    return { sid: curSid, projectId: pid, projectName: pname || pid }
  }, [initSessionForProjectAction])
  const markProjectSavedFlash = useCallback(() => {
    if (projectSaveFlashTimerRef.current) window.clearTimeout(projectSaveFlashTimerRef.current)
    setProjectSaveFlash(true)
    projectSaveFlashTimerRef.current = window.setTimeout(() => {
      setProjectSaveFlash(false)
      projectSaveFlashTimerRef.current = null
    }, 1500) as unknown as number
  }, [])
  const saveProjectCurrentNow = useCallback(async (opts?: { silent?: boolean; forceProjectId?: string; note?: string }) => {
    if (projectSavePending) return null
    setProjectSavePending(true)
    if (!opts?.silent) setProjectManagerError('')
    try {
      const curSid = await initSessionForProjectAction()
      let pid = String(opts?.forceProjectId || currentProjectId || selectedProjectId || '').trim()
      if (!pid) {
        throw new Error('No project bound. Create a project first.')
      }
      const snapshot = await captureProjectSaveSnapshotPayload()
      const res = await apiFetch('/project/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sid: curSid,
          projectId: pid,
          mode: 'current',
          note: opts?.note || undefined,
          snapshot: snapshot ?? undefined,
        }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`project save failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = await res.json().catch(() => ({}))
      pid = String(data?.projectId || pid)
      setCurrentProjectId(pid)
      const list = await refreshProjectList()
      const hit = list.find((p) => p.projectId === pid)
      setCurrentProjectName(hit ? (hit.name || hit.projectId) : (currentProjectName || pid))
      setSelectedProjectId(pid)
      if (projectManagerOpen) await refreshProjectDetail(pid)
      setProjectCurrentPreviewDirty(false)
      markProjectSavedFlash()
      return { projectId: pid }
    } catch (err: any) {
      console.warn('[project] save current failed:', err)
      setProjectManagerError(err?.message || String(err))
      throw err
    } finally {
      setProjectSavePending(false)
    }
  }, [
    projectSavePending,
    initSessionForProjectAction,
    currentProjectId,
    selectedProjectId,
    captureProjectSaveSnapshotPayload,
    refreshProjectList,
    refreshProjectDetail,
    projectManagerOpen,
    currentProjectName,
    markProjectSavedFlash,
  ])
  const handleCreateProject = useCallback(async () => {
    if (projectActionPending) return
    setProjectActionPending('create')
    setProjectManagerError('')
    try {
      const created = await createProjectAndBind(projectNameDraft)
      setProjectNameDraft('')
      const list = await refreshProjectList()
      const hit = list.find((p) => p.projectId === created.projectId)
      if (hit) setCurrentProjectName(hit.name || hit.projectId)
      if (created.projectId) await refreshProjectDetail(created.projectId)
      setProjectCurrentPreviewDirty(true)
    } catch (err: any) {
      console.warn('[project] create failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [projectActionPending, createProjectAndBind, projectNameDraft, refreshProjectList, refreshProjectDetail])
  const handleTopbarSaveProject = useCallback(async () => {
    if (projectSavePending || projectPromptSubmitting) return
    if (!currentProjectId) {
      setProjectPromptNameDraft(projectNameDraft.trim() || '')
      setProjectPromptOpen(true)
      return
    }
    try {
      await saveProjectCurrentNow({ forceProjectId: currentProjectId, note: 'manual current save' })
    } catch {
      // error already surfaced
    }
  }, [projectSavePending, projectPromptSubmitting, currentProjectId, projectNameDraft, saveProjectCurrentNow])
  const handleConfirmProjectPromptCreateAndSave = useCallback(async () => {
    if (projectPromptSubmitting) return
    const name = projectPromptNameDraft.trim()
    if (!name) return
    setProjectPromptSubmitting(true)
    setProjectManagerError('')
    try {
      const created = await createProjectAndBind(name)
      setProjectPromptOpen(false)
      setProjectPromptNameDraft('')
      setProjectNameDraft('')
      await refreshProjectList()
      if (created.projectId) {
        await saveProjectCurrentNow({ forceProjectId: created.projectId, note: 'initial current save' })
        if (projectManagerOpen) await refreshProjectDetail(created.projectId)
      }
    } catch (err: any) {
      console.warn('[project] create+save failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectPromptSubmitting(false)
    }
  }, [projectPromptSubmitting, projectPromptNameDraft, createProjectAndBind, refreshProjectList, saveProjectCurrentNow, projectManagerOpen, refreshProjectDetail])
  const handleCommitProject = useCallback(async () => {
    if (projectActionPending) return
    let pid = String(selectedProjectId || currentProjectId || '').trim()
    if (!pid) {
      setProjectManagerError('Select or create a project first.')
      return
    }
    setProjectActionPending('commit')
    setProjectManagerError('')
    try {
      const curSid = await initSessionForProjectAction()
      const snapshot = await captureProjectSaveSnapshotPayload()
      if (!snapshot) throw new Error('Failed to capture project snapshot for commit')
      const res = await apiFetch('/project/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sid: curSid,
          projectId: pid,
          message: projectCommitMessageDraft.trim() || undefined,
          snapshot,
        }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`project commit failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = await res.json().catch(() => ({}))
      pid = String(data?.projectId || pid)
      setCurrentProjectId(pid)
      setSelectedProjectId(pid)
      await refreshProjectList()
      await refreshProjectDetail(pid)
      setProjectCurrentPreviewDirty(false)
      setProjectCommitMessageDraft('')
    } catch (err: any) {
      console.warn('[project] commit failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [
    projectActionPending,
    selectedProjectId,
    currentProjectId,
    initSessionForProjectAction,
    captureProjectSaveSnapshotPayload,
    projectCommitMessageDraft,
    refreshProjectList,
    refreshProjectDetail,
  ])
  const applyOpenedProjectPayload = useCallback(async (data: any, opts?: { closeManager?: boolean }) => {
    const openedSid = String(data?.sid || '')
    const graphEnabled = Boolean(data?.graphEnabled)
    const openedProjectId = String(data?.projectId || '')
    const strokes = Array.isArray(data?.strokes) ? (data.strokes as AIStrokeV11[]) : []
    const entries = buildImportEntriesFromStrokes(strokes)
    clearImportQueue()
    projectSkipNextDirtyRef.current = true
    setPast([])
    setFuture([])
    setPreviews({})
    setCurrentPayloadId(null)
    setSelectedShapeId(null)
    setCompletionPreviews({})
    setShapes(entries.map((entry) => entry.draft))
    setDrawStack(entries.map((entry) => ({ ai: entry.stroke, draft: entry.draft })))
    graphKnownStrokeIdsRef.current = new Set(entries.map((entry) => String(entry.stroke.id)))
    graphCaptureBBoxRef.current = null
    if (openedSid) {
      setSid(openedSid)
      lastSentIndexRef.current = 0
    }
    if (graphEnabled) {
      setMode('full')
    }
    setAutoMaintain(graphEnabled)
    if (!graphEnabled) {
      setGraphSnapshot(null)
    } else if (openedSid) {
      await fetchGraphSnapshot(openedSid)
    }
    if (openedProjectId) {
      setCurrentProjectId(openedProjectId)
      setSelectedProjectId(openedProjectId)
    }
    setProjectCurrentPreviewDirty(false)
    if (opts?.closeManager !== false) setProjectManagerOpen(false)
    setProjectManagerError('')
  }, [clearImportQueue, fetchGraphSnapshot])
  const handleOpenProject = useCallback(async () => {
    if (projectActionPending) return
    const pid = String(selectedProjectId || '').trim()
    if (!pid) {
      setProjectManagerError('Select a project to open.')
      return
    }
    setProjectActionPending('open')
    setProjectManagerError('')
    try {
      const res = await apiFetch('/project/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: pid,
          sid: sid || undefined,
        }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`project open failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = await res.json().catch(() => ({}))
      await applyOpenedProjectPayload(data, { closeManager: true })
      await refreshProjectList()
      const detail = await refreshProjectDetail(pid)
      const list = await refreshProjectList()
      const hit = list.find((p) => p.projectId === pid)
      if (hit) setCurrentProjectName(hit.name || hit.projectId)
      else if (detail?.meta?.name) setCurrentProjectName(String(detail.meta.name))
    } catch (err: any) {
      console.warn('[project] open failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [projectActionPending, selectedProjectId, sid, applyOpenedProjectPayload, refreshProjectList, refreshProjectDetail])
  const handleCheckoutCommit = useCallback(async (projectId: string, commitId: string) => {
    if (projectActionPending) return
    const pid = String(projectId || '').trim()
    const cid = String(commitId || '').trim()
    if (!pid || !cid) return
    const ok = window.confirm('Switch current to this commit?\nThis will replace the current canvas and graph state with the selected commit snapshot.')
    if (!ok) return
    setProjectActionPending('checkout')
    setProjectManagerError('')
    try {
      const curSid = await initSessionForProjectAction()
      const res = await apiFetch('/project/commit/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sid: curSid, projectId: pid, commitId: cid }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`commit checkout failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      const data = await res.json().catch(() => ({}))
      await applyOpenedProjectPayload(data, { closeManager: false })
      const list = await refreshProjectList()
      await refreshProjectDetail(pid)
      const hit = list.find((p) => p.projectId === pid)
      if (hit) setCurrentProjectName(hit.name || hit.projectId)
    } catch (err: any) {
      console.warn('[project] commit checkout failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [projectActionPending, initSessionForProjectAction, applyOpenedProjectPayload, refreshProjectList, refreshProjectDetail])
  const handleDeleteProject = useCallback(async (projectId: string) => {
    if (projectActionPending) return
    const pid = String(projectId || '').trim()
    if (!pid) return
    const projName = projectList.find((p) => p.projectId === pid)?.name || pid
    if (!window.confirm(`Delete project "${projName}" and all commits?`)) return
    setProjectActionPending('delete-project')
    setProjectManagerError('')
    try {
      const res = await apiFetch('/project/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: pid, sid: sid || undefined }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`project delete failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      if (currentProjectId === pid) {
        setCurrentProjectId(null)
        setCurrentProjectName(null)
      }
      if (selectedProjectId === pid) {
        setProjectDetail(null)
      }
      setProjectContextMenuState(null)
      const list = await refreshProjectList()
      if (selectedProjectId === pid) {
        const nextId = list[0]?.projectId || ''
        setSelectedProjectId(nextId)
        if (nextId) await refreshProjectDetail(nextId)
      }
    } catch (err: any) {
      console.warn('[project] delete failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [projectActionPending, projectList, sid, currentProjectId, selectedProjectId, refreshProjectList, refreshProjectDetail])
  const handleDeleteCommit = useCallback(async (projectId: string, commitId: string) => {
    if (projectActionPending) return
    const pid = String(projectId || '').trim()
    const cid = String(commitId || '').trim()
    if (!pid || !cid) return
    if (!window.confirm('Delete this commit snapshot?')) return
    setProjectActionPending('delete-commit')
    setProjectManagerError('')
    try {
      const res = await apiFetch('/project/commit/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: pid, commitId: cid }),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`commit delete failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
      }
      setProjectContextMenuState(null)
      await refreshProjectList()
      await refreshProjectDetail(pid)
    } catch (err: any) {
      console.warn('[project] commit delete failed:', err)
      setProjectManagerError(err?.message || String(err))
    } finally {
      setProjectActionPending(null)
    }
  }, [projectActionPending, refreshProjectList, refreshProjectDetail])
  React.useEffect(() => {
    if (!currentProjectId) return
    if (projectSkipNextDirtyRef.current) {
      projectSkipNextDirtyRef.current = false
      return
    }
    setProjectCurrentPreviewDirty(true)
  }, [drawStack, currentProjectId])
  React.useEffect(() => {
    const onDocPointer = () => setProjectContextMenuState(null)
    window.addEventListener('pointerdown', onDocPointer)
    return () => window.removeEventListener('pointerdown', onDocPointer)
  }, [])
  React.useEffect(() => {
    return () => {
      if (projectSaveFlashTimerRef.current) {
        window.clearTimeout(projectSaveFlashTimerRef.current)
        projectSaveFlashTimerRef.current = null
      }
      if (projectAutoSnapshotTimerRef.current) {
        window.clearInterval(projectAutoSnapshotTimerRef.current)
        projectAutoSnapshotTimerRef.current = null
      }
    }
  }, [])
  React.useEffect(() => {
    if (projectAutoSnapshotTimerRef.current) {
      window.clearInterval(projectAutoSnapshotTimerRef.current)
      projectAutoSnapshotTimerRef.current = null
    }
    if (!currentProjectId || !sid) return
    if (projectSavePending || projectPromptSubmitting || projectActionPending) return
    const tick = async () => {
      if (!projectCurrentPreviewDirty) return
      if (document.visibilityState !== 'visible') return
      try {
        const snapshot = await captureProjectSaveSnapshotPayload()
        if (!snapshot) return
        const res = await apiFetch('/project/current-snapshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sid,
            projectId: currentProjectId,
            snapshot,
          }),
        })
        if (!res.ok) {
          const txt = await res.text().catch(() => '')
          throw new Error(`project current snapshot failed: ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
        }
        setProjectCurrentPreviewDirty(false)
        if (projectManagerOpen && selectedProjectId === currentProjectId) {
          void refreshProjectDetail(currentProjectId)
          void refreshProjectList()
        }
      } catch (err) {
        console.warn('[project] auto current snapshot failed:', err)
      }
    }
    projectAutoSnapshotTimerRef.current = window.setInterval(() => { void tick() }, 300000) as unknown as number
    return () => {
      if (projectAutoSnapshotTimerRef.current) {
        window.clearInterval(projectAutoSnapshotTimerRef.current)
        projectAutoSnapshotTimerRef.current = null
      }
    }
  }, [
    currentProjectId,
    sid,
    projectCurrentPreviewDirty,
    projectSavePending,
    projectPromptSubmitting,
    projectActionPending,
    captureProjectSaveSnapshotPayload,
    projectManagerOpen,
    selectedProjectId,
    refreshProjectDetail,
    refreshProjectList,
  ])
  const formatProjectTime = useCallback((raw?: string | null) => {
    if (!raw) return ''
    const t = Date.parse(String(raw))
    if (!Number.isFinite(t)) return String(raw)
    try {
      return new Date(t).toLocaleString()
    } catch {
      return String(raw)
    }
  }, [])
  const updateTextEditorState = useCallback((patch: Partial<TextEditorState>) => {
    setTextEditor((prev) => (prev ? { ...prev, ...patch } : prev))
  }, [])
  const applyTextStylePreset = useCallback((preset: TextStylePreset) => {
    updateTextEditorState({
      fontSize: preset.fontSize,
      fontWeight: preset.fontWeight,
      color: preset.color,
    })
  }, [updateTextEditorState])
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
      alert('Completion failed, please try again.')
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
  const getDefaultTextBoxSize = useCallback((fontSize: number) => {
    const safeScale = Math.max(view.scale || 1, 0.2)
    const minScreenWidth = Math.max(220, fontSize * 10)
    const minScreenHeight = Math.max(140, fontSize * 4)
    const targetScreenWidth = clamp(size.width * 0.18, minScreenWidth, 520)
    const targetScreenHeight = clamp(size.height * 0.16, minScreenHeight, 320)
    return {
      width: Math.max(80, targetScreenWidth / safeScale),
      height: Math.max(Math.round(fontSize * 1.6), targetScreenHeight / safeScale),
    }
  }, [size.height, size.width, view.scale])
  const commitTextEditor = useCallback(() => {
    if (!textEditor) return
    const content = textEditor.text.replace(/\s+$/g, '')
    if (!content.trim()) {
      alert('Connot commit empty text.')
      return
    }
    const summary = textEditor.summary.trim().slice(0, 30)
    const { width: fallbackWidth, height: fallbackHeight } = getDefaultTextBoxSize(textEditor.fontSize)
    const baseWidth = textEditor.w > 0 ? Math.max(textEditor.w, 80) : fallbackWidth
    const baseHeight = textEditor.h > 0 ? Math.max(textEditor.h, Math.round(textEditor.fontSize * 1.6)) : fallbackHeight
    const renderedText = renderMarkdownToCanvasText(content)
    const layout = computeTextBoxLayout({
      text: renderedText,
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
    const textRole = inferTextRole(textEditor.fontSize, textEditor.fontWeight)
    const sharedMeta = {
      author: 'human',
      text: content,
      summary,
      role: textRole,
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
  }, [textEditor, pushHistory, setShapes, setDrawStack, updateTextSettings, noteUserAction, computeTextBoxLayout, setSelectedShapeId, clearCompletionPreview, getDefaultTextBoxSize])
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
    const { width: fallbackWidth, height: fallbackHeight } = getDefaultTextBoxSize(fontSize)
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
  }, [textSettings, completionPreviews, getDefaultTextBoxSize])
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
    if (askInFlightRef.current) return
    askInFlightRef.current = true
    clearAutoTimer()
    try {
      const requestPhaseId = normalizePhaseId(experimentPhaseId)
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
        const snap = await snapshotCanvas(1024, "image/jpeg", 0.7, { hideGrid: true })
        if (snap.data) {
          image_data = snap.data
          image_mime = snap.mime as any
          snapshot_size = [snap.w, snap.h]
        }
      }
      // 3) Build request (include viewport to help backend validation)
      const snapshot = packAllStrokes()
      const runtimeModel = llmModel.trim()
      const runtimeTemperature = clamp(llmTemperature, 0, 2)
      const runtimeTopP = clamp(llmTopP, 0, 1)
      const runtimeMaxTokens = Math.max(256, Math.min(32768, Math.round(llmMaxTokens || VITE_LLM_MAX_TOKENS_DEFAULT)))
      const baseReq = {
        sid: curSid!,
        canvas: { viewport: [0, 0, size.width, size.height] as [number, number, number, number] },
        delta: { strokes: deltaStrokes },
        context: { version: 1, intent: 'complete', strokes: snapshot },
        hint,
        auto_complete_enabled: autoComplete,
        prefer_explanatory_drawing: preferExplanatoryDrawing,
        group_promote_mode: groupPromoteMode,
        vision_image_mode: visionImageMode,
        ...(runtimeModel ? { model: runtimeModel } : {}),
        temperature: runtimeTemperature,
        top_p: runtimeTopP,
        max_tokens: runtimeMaxTokens,
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
        const requestId1 = recordExperimentRequestSent(mode)
        let res1 = await doPost(req1)
        if (!res1.ok) {
          const t = await res1.text().catch(()=> '')
          recordExperimentRequestFailed(requestId1, `Vision 2.0 step1 failed: ${res1.status} ${res1.statusText}${t ? `\n${t}` : ''}`)
          throw new Error(`Vision 2.0 step1 failed: ${res1.status} ${res1.statusText}\n${t}`)
        }
        const data1 = await res1.json()
        recordExperimentRequestCompleted(requestId1, data1?.usage)
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
          auto_complete_enabled: autoComplete,
          prefer_explanatory_drawing: preferExplanatoryDrawing,
          group_promote_mode: groupPromoteMode,
          vision_image_mode: visionImageMode,
          ...(runtimeModel ? { model: runtimeModel } : {}),
          temperature: runtimeTemperature,
          top_p: runtimeTopP,
          max_tokens: runtimeMaxTokens,
          gen_scale: aiScale,
        }
        const requestId2 = recordExperimentRequestSent(mode)
        let res2 = await doPost(req2)
        if (!res2.ok) {
          const t = await res2.text().catch(()=> '')
          recordExperimentRequestFailed(requestId2, `Vision 2.0 step2 failed: ${res2.status} ${res2.statusText}${t ? `\n${t}` : ''}`)
          throw new Error(`Vision 2.0 step2 failed: ${res2.status} ${res2.statusText}\n${t}`)
        }
        const data2 = await res2.json()
        recordExperimentRequestCompleted(requestId2, data2?.usage)
        if (data2?.usage?.new_sid) setSid(String(data2.usage.new_sid))
        setPlannerNextStepHint(String(data2?.usage?.planner_next_step || '').trim())
        const payload2 = data2?.payload
        if (!payload2) throw new Error('No payload in step2 response')
        localStorage.setItem('ai_suggestions_v1', JSON.stringify(payload2))
        const items2 = (payload2.strokes || []).map((s:any) => ({ id: s.id, desc: (s.meta as any)?.desc }))
        setAiFeed(prev => ([{ payloadId: 'srv_'+Date.now().toString(36), time: Date.now(), items: items2 }, ...prev].slice(0, 50)))
        const v = validateAIStrokePayload(payload2)
        if (!v.ok || !v.payload) throw new Error('Invalid AI payload: ' + v.errors.join('; '))
        const norm = normalizeAIStrokePayload(v.payload)
        const drafts = planDrafts(norm)
        const usage2 = extractUsageUpdate(data2?.usage)
        setPreviews(prev => ({
          ...prev,
          [norm.payloadId]: {
            payloadId: norm.payloadId,
            drafts,
            requestId: requestId2,
            phaseId: requestPhaseId,
            promptTokens: usage2.promptTokens,
            activeBlockIds: usage2.activeBlockIds,
            planTargetBlockIds: usage2.planTargetBlockIds,
          },
        }))
        setCurrentPayloadId(norm.payloadId)
        recordExperimentPreview(norm.payloadId, requestId2, requestPhaseId)
        return
      }
      // === Legacy flow (Vision 1.0 / full / light) ===
      let requestId = recordExperimentRequestSent(mode)
      let res = await doPost({ ...baseReq, sid: curSid! })
      if (!res.ok) {
        // If session expired, re-initialize once then retry
        const txt = await res.text().catch(()=>'')
        if (res.status === 404 && /session not found/i.test(txt)) {
          recordExperimentRequestFailed(requestId, `HTTP ${res.status} ${res.statusText}${txt ? `\n${txt}` : ''}`)
          const r1 = await apiFetch('/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'light_helper', init_goal: hint }),
          })
          if (r1.ok) {
            const j1 = await r1.json()
            curSid = j1.sid
            setSid(curSid)
            requestId = recordExperimentRequestSent(mode)
            res = await doPost({ ...baseReq, sid: curSid! })
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
        recordExperimentRequestFailed(requestId, msg)
        throw new Error(msg)
      }
      // 4) Parse response payload and stage previews
      const data = await res.json()
      recordExperimentRequestCompleted(requestId, data?.usage)
      if (data?.usage?.new_sid) setSid(String(data.usage.new_sid))
      setPlannerNextStepHint(String(data?.usage?.planner_next_step || '').trim())
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
      const usage = extractUsageUpdate(data?.usage)
      setPreviews(prev => ({
        ...prev,
        [norm.payloadId]: {
          payloadId: norm.payloadId,
          drafts,
          requestId,
          phaseId: requestPhaseId,
          promptTokens: usage.promptTokens,
          activeBlockIds: usage.activeBlockIds,
          planTargetBlockIds: usage.planTargetBlockIds,
        },
      }))
      setCurrentPayloadId(norm.payloadId)
      recordExperimentPreview(norm.payloadId, requestId, requestPhaseId)
    } catch (err: any) {
      const errorMessage = err?.message || String(err)
      console.error('[askAI] error:', err)
      alert('Ask AI failed:\n' + errorMessage)
    } finally {
      askInFlightRef.current = false
    }
  }, [
    sid,
    drawStack,
    size.width,
    size.height,
    hint,
    autoComplete,
    preferExplanatoryDrawing,
    groupPromoteMode,
    visionImageMode,
    llmModel,
    llmTemperature,
    llmTopP,
    llmMaxTokens,
    aiScale,
    mode,
    visionVersion,
    clearAutoTimer,
    experimentPhaseId,
    recordExperimentPreview,
    recordExperimentRequestCompleted,
    recordExperimentRequestFailed,
    recordExperimentRequestSent,
  ])
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
    if (graphSelectionDragging) {
      if (graphPollRef.current) {
        window.clearInterval(graphPollRef.current)
        graphPollRef.current = null
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
  }, [autoMaintain, sid, fetchGraphSnapshot, graphSelectionDragging])
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
        const fontWeightToken = String(d.meta?.fontWeight ?? '400')
        const fontStyle = (fontWeightToken === 'bold' || fontWeightToken === '700')
          ? 'bold'
          : 'normal'
        const renderTextRole = String(d.meta?.role ?? inferTextRole(fontSize, fontWeightToken)).toLowerCase()
        const autoCenterText = renderTextRole === 'title' || renderTextRole === 'subtitle'
        const boxW = d.w ?? 160
        const boxH = d.h ?? 80
        const lineHeight = typeof d.meta?.lineHeight === 'number' && Number.isFinite(d.meta.lineHeight)
          ? (d.meta.lineHeight as number)
          : TEXT_LINE_HEIGHT
        const hasCompletion = !!completionText
        const isHighlighted = !!editHighlight
        let borderColor = preview ? 'rgba(15,23,42,0.22)' : 'rgba(15,23,42,0.34)'
        let borderDash: number[] | undefined = preview ? [4,4] : undefined
        let fillColorOverlay: string | undefined = preview ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.18)'
        if (isHighlighted) {
          borderColor = '#ffb74d'
          borderDash = [2, 2]
          fillColorOverlay = 'rgba(255,183,77,0.14)'
        } else if (hasCompletion) {
          borderColor = '#2563eb'
          borderDash = [6, 4]
          fillColorOverlay = 'rgba(37,99,235,0.1)'
        } else if (selected) {
          borderColor = '#4aa3ff'
          borderDash = [8,4]
          fillColorOverlay = 'rgba(74,163,255,0.08)'
        }
        const frameCorner = Math.max(8 / view.scale, 2 / view.scale)
        const frameShadowBlur = (isHighlighted || hasCompletion || selected) ? Math.max(16 / view.scale, 4 / view.scale) : 0
        const rawTextContent = String((d.meta?.text as string) ?? d.text ?? '')
        const isMarkdownText = looksLikeMarkdownText(rawTextContent)
        const storedRenderedText = typeof d.meta?.renderedText === 'string'
          ? (d.meta.renderedText as string)
          : ''
        const displayText = isMarkdownText
          ? (storedRenderedText && storedRenderedText !== rawTextContent
              ? storedRenderedText
              : renderMarkdownToCanvasText(rawTextContent))
          : (storedRenderedText || rawTextContent)
        const textOpacity = preview ? Math.min(0.35, d.style?.opacity ?? 1) : (d.style?.opacity ?? 1)
        let markdownStyledContent: React.ReactNode = null
        if (isMarkdownText && !autoCenterText) {
          const blocks = parseMarkdownDisplayBlocks(rawTextContent)
          if (blocks.length > 0) {
            const nodes: React.ReactNode[] = []
            let cursorY = d.y
            const maxY = d.y + boxH
            const baseLinePx = Math.max(fontSize * lineHeight, fontSize)
            const baseWeight = String((d.meta?.fontWeight as string) ?? (fontStyle === 'bold' ? '700' : '400'))
            for (let i = 0; i < blocks.length; i++) {
              const block = blocks[i]
              if (block.kind === 'blank') {
                cursorY += Math.max(baseLinePx * 0.45, 6)
                continue
              }
              if (cursorY > maxY + baseLinePx) break

              let blockText = ''
              let inlineSource = ''
              let blockFontSize = fontSize
              let blockFontWeight = baseWeight
              let blockFontStyle: 'normal' | 'bold' | 'italic' | 'bold italic' = fontStyle === 'bold' ? 'bold' : 'normal'
              let blockFill = fillColor
              let indentPx = 0
              let afterGap = 0
              let quoteBar: React.ReactNode = null

              if (block.kind === 'heading') {
                blockText = block.text
                inlineSource = String(block.raw ?? block.text ?? '')
                const level = Math.max(1, Math.min(block.level, 6))
                const scales = [1.18, 1.1, 1.04, 1, 1, 1]
                blockFontSize = Math.max(fontSize, Math.round(fontSize * scales[level - 1]))
                blockFontWeight = '700'
                blockFontStyle = 'bold'
                blockFill = level <= 2 ? '#0f172a' : '#1f2937'
                afterGap = Math.max(blockFontSize * 0.12, 2)
              } else if (block.kind === 'list-item') {
                const prefix = block.ordered
                  ? `${Number.isFinite(block.index as number) ? block.index : 1}. `
                  : (typeof block.checked === 'boolean'
                      ? `${block.checked ? '[x]' : '[ ]'} `
                      : '• ')
                indentPx = Math.max(0, block.indent) * Math.max(fontSize * 0.7, 10)
                blockText = `${prefix}${block.text}`
                inlineSource = String(block.raw ?? block.text ?? '')
              } else if (block.kind === 'quote') {
                const depth = Math.max(1, block.depth)
                indentPx = depth * Math.max(fontSize * 0.45, 8) + 8
                blockText = block.text || ''
                inlineSource = String(block.raw ?? block.text ?? '')
                blockFill = '#475569'
                blockFontStyle = blockFontStyle === 'bold' ? 'bold italic' : 'italic'
                const barX = d.x + (depth - 1) * Math.max(fontSize * 0.4, 6)
                const barW = Math.max(2 / view.scale, 1.5 / view.scale)
                quoteBar = (
                  <KRect
                    key={`md-quote-bar-${d.id}-${i}`}
                    x={barX}
                    y={cursorY + 1}
                    width={barW}
                    height={Math.max(baseLinePx - 2, 4)}
                    fill="rgba(148,163,184,0.8)"
                    cornerRadius={barW / 2}
                    listening={false}
                  />
                )
              } else {
                blockText = block.text
                inlineSource = String(block.raw ?? block.text ?? '')
              }

              if (!blockText.trim()) {
                cursorY += Math.max(baseLinePx * 0.35, 4)
                continue
              }

              const textX = d.x + indentPx
              const textWidth = Math.max(24, boxW - indentPx)
              const measured = computeTextBoxLayout({
                text: blockText,
                fontFamily,
                fontSize: blockFontSize,
                fontWeight: blockFontWeight,
                baseWidth: textWidth,
                baseHeight: Math.max(blockFontSize * lineHeight, 1),
                growDir: 'down',
                padding: 0,
                lineHeight,
              })
              const blockHeight = Math.max(
                Math.max(blockFontSize * lineHeight, 1),
                measured.contentHeight,
              )
              let renderedBlockHeight = blockHeight
              const canInlineStyle =
                hasInlineMarkdownStyle(inlineSource) &&
                textWidth > 12
              if (canInlineStyle) {
                const runs = parseMarkdownInlineRuns(inlineSource)
                const prefixLen = Math.max(0, blockText.length - block.text.length)
                const prefixText = prefixLen > 0 ? blockText.slice(0, prefixLen) : ''
                const baseFontWeightForRun = String(blockFontWeight || '400')
                const baseIsBold = blockFontStyle === 'bold' || blockFontStyle === 'bold italic' || Number(baseFontWeightForRun) >= 600
                const baseIsItalic = blockFontStyle === 'italic' || blockFontStyle === 'bold italic'
                type MdInlineSeg = {
                  text: string
                  fontFamily: string
                  fontSize: number
                  fontWeight: string
                  fontStyle: 'normal' | 'bold' | 'italic' | 'bold italic'
                  fill: string
                  code?: boolean
                }
                const prefixW = prefixText
                  ? measureTextWidth(prefixText, fontFamily, blockFontSize, baseFontWeightForRun)
                  : 0
                const continuationIndent = prefixText
                  ? Math.min(Math.max(prefixW, 0), Math.max(textWidth - 12, 0))
                  : 0
                const segments: MdInlineSeg[] = []
                if (prefixText) {
                  segments.push({
                    text: prefixText,
                    fontFamily,
                    fontSize: blockFontSize,
                    fontWeight: baseFontWeightForRun,
                    fontStyle: blockFontStyle,
                    fill: blockFill,
                  })
                }
                for (const run of runs) {
                  const runText = String(run.text ?? '')
                  if (!runText) continue
                  const runFontFamily = run.code ? 'monospace' : fontFamily
                  const runFontSize = run.code ? Math.max(12, Math.round(blockFontSize * 0.92)) : blockFontSize
                  const runWeight = (run.bold || baseIsBold) ? '700' : baseFontWeightForRun
                  const runItalic = !!run.italic || baseIsItalic
                  const runFontStyle: 'normal' | 'bold' | 'italic' | 'bold italic' =
                    (runWeight === '700' && runItalic)
                      ? 'bold italic'
                      : (runWeight === '700')
                        ? 'bold'
                        : runItalic
                          ? 'italic'
                          : 'normal'
                  segments.push({
                    text: runText,
                    fontFamily: runFontFamily,
                    fontSize: runFontSize,
                    fontWeight: runWeight,
                    fontStyle: runFontStyle,
                    fill: run.code ? '#334155' : blockFill,
                    code: !!run.code,
                  })
                }

                const lineHeightPx = Math.max(blockFontSize * lineHeight, blockFontSize)
                let lineIndex = 0
                let lineCursorX = textX
                let lineCursorY = cursorY
                let lineAvail = textWidth
                let lineHasContent = false
                let runIndex = 0

                const lineStartX = (index: number) => textX + (index > 0 ? continuationIndent : 0)
                const lineWidth = (index: number) => Math.max(12, textWidth - (index > 0 ? continuationIndent : 0))
                const startNewLine = () => {
                  lineIndex += 1
                  lineCursorY = cursorY + lineIndex * lineHeightPx
                  lineCursorX = lineStartX(lineIndex)
                  lineAvail = lineWidth(lineIndex)
                  lineHasContent = false
                }
                const fitPrefixByWidth = (txt: string, seg: MdInlineSeg, availWidth: number) => {
                  if (!txt) return { chunk: '', width: 0 }
                  if (availWidth <= 1) return { chunk: txt.slice(0, 1), width: measureTextWidth(txt.slice(0, 1), seg.fontFamily, seg.fontSize, seg.fontWeight) }
                  let lo = 1
                  let hi = txt.length
                  let bestLen = 1
                  let bestWidth = measureTextWidth(txt.slice(0, 1), seg.fontFamily, seg.fontSize, seg.fontWeight)
                  while (lo <= hi) {
                    const mid = Math.floor((lo + hi) / 2)
                    const chunk = txt.slice(0, mid)
                    const w = measureTextWidth(chunk, seg.fontFamily, seg.fontSize, seg.fontWeight)
                    if (w <= availWidth + 1e-3) {
                      bestLen = mid
                      bestWidth = w
                      lo = mid + 1
                    } else {
                      hi = mid - 1
                    }
                  }
                  return { chunk: txt.slice(0, bestLen), width: bestWidth }
                }

                for (const seg of segments) {
                  let remaining = seg.text
                  while (remaining) {
                    if (!lineHasContent && !seg.code) {
                      const trimmedLeading = remaining.replace(/^\s+/, '')
                      if (trimmedLeading !== remaining) {
                        remaining = trimmedLeading
                        if (!remaining) break
                      }
                    }
                    if (lineAvail <= 1 && lineHasContent) {
                      startNewLine()
                      continue
                    }
                    let rawW = measureTextWidth(remaining, seg.fontFamily, seg.fontSize, seg.fontWeight)
                    let chunk = remaining
                    if (rawW > lineAvail + 1e-3) {
                      const canSplitThisLine = !seg.code && remaining.length > 1 && lineAvail > 1
                      if (canSplitThisLine) {
                        const fit = fitPrefixByWidth(remaining, seg, lineAvail)
                        if (fit.chunk && fit.chunk.length < remaining.length) {
                          chunk = fit.chunk
                          rawW = fit.width
                        } else if (lineHasContent) {
                          startNewLine()
                          continue
                        } else {
                          chunk = fit.chunk
                          rawW = fit.width
                        }
                      } else if (lineHasContent) {
                        startNewLine()
                        continue
                      } else {
                        const fit = fitPrefixByWidth(remaining, seg, lineAvail)
                        chunk = fit.chunk
                        rawW = fit.width
                      }
                    }
                    if (!seg.code && chunk.length < remaining.length) {
                      const trimmedRight = chunk.replace(/\s+$/g, '')
                      if (trimmedRight && trimmedRight.length < chunk.length) {
                        chunk = trimmedRight
                        rawW = measureTextWidth(chunk, seg.fontFamily, seg.fontSize, seg.fontWeight)
                      }
                    }
                    if (!chunk) break
                    const codePad = seg.code ? 8 : 0
                    if (seg.code) {
                      nodes.push(
                        <KRect
                          key={`md-code-bg-${d.id}-${i}-${runIndex}`}
                          x={lineCursorX}
                          y={lineCursorY + Math.max(seg.fontSize * 0.12, 1)}
                          width={Math.max(1, Math.min(rawW + 6, lineAvail))}
                          height={Math.max(4, lineHeightPx - Math.max(seg.fontSize * 0.22, 2))}
                          cornerRadius={Math.max(3 / view.scale, 1.5)}
                          fill="rgba(148,163,184,0.18)"
                          stroke="rgba(148,163,184,0.26)"
                          strokeWidth={0.6 / view.scale}
                          listening={false}
                        />
                      )
                    }
                    nodes.push(
                      <KText
                        key={`md-run-${d.id}-${i}-${runIndex}`}
                        x={lineCursorX + (seg.code ? 3 : 0)}
                        y={lineCursorY}
                        width={Math.max(1, Math.min(rawW + 1, lineAvail))}
                        height={Math.max(1, lineHeightPx)}
                        text={chunk}
                        fontFamily={seg.fontFamily}
                        fontSize={seg.fontSize}
                        fontStyle={seg.fontStyle}
                        fill={seg.fill}
                        opacity={textOpacity}
                        align="left"
                        verticalAlign="top"
                        listening={false}
                        wrap="none"
                        lineHeight={lineHeight}
                      />
                    )
                    lineCursorX += rawW + codePad
                    lineAvail = Math.max(0, lineAvail - rawW - codePad)
                    lineHasContent = true
                    remaining = remaining.slice(chunk.length)
                    runIndex += 1
                    if (remaining) startNewLine()
                  }
                }
                renderedBlockHeight = Math.max(blockHeight, (lineIndex + 1) * lineHeightPx)
              } else {
                nodes.push(
                  <KText
                    key={`md-block-${d.id}-${i}`}
                    x={textX}
                    y={cursorY}
                    width={textWidth}
                    height={Math.max(1, blockHeight)}
                    text={measured.renderedText || blockText}
                    fontFamily={fontFamily}
                    fontSize={blockFontSize}
                    fontStyle={blockFontStyle}
                    fill={blockFill}
                    opacity={textOpacity}
                    align="left"
                    verticalAlign="top"
                    listening={false}
                    wrap="char"
                    lineHeight={lineHeight}
                  />
                )
              }
              if (quoteBar) {
                quoteBar = React.cloneElement(quoteBar as React.ReactElement<any>, {
                  height: Math.max(renderedBlockHeight - 2, 4),
                })
                nodes.push(quoteBar)
              }
              cursorY += renderedBlockHeight + afterGap
            }
            markdownStyledContent = (
              <Group
                listening={false}
                clipX={d.x}
                clipY={d.y}
                clipWidth={boxW}
                clipHeight={boxH}
              >
                {nodes}
              </Group>
            )
          }
        }
        return (
          <Group listening={false}>
            <KRect
              x={d.x}
              y={d.y}
              width={boxW}
              height={boxH}
              cornerRadius={frameCorner}
              stroke={borderColor}
              strokeWidth={1 / view.scale}
              dash={borderDash}
              fill={fillColorOverlay}
              shadowColor={borderColor}
              shadowBlur={frameShadowBlur}
              shadowOpacity={frameShadowBlur > 0 ? 0.18 : 0}
              opacity={0.85}
            />
            {markdownStyledContent ?? (
              <KText
                x={d.x}
                y={d.y}
                width={boxW}
                height={boxH}
                text={displayText}
                fontFamily={fontFamily}
                fontSize={fontSize}
                fontStyle={fontStyle}
                fill={fillColor}
                opacity={textOpacity}
                align={autoCenterText ? 'center' : 'left'}
                verticalAlign="top"
                listening={false}
                wrap="char"
                lineHeight={lineHeight}
              />
            )}
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
    const content = String(meta.text ?? draft.text ?? target.text ?? '')
    const summary = String(meta.summary ?? target.summary ?? '')
    const fontFamily = String(meta.fontFamily ?? currentMeta.fontFamily ?? 'sans-serif')
    const fontWeight = String(meta.fontWeight ?? currentMeta.fontWeight ?? '400')
    const fontSize = Number(meta.fontSize ?? currentMeta.fontSize ?? 16) || 16
    const role = String(meta.role ?? inferTextRole(fontSize, fontWeight))
    const growDir = (meta.growDir as TextGrowDir) ?? (currentMeta.growDir as TextGrowDir) ?? 'right-down'
    const padding = Number(meta.padding ?? currentMeta.padding ?? 0)
    const baseWidth = Number(meta.configuredWidth ?? currentMeta.configuredWidth ?? target.w ?? 240)
    const baseHeight = Number(meta.configuredHeight ?? currentMeta.configuredHeight ?? target.h ?? 160)
    const rawLineHeight = typeof meta.lineHeight === 'number'
      ? Number(meta.lineHeight)
      : typeof currentMeta.lineHeight === 'number'
        ? Number(currentMeta.lineHeight)
        : TEXT_LINE_HEIGHT
    const renderedText = renderMarkdownToCanvasText(content)
    const layout = computeTextBoxLayout({
      text: renderedText,
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
      role,
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
    const acceptedShapeIds: string[] = []
    const userChangeTrackedShapeIds: string[] = []
    const baselineShapes: Record<string, ReturnType<typeof shapeToSnapshot>> = {}
    let targetBBox: BBox | null = null
    const seenAccepted = new Set<string>()
    for (const draft of entry.drafts) {
      const meta = draft.meta ?? {}
      const acceptedShapeId = draft.kind === 'edit'
        ? String(draft.targetId ?? meta.targetId ?? meta.target ?? meta.id ?? '').trim()
        : String(draft.id || '').trim()
      if (!acceptedShapeId || seenAccepted.has(acceptedShapeId)) continue
      const acceptedShape = nextShapes.find((shape) => String(shape.id) === acceptedShapeId)
      if (!acceptedShape) continue
      seenAccepted.add(acceptedShapeId)
      acceptedShapeIds.push(acceptedShapeId)
      if (draft.kind === 'text' && acceptedShape.kind === 'text') {
        userChangeTrackedShapeIds.push(acceptedShapeId)
      }
      baselineShapes[acceptedShapeId] = shapeToSnapshot(acceptedShape)
      targetBBox = mergeBBox(targetBBox, computeShapeBBox(acceptedShape))
    }
    if (experimentActive && acceptedShapeIds.length > 0) {
      const activeBlockIdSet = new Set((entry.activeBlockIds ?? []).map((blockId) => String(blockId)))
      const activeAligned = graphBlockCards
        .filter((block) => activeBlockIdSet.has(block.blockId))
        .some((block) => bboxIntersects(targetBBox, block.bbox ?? null))
      commitExperimentRun((current) => addAcceptedSuggestion(current, {
        payloadId: entry.payloadId,
        requestId: entry.requestId ?? null,
        phaseId: entry.phaseId ?? experimentPhaseId,
        acceptedShapeIds,
        userChangeTrackedShapeIds,
        baselineShapes,
        usableUnits: estimateDraftUsableUnits(entry.drafts),
        targetBBox,
        activeBlockIds: [...(entry.activeBlockIds ?? [])],
        planTargetBlockIds: [...(entry.planTargetBlockIds ?? [])],
        activeBlockAligned: activeAligned,
      }))
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
  }, [currentPayloadId, previews, pushHistory, noteUserAction, shapes, drawStack, applyEditDraftToState, experimentActive, graphBlockCards, commitExperimentRun, experimentPhaseId])
  const dismissAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    const entry = previews[currentPayloadId]
    if (entry) {
      recordExperimentDismiss(entry.payloadId, entry.requestId ?? null, entry.phaseId ?? experimentPhaseId, 'manual')
    }
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    clearAutoTimer()
  }, [currentPayloadId, previews, recordExperimentDismiss, experimentPhaseId, clearAutoTimer])
  const buildEditPreviewNode = useCallback((draft: ShapeDraft, key: string) => {
    const meta = draft.meta ?? {}
    const targetId = (meta.targetId ?? draft.targetId) as string | undefined
    const target = targetId ? shapesById[targetId] : undefined
    const baseX = target?.x ?? draft.x ?? 0
    const baseY = target?.y ?? draft.y ?? 0
    const baseWidth = target?.w ?? draft.w ?? 220
    const operator = String(meta.operation ?? draft.operation ?? 'Edit­')
    const content = String(meta.text ?? draft.text ?? '')
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
        setSelectDeleteDragActive(false)
        setSelectDeleteHover(false)
        selectDragRef.current = null
        setSuspendSessionSync(false)
        return
      }
      noteUserAction()
      setSelectedShapeId(target.id)
      setSelectDeleteDragActive(false)
      setSelectDeleteHover(false)
      setSuspendSessionSync(false)
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
      if (!drag) {
        setSelectDeleteDragActive(false)
        setSelectDeleteHover(false)
        return
      }
      const wpt = screenToWorld(pos.x, pos.y)
      const nextX = wpt.x - drag.offsetX
      const nextY = wpt.y - drag.offsetY
      if (!drag.moved) {
        const dist = Math.hypot(nextX - drag.startX, nextY - drag.startY)
        if (dist > 0.5) {
          drag.moved = true
          setSelectDeleteDragActive(true)
          setSuspendSessionSync(true)
          pushHistory()
        } else {
          setSelectDeleteHover(false)
          return
        }
      }
      moveTextShape(drag.id, nextX, nextY)
      setSelectDeleteHover(isPointInSelectDeleteZone(pos.x, pos.y))
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
  }, [isDrawing, boxDraft, snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld, textEditor, moveTextShape, pushHistory, isPointInSelectDeleteZone])
  const onMouseUp = useCallback(() => {
    if (textEditor) return
    if (toolMode === 'hand') return
    if (toolMode === 'select') {
      const drag = selectDragRef.current
      const targetId = drag?.id ?? selectedShapeId
      const shouldDelete = !!(drag && drag.moved && selectDeleteHover)
      selectDragRef.current = null
      setSelectDeleteDragActive(false)
      setSelectDeleteHover(false)
      setSuspendSessionSync(false)
      if (drag && drag.moved) {
        if (shouldDelete && drag.id) {
          deleteShapeById(drag.id, { skipHistory: true })
        }
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
        const meta = boxDraft.meta ?? {}
        const fontSize = Number((meta as any).fontSize ?? textSettings.fontSize) || textSettings.fontSize
        const defaults = getDefaultTextBoxSize(fontSize)
        width = defaults.width
        height = defaults.height
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
  }, [isDrawing, rawPoints, boxDraft, snap, snapPoint, toolMode, curveTurns, currentBrush, pushHistory, setShapes, setDrawStack, brushColor, openTextEditor, textEditor, openEditorForShape, shapes, selectedShapeId, selectDeleteHover, deleteShapeById, textSettings.fontSize, getDefaultTextBoxSize])
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
  React.useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const tgt = ev.target as HTMLElement | null
      const isTyping = !!(tgt && (
        tgt.tagName === 'INPUT' ||
        tgt.tagName === 'TEXTAREA' ||
        (tgt as any).isContentEditable
      ))
      if (isTyping) return

      const key = (ev.key || '').toLowerCase()
      const hasPrimaryMod = ev.ctrlKey || ev.metaKey

      if (hasPrimaryMod && !ev.altKey) {
        if (key === 'z') {
          ev.preventDefault()
          if (ev.shiftKey) redo()
          else undo()
          return
        }
        if (key === 'y' && !ev.shiftKey) {
          ev.preventDefault()
          redo()
          return
        }
      }

      if (hasPrimaryMod || ev.altKey) return

      if (key === 'escape') {
        if (currentPayloadId) {
          ev.preventDefault()
          dismissAI()
        }
        return
      }
      if (key === 'delete' || key === 'backspace') {
        if (selectedShapeId) {
          ev.preventDefault()
          deleteSelectedShape()
        }
        return
      }

      switch (key) {
        case 'h':
          ev.preventDefault()
          setToolMode('hand')
          return
        case 'v':
          ev.preventDefault()
          setToolMode('select')
          return
        case 'p':
          ev.preventDefault()
          setToolMode('pen')
          return
        case 'e':
          ev.preventDefault()
          setToolMode('eraser')
          return
        case 'o':
          ev.preventDefault()
          setToolMode('ellipse')
          return
        case 't':
          ev.preventDefault()
          setToolMode('text')
          return
        case 'g':
          ev.preventDefault()
          toggleGrid()
          return
        case 's':
          ev.preventDefault()
          toggleSnap()
          return
        case 'c':
          ev.preventDefault()
          toggleCurveTurns()
          return
        default:
          return
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    undo,
    redo,
    currentPayloadId,
    dismissAI,
    selectedShapeId,
    deleteSelectedShape,
    setToolMode,
    toggleGrid,
    toggleSnap,
    toggleCurveTurns,
  ])
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
      setSelectDeleteDragActive(false)
      setSelectDeleteHover(false)
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
  const selectEditDialogActive = !!(textEditor?.isEditing && toolMode === 'select')
  const centerTextEditorDialog = !!textEditor && (!textEditor.isEditing || selectEditDialogActive)
  const fixedSelectEditDialogSize = selectEditDialogActive
    ? {
        width: Math.max(320, Math.min(560, size.width - 24)),
        height: Math.max(640, Math.min(800, size.height - 88)),
      }
    : null
  const textEditorScreen = textEditor ? worldToScreen(textEditor.x, textEditor.y) : null
  const textEditorSize = textEditor ? {
    width: Math.max(textEditor.w * view.scale, 260),
    height: Math.max(textEditor.h * view.scale, 200),
  } : null
  const formatExperimentMetric = useCallback((value: number | null | undefined, digits = 2) => {
    if (value == null || Number.isNaN(value)) return 'n/a'
    return Number(value).toFixed(digits)
  }, [])
  const formatExperimentPercent = useCallback((value: number | null | undefined) => {
    if (value == null || Number.isNaN(value)) return 'n/a'
    return `${(value * 100).toFixed(1)}%`
  }, [])
  const experimentStatusLabel = experimentActive ? 'RUNNING' : experimentRun ? 'ENDED' : 'IDLE'
// ===== Stage binds camera (x/y/scale); hand mode enables dragging =====
  return (
    <div
      ref={rootRef}
      style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background:
          'radial-gradient(circle at 12% 10%, rgba(14,165,233,0.1), transparent 38%), radial-gradient(circle at 85% 18%, rgba(34,197,94,0.08), transparent 36%), linear-gradient(180deg, rgba(255,255,255,0.7), rgba(248,250,252,0.55))',
      }}
    >
      <div
        aria-hidden
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'radial-gradient(circle at 20% 24%, rgba(255,255,255,0.55), transparent 46%), radial-gradient(circle at 78% 30%, rgba(255,255,255,0.42), transparent 44%), radial-gradient(circle at 50% 68%, rgba(148,163,184,0.08), transparent 55%)',
          maskImage: 'radial-gradient(circle at 50% 45%, black 35%, transparent 92%)',
        }}
      />
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: -120,
          left: -80,
          width: 360,
          height: 360,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(37,99,235,0.14), rgba(37,99,235,0.02) 58%, transparent 72%)',
          filter: 'blur(4px)',
          pointerEvents: 'none',
        }}
      />
      <div
        aria-hidden
        style={{
          position: 'absolute',
          right: -120,
          bottom: 120,
          width: 420,
          height: 420,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(20,184,166,0.1), rgba(20,184,166,0.02) 56%, transparent 72%)',
          filter: 'blur(8px)',
          pointerEvents: 'none',
        }}
      />
      <TopToolbar
        onAskAI={askAI}
        onAcceptAI={acceptAI}
        onDismissAI={dismissAI}
        autoComplete={autoComplete}
        autoCountdown={autoCountdown}
        hasActivePreview={hasActivePreview}
        onToggleAutoComplete={handleAutoCompleteToggle}
        preferExplanatoryDrawing={preferExplanatoryDrawing}
        onTogglePreferExplanatoryDrawing={setPreferExplanatoryDrawing}
        onSaveProjectCurrent={() => { void handleTopbarSaveProject() }}
        projectHasBinding={!!currentProjectId}
        projectBoundName={currentProjectName || undefined}
        projectSavePending={projectSavePending}
        projectSaveFlash={projectSaveFlash}
      />
      <SettingsButton
        open={settingsOpen}
        onToggle={() => setSettingsOpen((value) => !value)}
      />
      {experimentWidget.open ? (
        <div
          style={{
            position: 'absolute',
            top: experimentWidget.y,
            left: experimentWidget.x,
            zIndex: 1120,
            width: experimentPanelWidth,
            maxWidth: 'calc(100vw - 96px)',
            borderRadius: 18,
            border: '1px solid rgba(148,163,184,0.24)',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
            boxShadow: '0 18px 40px rgba(15,23,42,0.14)',
            backdropFilter: 'blur(12px) saturate(120%)',
            padding: 14,
            display: 'grid',
            gap: 10,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: '.08em', color: '#0f172a' }}>EXPERIMENT</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                Status: {experimentStatusLabel}{experimentRun ? ` · ${experimentRun.runId}` : ''}
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  color: experimentActive ? '#166534' : experimentRun ? '#92400e' : '#475569',
                  background: experimentActive
                    ? 'rgba(220,252,231,0.96)'
                    : experimentRun
                      ? 'rgba(254,243,199,0.96)'
                      : 'rgba(226,232,240,0.96)',
                  borderRadius: 999,
                  padding: '4px 8px',
                  letterSpacing: '.06em',
                }}
              >
                {experimentStatusLabel}
              </span>
              <button
                type="button"
                onClick={closeExperimentWidget}
                title="关闭实验面板"
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: 999,
                  border: '1px solid rgba(148,163,184,0.28)',
                  background: 'rgba(255,255,255,0.78)',
                  color: '#64748b',
                  cursor: 'pointer',
                  fontSize: 16,
                  lineHeight: 1,
                }}
              >
                ×
              </button>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '88px 1fr', alignItems: 'center', gap: 8 }}>
            <label htmlFor="experiment-phase" style={{ fontSize: 11, fontWeight: 700, color: '#334155' }}>
              Phase ID
            </label>
            <input
              id="experiment-phase"
              type="text"
              value={experimentPhaseId}
              onChange={(e) => handleExperimentPhaseChange(e.target.value)}
              placeholder="phase-1"
              style={{ ...INPUT_BASE, width: '100%', padding: '7px 10px', fontSize: 12 }}
            />
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <button
              type="button"
              onClick={experimentActive ? endExperiment : startExperiment}
              style={{
                ...BUTTON_BASE,
                padding: '7px 12px',
                fontSize: 12,
                color: experimentActive ? '#b91c1c' : '#166534',
                border: experimentActive ? '1px solid rgba(248,113,113,0.28)' : '1px solid rgba(34,197,94,0.28)',
                background: experimentActive ? 'rgba(254,242,242,0.86)' : 'rgba(240,253,244,0.92)',
              }}
            >
              {experimentActive ? '结束实验' : '开始实验'}
            </button>
            <button
              type="button"
              onClick={exportExperiment}
              disabled={!experimentRun}
              style={{
                ...BUTTON_BASE,
                padding: '7px 12px',
                fontSize: 12,
                opacity: experimentRun ? 1 : 0.55,
              }}
            >
              导出 JSON
            </button>
          </div>
          {experimentSummary ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '6px 10px', fontSize: 11, color: '#334155' }}>
                <span>ai_invoke_times</span>
                <strong style={{ color: '#0f172a' }}>{experimentSummary.aiInvokeTimes}</strong>
                <span>suggestion_acceptance_rate</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentPercent(experimentSummary.suggestionAcceptanceRate)}</strong>
                <span>dismiss_rate</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentPercent(experimentSummary.dismissRate)}</strong>
                <span>straight_use_rate</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentPercent(experimentSummary.straightUseRate)}</strong>
                <span>user_changed_rate</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentPercent(experimentSummary.userChangedRate)}</strong>
                <span>prompt_tokens_per_round</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentMetric(experimentSummary.promptTokensPerRound, 1)}</strong>
                <span>accepted_usable_content_per_1k_tokens</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentMetric(experimentSummary.acceptedUsableContentPer1kTokens, 2)}</strong>
                <span>active_block_alignment_rate</span>
                <strong style={{ color: '#0f172a' }}>{formatExperimentPercent(experimentSummary.activeBlockAlignmentRate)}</strong>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 10, color: '#64748b' }}>
                <span>preview={experimentSummary.previewCount}</span>
                <span>accept={experimentSummary.acceptCount}</span>
                <span>dismiss={experimentSummary.dismissCount}</span>
                <span>prompt_tokens={experimentSummary.totalPromptTokens}</span>
                <span>usable_units={experimentSummary.acceptedUsableUnits}</span>
                <span>accepted_text_chars={experimentSummary.acceptedTextChars}</span>
                <span>changed_text_chars={experimentSummary.changedTextChars}</span>
              </div>
              <div style={{ borderTop: '1px solid rgba(148,163,184,0.2)', paddingTop: 8, display: 'grid', gap: 6 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#334155' }}>phase_specific_efficiency</div>
                {experimentSummary.phaseSpecificEfficiency.length === 0 ? (
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>尚无 phase 数据</div>
                ) : (
                  experimentSummary.phaseSpecificEfficiency.map((item) => (
                    <div
                      key={item.phaseId}
                      style={{
                        border: '1px solid rgba(148,163,184,0.18)',
                        borderRadius: 12,
                        padding: '8px 10px',
                        background: 'rgba(255,255,255,0.78)',
                        display: 'grid',
                        gap: 4,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                        <strong style={{ fontSize: 11, color: '#0f172a' }}>{item.phaseId}</strong>
                        <span style={{ fontSize: 10, color: '#64748b' }}>invoke={item.invoke}</span>
                      </div>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 10, color: '#475569' }}>
                        <span>straight-use={formatExperimentPercent(item.straightUseRate)}</span>
                        <span>accepted/1k={formatExperimentMetric(item.acceptedOutputPer1kToken, 2)}</span>
                        <span>usable={item.acceptedUsableUnits}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <div style={{ fontSize: 11, color: '#94a3b8' }}>点击“开始实验”后开始实时统计。</div>
          )}
        </div>
      ) : (
        <button
          type="button"
          onClick={() => {
            if (experimentDragSuppressClickRef.current) {
              experimentDragSuppressClickRef.current = false
              return
            }
            openExperimentWidget()
          }}
          onPointerDown={onExperimentChipPointerDown}
          onPointerMove={onExperimentChipPointerMove}
          onPointerUp={onExperimentChipPointerUp}
          onPointerCancel={onExperimentChipPointerCancel}
          title="点击展开实验面板，按住拖动位置"
          style={{
            position: 'absolute',
            top: experimentWidget.y,
            left: experimentWidget.x,
            zIndex: 1120,
            minHeight: 44,
            minWidth: experimentChipWidth,
            padding: '10px 14px',
            borderRadius: 999,
            border: '1px solid rgba(148,163,184,0.28)',
            background: 'linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,245,249,0.94))',
            boxShadow: '0 14px 28px rgba(15,23,42,0.12)',
            color: '#0f172a',
            cursor: 'grab',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: experimentActive ? '#22c55e' : experimentRun ? '#f59e0b' : '#94a3b8',
              boxShadow: experimentActive ? '0 0 0 4px rgba(34,197,94,0.16)' : 'none',
              flexShrink: 0,
            }}
          />
          <span style={{ display: 'grid', textAlign: 'left' }}>
            <strong style={{ fontSize: 12, letterSpacing: '.05em' }}>Experiment</strong>
            <span style={{ fontSize: 10, color: '#64748b' }}>{experimentStatusLabel}</span>
          </span>
        </button>
      )}
      {!aiFeedSidebarOpen && (
        <button
          type="button"
          title="Open AI feed"
          onClick={() => setAiFeedSidebarOpen(true)}
          style={{
            position: 'absolute',
            top: 76,
            left: 14,
            zIndex: 1100,
            height: 42,
            minWidth: 104,
            padding: '0 14px',
            borderRadius: 999,
            border: '1px solid rgba(148,163,184,0.34)',
            background: 'linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,245,249,0.94))',
            color: '#0f172a',
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: '.04em',
            cursor: 'pointer',
            boxShadow: '0 8px 16px rgba(15,23,42,0.12)',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}
        >
          <span aria-hidden style={{ fontSize: 14, lineHeight: 1 }}>≡</span>
          AI Feed
        </button>
      )}
      <button
        type="button"
        title={currentProjectId ? `Open Project Manager (${currentProjectName || currentProjectId})` : 'Open Project Manager'}
        onClick={() => { void openProjectManager() }}
        style={{
          position: 'absolute',
          top: aiFeedSidebarOpen ? 76 : 24,
          left: 14,
          zIndex: 1100,
          height: 48,
          minWidth: 176,
          padding: '0 16px',
          borderRadius: 14,
          border: currentProjectId ? '1px solid rgba(59,130,246,0.35)' : '1px solid rgba(148,163,184,0.34)',
          background: projectManagerOpen
            ? 'linear-gradient(135deg, rgba(219,234,254,0.98), rgba(224,231,255,0.94))'
            : currentProjectId
              ? 'linear-gradient(135deg, rgba(239,246,255,0.98), rgba(224,242,254,0.94))'
              : 'linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,245,249,0.94))',
          color: '#0f172a',
          fontSize: 13,
          fontWeight: 800,
          letterSpacing: '.03em',
          cursor: 'pointer',
          boxShadow: '0 10px 20px rgba(15,23,42,0.14)',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 10,
        }}
      >
        <span aria-hidden style={{ fontSize: 16, lineHeight: 1 }}>▣</span>
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'flex-start', lineHeight: 1.05 }}>
          <span>Project Manager</span>
          {currentProjectId ? (
            <span style={{ fontSize: 10, fontWeight: 700, color: '#64748b', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {currentProjectName || currentProjectId}
            </span>
          ) : null}
        </span>
      </button>
      {projectManagerOpen && (
        <div
          onMouseDown={(ev) => {
            if (ev.target === ev.currentTarget) setProjectManagerOpen(false)
          }}
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 2500,
            background: 'rgba(15,23,42,0.18)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div
            style={{
              width: `min(86vw, 900px)`,
              height: `min(86vh, 900px)`,
              aspectRatio: '1 / 1',
              background: 'linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96))',
              border: '1px solid rgba(148,163,184,0.28)',
              borderRadius: 22,
              boxShadow: '0 28px 80px rgba(15,23,42,0.22), 0 8px 24px rgba(15,23,42,0.12)',
              display: 'grid',
              gridTemplateRows: 'auto 1fr',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                padding: '14px 16px',
                borderBottom: '1px solid rgba(148,163,184,0.2)',
                background: 'linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,252,0.9))',
              }}
            >
              <div>
                <div style={{ fontSize: 14, fontWeight: 800, color: '#0f172a', letterSpacing: '.02em' }}>Project Manager</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  Current saves live state. Commit stores an immutable node with preview.
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  value={projectNameDraft}
                  onChange={(e) => setProjectNameDraft(e.target.value)}
                  placeholder="Project name"
                  style={{ ...INPUT_BASE, width: 180, padding: '6px 10px', borderRadius: 10, fontSize: 12 }}
                />
                <button
                  type="button"
                  onClick={() => { void handleCreateProject() }}
                  disabled={projectActionPending !== null}
                  style={{
                    ...BUTTON_BASE,
                    opacity: projectActionPending ? 0.65 : 1,
                    padding: '7px 12px',
                    fontSize: 12,
                  }}
                >
                  {projectActionPending === 'create' ? 'Creating…' : 'Create'}
                </button>
                <input
                  value={projectCommitMessageDraft}
                  onChange={(e) => setProjectCommitMessageDraft(e.target.value)}
                  placeholder="Commit message (optional)"
                  style={{ ...INPUT_BASE, width: 210, padding: '6px 10px', borderRadius: 10, fontSize: 12 }}
                />
                <button
                  type="button"
                  onClick={() => { void handleCommitProject() }}
                  disabled={projectActionPending !== null || !selectedProjectId}
                  style={{
                    ...BUTTON_BASE,
                    opacity: (projectActionPending || !selectedProjectId) ? 0.65 : 1,
                    padding: '7px 12px',
                    fontSize: 12,
                  }}
                >
                  {projectActionPending === 'commit' ? 'Committing…' : 'Commit'}
                </button>
                <button
                  type="button"
                  onClick={() => { void handleOpenProject() }}
                  disabled={projectActionPending !== null || !selectedProjectId}
                  style={{
                    ...BUTTON_BASE,
                    opacity: (projectActionPending || !selectedProjectId) ? 0.65 : 1,
                    padding: '7px 12px',
                    fontSize: 12,
                  }}
                >
                  {projectActionPending === 'open' ? 'Opening…' : 'Open'}
                </button>
                <button
                  type="button"
                  onClick={() => setProjectManagerOpen(false)}
                  style={{ ...BUTTON_BASE, padding: '7px 12px', fontSize: 12 }}
                >
                  Close
                </button>
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(240px, 30%) 1fr',
                minHeight: 0,
              }}
            >
              <div
                style={{
                  borderRight: '1px solid rgba(148,163,184,0.18)',
                  padding: 12,
                  overflow: 'auto',
                  background: 'linear-gradient(180deg, rgba(248,250,252,0.75), rgba(241,245,249,0.7))',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#334155', letterSpacing: '.04em', textTransform: 'uppercase' }}>
                    Projects
                  </div>
                  <button
                    type="button"
                    onClick={() => { void refreshProjectList() }}
                    style={{ ...BUTTON_BASE, padding: '4px 10px', fontSize: 11 }}
                  >
                    Refresh
                  </button>
                </div>
                {projectManagerBusy && projectList.length === 0 ? (
                  <div style={{ fontSize: 12, color: '#64748b' }}>Loading projects…</div>
                ) : projectList.length === 0 ? (
                  <div style={{ fontSize: 12, color: '#64748b' }}>
                    No local projects yet. Create one, then use topbar Save to persist current state.
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: 8 }}>
                    {projectList.map((proj) => {
                      const active = selectedProjectId === proj.projectId
                      const stats = proj.stats || {}
                      const previewUrl = proj.currentPreview ? apiUrl(`/project/current/image?project_id=${encodeURIComponent(proj.projectId)}`) : null
                      return (
                        <button
                          key={proj.projectId}
                          type="button"
                          onClick={() => setSelectedProjectId(proj.projectId)}
                          onContextMenu={(ev) => {
                            ev.preventDefault()
                            ev.stopPropagation()
                            setProjectContextMenuState({ kind: 'project', projectId: proj.projectId, x: ev.clientX, y: ev.clientY })
                          }}
                          style={{
                            textAlign: 'left',
                            borderRadius: 14,
                            padding: '10px 11px',
                            border: active ? '1px solid rgba(59,130,246,0.45)' : '1px solid rgba(148,163,184,0.22)',
                            background: active
                              ? 'linear-gradient(180deg, rgba(239,246,255,0.96), rgba(224,242,254,0.9))'
                              : 'linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.86))',
                            boxShadow: active ? '0 10px 20px rgba(59,130,246,0.1)' : '0 4px 12px rgba(15,23,42,0.04)',
                            cursor: 'pointer',
                            color: '#0f172a',
                            display: 'grid',
                            gap: 8,
                          }}
                        >
                          <div
                            style={{
                              aspectRatio: '16 / 10',
                              borderRadius: 10,
                              border: '1px solid rgba(148,163,184,0.14)',
                              overflow: 'hidden',
                              background: 'linear-gradient(135deg, rgba(226,232,240,0.7), rgba(241,245,249,0.8))',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            {previewUrl ? (
                              <img src={previewUrl} alt={proj.projectId} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} loading="lazy" />
                            ) : (
                              <span style={{ fontSize: 11, color: '#64748b' }}>No current preview</span>
                            )}
                          </div>
                          <div style={{ fontSize: 13, fontWeight: 700, lineHeight: 1.25 }}>
                            {proj.name || proj.projectId}
                          </div>
                          <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.35, wordBreak: 'break-all' }}>
                            {proj.projectId}
                          </div>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: '#334155' }}>
                            <span>{Number(stats.strokeCount || 0)} strokes</span>
                            <span>{Number(stats.blockCount || 0)} blocks</span>
                            <span>{Number(proj.commitCount || 0)} commits</span>
                          </div>
                          <div style={{ fontSize: 11, color: '#64748b' }}>
                            Saved: {formatProjectTime(proj.lastSavedAt || proj.updatedAt) || '—'}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateRows: 'auto 1fr', minHeight: 0 }}>
                <div style={{ padding: '12px 14px 8px', borderBottom: '1px solid rgba(148,163,184,0.14)' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#334155', letterSpacing: '.04em', textTransform: 'uppercase' }}>
                    Project History {selectedProjectId ? `• ${selectedProjectId}` : ''}
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                    Current preview updates on Save / Commit (and auto current snapshots). Commits are immutable checkpoints.
                  </div>
                  {projectManagerError && (
                    <div style={{ marginTop: 8, fontSize: 12, color: '#b91c1c', background: 'rgba(254,242,242,0.9)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, padding: '8px 10px' }}>
                      {projectManagerError}
                    </div>
                  )}
                </div>
                <div style={{ padding: 14, overflow: 'auto' }}>
                  {!selectedProjectId ? (
                    <div style={{ fontSize: 13, color: '#64748b' }}>Select a project to view current state and commits.</div>
                  ) : !projectDetail || projectDetail.projectId !== selectedProjectId ? (
                    <div style={{ fontSize: 13, color: '#64748b' }}>Loading project detail…</div>
                  ) : (
                    <div style={{ display: 'grid', gap: 14, alignContent: 'start' }}>
                      <div
                        style={{
                          borderRadius: 14,
                          border: '1px solid rgba(148,163,184,0.2)',
                          background: 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
                          boxShadow: '0 8px 18px rgba(15,23,42,0.06)',
                          overflow: 'hidden',
                        }}
                      >
                        <div style={{ padding: '10px 12px', borderBottom: '1px solid rgba(148,163,184,0.14)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                          <div style={{ fontSize: 12, fontWeight: 700, color: '#334155', letterSpacing: '.04em', textTransform: 'uppercase' }}>Current</div>
                          <div style={{ fontSize: 11, color: '#64748b' }}>
                            Ref: {String(projectDetail.meta?.currentRef || 'none')}
                          </div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 320px) 1fr', gap: 12, padding: 12 }}>
                          <div style={{ aspectRatio: '16 / 10', borderRadius: 12, overflow: 'hidden', border: '1px solid rgba(148,163,184,0.14)', background: 'linear-gradient(135deg, rgba(226,232,240,0.7), rgba(241,245,249,0.8))', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {projectDetail.current?.imageUrl ? (
                              <img src={apiUrl(projectDetail.current.imageUrl)} alt="current preview" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} loading="lazy" />
                            ) : (
                              <span style={{ fontSize: 12, color: '#64748b' }}>No current preview yet</span>
                            )}
                          </div>
                          <div style={{ display: 'grid', gap: 6, alignContent: 'start' }}>
                            <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                              {(projectList.find((p) => p.projectId === selectedProjectId)?.name) || selectedProjectId}
                            </div>
                            <div style={{ fontSize: 12, color: '#64748b' }}>
                              Current preview updated: {formatProjectTime(projectDetail.current?.updatedAt || (projectDetail.meta?.currentPreviewUpdatedAt as any)) || '—'}
                            </div>
                            {projectDetail.current?.width && projectDetail.current?.height ? (
                              <div style={{ fontSize: 11, color: '#475569' }}>
                                {projectDetail.current.width}×{projectDetail.current.height} • {String(projectDetail.current.mime || '').replace('image/', '')}
                              </div>
                            ) : null}
                            <div style={{ fontSize: 11, color: '#64748b' }}>
                              Last saved: {formatProjectTime((projectDetail.meta?.lastSavedAt as any) || (projectList.find((p) => p.projectId === selectedProjectId)?.lastSavedAt as any)) || '—'}
                            </div>
                          </div>
                        </div>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, gap: 10 }}>
                          <div style={{ fontSize: 12, fontWeight: 700, color: '#334155', letterSpacing: '.04em', textTransform: 'uppercase' }}>
                            Commits ({projectDetail.commits?.length || 0})
                          </div>
                          <div style={{ fontSize: 11, color: '#64748b' }}>
                            Click a commit to switch current (with confirmation)
                          </div>
                        </div>
                        {(projectDetail.commits?.length || 0) === 0 ? (
                          <div style={{ fontSize: 13, color: '#64748b' }}>
                            No commits yet. Use <strong>Commit</strong> to capture an immutable checkpoint.
                          </div>
                        ) : (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12, alignContent: 'start' }}>
                            {projectDetail.commits.map((commit) => {
                              const isCurrentRef = String(projectDetail.meta?.currentRef || '') === commit.commitId
                              return (
                                <button
                                  key={commit.commitId}
                                  type="button"
                                  onClick={() => { void handleCheckoutCommit(selectedProjectId, commit.commitId) }}
                                  onContextMenu={(ev) => {
                                    ev.preventDefault()
                                    ev.stopPropagation()
                                    setProjectContextMenuState({ kind: 'commit', projectId: selectedProjectId, commitId: commit.commitId, x: ev.clientX, y: ev.clientY })
                                  }}
                                  style={{
                                    textAlign: 'left',
                                    borderRadius: 14,
                                    overflow: 'hidden',
                                    border: isCurrentRef ? '1px solid rgba(34,197,94,0.34)' : '1px solid rgba(148,163,184,0.2)',
                                    background: isCurrentRef
                                      ? 'linear-gradient(180deg, rgba(240,253,244,0.96), rgba(236,253,245,0.92))'
                                      : 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
                                    boxShadow: isCurrentRef ? '0 8px 18px rgba(34,197,94,0.09)' : '0 8px 18px rgba(15,23,42,0.06)',
                                    display: 'grid',
                                    gridTemplateRows: 'auto auto',
                                    cursor: 'pointer',
                                    padding: 0,
                                  }}
                                >
                                  <div style={{ aspectRatio: '16 / 10', background: 'linear-gradient(135deg, rgba(226,232,240,0.7), rgba(241,245,249,0.7))', borderBottom: '1px solid rgba(148,163,184,0.14)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                                    {commit.imageUrl ? (
                                      <img src={apiUrl(commit.imageUrl)} alt={commit.commitId} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} loading="lazy" />
                                    ) : (
                                      <span style={{ fontSize: 12, color: '#64748b' }}>No Preview</span>
                                    )}
                                  </div>
                                  <div style={{ padding: '9px 10px', display: 'grid', gap: 5 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
                                      <div style={{ fontSize: 12, fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>
                                        {formatProjectTime(commit.createdAt) || commit.commitId}
                                      </div>
                                      {isCurrentRef ? (
                                        <span style={{ fontSize: 10, fontWeight: 700, color: '#166534', background: 'rgba(187,247,208,0.9)', borderRadius: 999, padding: '2px 6px' }}>CURRENT</span>
                                      ) : null}
                                    </div>
                                    <div style={{ fontSize: 11, color: '#64748b', wordBreak: 'break-word' }}>
                                      {commit.message || 'No message'}
                                    </div>
                                    <div style={{ fontSize: 10, color: '#94a3b8', wordBreak: 'break-all' }}>
                                      {commit.commitId}
                                    </div>
                                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: '#475569' }}>
                                      {commit.width && commit.height ? <span>{commit.width}×{commit.height}</span> : null}
                                      {commit.mime ? <span>{String(commit.mime).replace('image/', '')}</span> : null}
                                    </div>
                                  </div>
                                </button>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {projectPromptOpen && (
        <div
          onMouseDown={(ev) => {
            if (ev.target === ev.currentTarget && !projectPromptSubmitting) setProjectPromptOpen(false)
          }}
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 2600,
            background: 'rgba(15,23,42,0.18)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16,
          }}
        >
          <div
            style={{
              width: 'min(92vw, 420px)',
              borderRadius: 18,
              border: '1px solid rgba(148,163,184,0.24)',
              background: 'linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96))',
              boxShadow: '0 24px 60px rgba(15,23,42,0.2)',
              padding: 16,
              display: 'grid',
              gap: 12,
            }}
          >
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: '#0f172a' }}>Create Project Before Save</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                Topbar Save stores the current working state in a project. Enter a project name to continue.
              </div>
            </div>
            <input
              autoFocus
              value={projectPromptNameDraft}
              onChange={(e) => setProjectPromptNameDraft(e.target.value)}
              placeholder="Project name"
              style={{ ...INPUT_BASE, width: '100%', padding: '8px 10px', borderRadius: 10, fontSize: 13 }}
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                type="button"
                onClick={() => setProjectPromptOpen(false)}
                disabled={projectPromptSubmitting}
                style={{ ...BUTTON_BASE, padding: '7px 12px', fontSize: 12, opacity: projectPromptSubmitting ? 0.65 : 1 }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => { void handleConfirmProjectPromptCreateAndSave() }}
                disabled={projectPromptSubmitting || !projectPromptNameDraft.trim()}
                style={{ ...BUTTON_BASE, padding: '7px 12px', fontSize: 12, opacity: (projectPromptSubmitting || !projectPromptNameDraft.trim()) ? 0.65 : 1 }}
              >
                {projectPromptSubmitting ? 'Creating & Saving…' : 'Create & Save'}
              </button>
            </div>
          </div>
        </div>
      )}
      {projectContextMenuState && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(projectContextMenuState.x, size.width - 200),
            top: Math.min(projectContextMenuState.y, size.height - 120),
            zIndex: 2700,
            minWidth: 180,
            borderRadius: 12,
            border: '1px solid rgba(148,163,184,0.24)',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96))',
            boxShadow: '0 18px 40px rgba(15,23,42,0.18)',
            padding: 6,
          }}
          onPointerDown={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {projectContextMenuState.kind === 'project' ? (
            <button
              type="button"
              onClick={() => { void handleDeleteProject(projectContextMenuState.projectId) }}
              style={{ ...BUTTON_BASE, width: '100%', justifyContent: 'flex-start', borderRadius: 10, fontSize: 12, color: '#b91c1c', border: '1px solid rgba(248,113,113,0.2)', background: 'rgba(254,242,242,0.7)' }}
            >
              Delete Project
            </button>
          ) : (
            <button
              type="button"
              onClick={() => { void handleDeleteCommit(projectContextMenuState.projectId, projectContextMenuState.commitId) }}
              style={{ ...BUTTON_BASE, width: '100%', justifyContent: 'flex-start', borderRadius: 10, fontSize: 12, color: '#b91c1c', border: '1px solid rgba(248,113,113,0.2)', background: 'rgba(254,242,242,0.7)' }}
            >
              Delete Commit
            </button>
          )}
        </div>
      )}
      <AIFeedSidebar
        open={aiFeedSidebarOpen}
        onClose={() => setAiFeedSidebarOpen(false)}
        entries={aiFeed}
        viewportHeight={size.height}
      />
      <GraphBlocksDrawer
        open={graphBlocksDrawerOpen}
        onToggle={() => setGraphBlocksDrawerOpen((prev) => !prev)}
        visible={graphInspectorVisible && autoMaintain}
        viewportWidth={size.width}
        viewportHeight={size.height}
        graphBlocksDetailed={graphBlockCards}
        onFragmentFocus={focusOnFragment}
        onBlockFocus={focusOnBlock}
        onFragmentHover={handleFragmentHover}
        onBlockHover={handleBlockHover}
      />
      <SidePanel
        showGrid={showGrid}
        snap={snap}
        curveTurns={curveTurns}
        onToggleGrid={toggleGrid}
        onToggleSnap={toggleSnap}
        onToggleCurve={toggleCurveTurns}
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
        llmModel={llmModel}
        llmTemperature={llmTemperature}
        llmTopP={llmTopP}
        llmMaxTokens={llmMaxTokens}
        onLlmModelChange={setLlmModel}
        onLlmTemperatureChange={(value) => setLlmTemperature(clamp(value, 0, 2))}
        onLlmTopPChange={(value) => setLlmTopP(clamp(value, 0, 1))}
        onLlmMaxTokensChange={(value) => setLlmMaxTokens(Math.max(256, Math.min(32768, Math.round(value || 0))))}
        onResetLLMSettings={resetRuntimeLLMSettings}
        groupPromoteMode={groupPromoteMode}
        onGroupPromoteModeChange={(value) => setGroupPromoteMode(normalizeGroupPromoteMode(value))}
        visionImageMode={visionImageMode}
        onVisionImageModeChange={(value) => setVisionImageMode(normalizeVisionImageMode(value))}
        settingsOpen={settingsOpen}
        onCloseSettings={() => setSettingsOpen(false)}
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
            left: centerTextEditorDialog ? '50%' : textEditorScreen.x,
            top: centerTextEditorDialog ? '50%' : textEditorScreen.y,
            transform: centerTextEditorDialog ? 'translate(-50%, -50%)' : undefined,
            width: selectEditDialogActive
              ? (fixedSelectEditDialogSize?.width ?? textEditorSize.width)
              : textEditorSize.width,
            height: selectEditDialogActive ? (fixedSelectEditDialogSize?.height ?? undefined) : undefined,
            minWidth: 260,
            maxWidth: selectEditDialogActive ? (fixedSelectEditDialogSize?.width ?? undefined) : 420,
            background: 'linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,250,252,0.94))',
            border: '1px solid rgba(148,163,184,0.28)',
            borderRadius: 18,
            padding: 14,
            zIndex: 2000,
            boxShadow: '0 20px 44px rgba(15,23,42,0.16), 0 6px 16px rgba(15,23,42,0.08)',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            backdropFilter: 'blur(10px) saturate(115%)',
            overflowY: selectEditDialogActive ? 'auto' : undefined,
          }}
        >
          {(() => {
            const editorRole = inferTextRole(textEditor.fontSize, textEditor.fontWeight)
            const editorAutoCenter = editorRole === 'title' || editorRole === 'subtitle'
            return (
              <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                aria-hidden
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 8,
                  background: 'linear-gradient(135deg, rgba(37,99,235,0.16), rgba(20,184,166,0.14))',
                  border: '1px solid rgba(37,99,235,0.18)',
                  color: '#1d4ed8',
                  display: 'grid',
                  placeItems: 'center',
                  fontSize: 12,
                  fontWeight: 700,
                }}
              >
                T
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a', lineHeight: 1.1 }}>
                  {textEditor.isEditing ? 'Edit text box' : 'Create text box'}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                  Auto-wrap + {textEditor.growDir} layout
                </div>
              </div>
            </div>
            <button
              onClick={cancelTextEditor}
              style={{
                width: 28,
                height: 28,
                borderRadius: 8,
                border: '1px solid rgba(148,163,184,0.28)',
                background: 'rgba(255,255,255,0.8)',
                fontSize: 15,
                cursor: 'pointer',
                color: '#64748b',
                boxShadow: '0 1px 0 rgba(255,255,255,0.85) inset',
              }}
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 12, color: '#4b5563' }}>快捷样式</span>
            <div style={{ display: 'flex', gap: 8 }}>
              {TEXT_STYLE_PRESETS.map((preset) => {
                const active =
                  Math.round(textEditor.fontSize) === preset.fontSize
                  && textEditor.fontWeight === preset.fontWeight
                  && textEditor.color === preset.color
                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => applyTextStylePreset(preset)}
                    style={{
                      ...BUTTON_BASE,
                      flex: 1,
                      padding: '7px 10px',
                      borderColor: active ? 'rgba(74,163,255,0.45)' : 'rgba(148,163,184,0.28)',
                      background: active
                        ? 'linear-gradient(180deg, rgba(239,246,255,0.96), rgba(219,234,254,0.9))'
                        : 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92))',
                      color: active ? '#1d4ed8' : '#334155',
                      fontWeight: 600,
                    }}
                  >
                    {preset.label}
                  </button>
                )
              })}
            </div>
          </div>
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
                {(['right-down', 'down', 'right', 'up', 'left'] as const).map(dir => (
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
                    width: 26,
                    height: 26,
                    borderRadius: 8,
                    border: `2px solid ${textEditor.color === c ? '#4aa3ff' : '#e2e8f0'}`,
                    background: c === 'white' ? '#fff' : c.replace('light-', 'light'),
                    cursor: 'pointer',
                    boxShadow: textEditor.color === c
                      ? '0 0 0 3px rgba(74,163,255,0.12)'
                      : '0 1px 2px rgba(15,23,42,0.06)',
                  }}
                />
              ))}
            </div>
          </div>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 12, color: '#4b5563' }}>Content (Type ::: to trigger auto-completion, Ctrl/Cmd + Enter to save)</span>
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
                padding: '10px 12px',
                borderRadius: 12,
                border: '1px solid rgba(148,163,184,0.38)',
                background: 'linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96))',
                color: '#0f172a',
                boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.04)',
                fontFamily: textEditor.fontFamily,
                fontSize: textEditor.fontSize,
                lineHeight: TEXT_LINE_HEIGHT,
                textAlign: editorAutoCenter ? 'center' as const : 'left' as const,
                outline: 'none',
              }}
            />
          </label>
          {textEditor.completing && (
            <div style={{ fontSize: 12, color: '#2563eb' }}>In progress­</div>
          )}
          {textEditor.pendingCompletion && !textEditor.completing && (
            <div
              style={{
                fontSize: 12,
                fontStyle: 'italic',
                color: '#2563eb',
                background: 'linear-gradient(180deg, rgba(239,246,255,0.95), rgba(219,234,254,0.82))',
                border: '1px solid rgba(59,130,246,0.18)',
                padding: '7px 10px',
                borderRadius: 10,
              }}
            >
              {textEditor.pendingCompletion}
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button
              onClick={cancelTextEditor}
              style={{
                ...BUTTON_BASE,
                borderColor: 'rgba(148,163,184,0.34)',
                background: 'linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9))',
                color: '#334155',
              }}
            >
              Cancel
            </button>
            <button
              onClick={commitTextEditor}
              style={{
                ...BUTTON_BASE,
                borderColor: 'rgba(59,130,246,0.34)',
                background: 'linear-gradient(180deg, rgba(239,246,255,0.96), rgba(219,234,254,0.9))',
                color: '#1d4ed8',
                boxShadow: '0 8px 16px rgba(37,99,235,0.12), 0 1px 0 rgba(255,255,255,0.9) inset',
              }}
            >
              Save
            </button>
          </div>
              </>
            )
          })()}
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
        <Layer ref={gridLayerRef} listening={false}>{showGrid && <Grid />}</Layer>
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
            {graphRelationshipEdges.map((edge) => {
              const strokeWidth = edge.highlighted ? 3.2 : 2
              const opacity = edge.highlighted ? 0.95 : 0.65
              const shadowBlur = edge.highlighted ? 18 : 8
              return (
                <Group key={`edge:${edge.key}`} listening={false}>
                  <KArrow
                    points={edge.points}
                    stroke={edge.color}
                    fill={edge.color}
                    strokeWidth={strokeWidth}
                    pointerLength={18}
                    pointerWidth={18}
                    tension={0}
                    lineCap="round"
                    lineJoin="round"
                    opacity={opacity}
                    shadowColor={edge.color}
                    shadowBlur={shadowBlur}
                    shadowOpacity={edge.highlighted ? 0.35 : 0.18}
                  />
                  <KLabel x={edge.labelPos.x} y={edge.labelPos.y} listening={false}>
                    <KTag
                      fill={hexToRgba(edge.color, edge.highlighted ? 0.85 : 0.6)}
                      stroke={edge.color}
                      lineJoin="round"
                      cornerRadius={6}
                      shadowColor={edge.color}
                      shadowBlur={edge.highlighted ? 14 : 6}
                      shadowOpacity={edge.highlighted ? 0.35 : 0.2}
                    />
                    <KText
                      text={edge.label}
                      fontSize={12}
                      fill="#0f172a"
                      padding={6}
                    />
                  </KLabel>
                </Group>
              )
            })}
            {pendingVisionGroupOverlays.map((group) => {
              const [gx0, gy0, gx1, gy1] = group.bbox
              const width = Math.max(12, gx1 - gx0)
              const height = Math.max(12, gy1 - gy0)
              const labelY = gy0 - 24 >= 12 ? gy0 - 24 : gy0 + 8
              const stroke = group.eligible ? '#f59e0b' : '#fbbf24'
              return (
                <Group key={`vision-pending-${group.groupId}`} listening={false}>
                  <KRect
                    x={gx0}
                    y={gy0}
                    width={width}
                    height={height}
                    stroke={stroke}
                    strokeWidth={1.8}
                    dash={[8, 6]}
                    cornerRadius={12}
                    opacity={0.95}
                    shadowColor={stroke}
                    shadowBlur={10}
                    shadowOpacity={0.15}
                  />
                  <KLabel x={gx0} y={labelY} listening={false}>
                    <KTag
                      fill={hexToRgba(stroke, 0.16)}
                      stroke={stroke}
                      lineJoin="round"
                      cornerRadius={6}
                    />
                    <KText
                      text={`vision pending · ${group.count} stroke${group.count === 1 ? '' : 's'}`}
                      fontSize={11}
                      fill="#78350f"
                      padding={5}
                    />
                  </KLabel>
                </Group>
              )
            })}
            {graphSelectedFragments.map((frag) => {
              const [fx0, fy0, fx1, fy1] = frag.bbox
              const width = Math.max(8, fx1 - fx0)
              const height = Math.max(8, fy1 - fy0)
              return (
                <KRect
                  key={`graph-selection-frag-${frag.id}`}
                  x={fx0}
                  y={fy0}
                  width={width}
                  height={height}
                  stroke="#22c55e"
                  strokeWidth={2.2}
                  dash={[6, 4]}
                  cornerRadius={10}
                  fill="rgba(34,197,94,0.08)"
                  listening={false}
                  shadowColor="#22c55e"
                  shadowBlur={12}
                  shadowOpacity={0.18}
                />
              )
            })}
            {graphBlockCards.map((block) => {
              const fragments = block.fragments?.filter((frag) => frag.type === 'text' && frag.bbox) ?? []
              const hasBlockBox = !!block.bbox
              if (!fragments.length && !hasBlockBox) return null
              const blockIsHovered = hoveredGraphBlockId === block.blockId
              const blockContainsHoveredFragment = block.fragments?.some((frag) => frag.id === hoveredGraphFragmentId)
              const blockActive = blockIsHovered || blockContainsHoveredFragment
              const overlays = fragments.map((frag) => {
                if (!frag.bbox) return null
                const [fx0, fy0, fx1, fy1] = frag.bbox
                const width = Math.max(4, fx1 - fx0)
                const height = Math.max(4, fy1 - fy0)
                const fragActive = hoveredGraphFragmentId === frag.id
                const fillAlpha = fragActive ? 0.45 : 0.2
                return (
                  <KRect
                    key={frag.id}
                    x={fx0}
                    y={fy0}
                    width={width}
                    height={height}
                    fill={hexToRgba(block.color, fillAlpha)}
                    listening={false}
                    cornerRadius={8}
                    stroke={fragActive ? block.color : undefined}
                    strokeWidth={fragActive ? 1.6 : 0}
                    shadowColor={block.color}
                    shadowBlur={fragActive ? 14 : 0}
                    opacity={fragActive ? 0.9 : 0.5}
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
                      strokeWidth={blockActive ? 2.6 : 1.6}
                      dash={[10, 6]}
                      listening={false}
                      cornerRadius={14}
                      opacity={blockActive ? 1 : 0.9}
                      shadowColor={block.color}
                      shadowBlur={blockActive ? 20 : 10}
                      shadowOpacity={blockActive ? 0.35 : 0.15}
                    />
                    <KLabel x={bx0} y={labelY} listening={false}>
                      <KTag
                        fill={hexToRgba(block.color, blockActive ? 0.8 : 0.55)}
                        stroke={block.color}
                        lineJoin="round"
                        cornerRadius={6}
                        shadowColor={block.color}
                        shadowBlur={blockActive ? 14 : 6}
                        shadowOpacity={blockActive ? 0.32 : 0.2}
                      />
                      <KText
                        text={block.label || block.blockId}
                        fontSize={12}
                        fill={blockActive ? '#0f172a' : '#0f172a'}
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
      {toolMode === 'select' && selectDeleteDragActive && (
        <div
          style={{
            position: 'absolute',
            left: selectDeleteZone.left,
            top: selectDeleteZone.top,
            width: selectDeleteZone.width,
            height: selectDeleteZone.height,
            zIndex: 1140,
            pointerEvents: 'none',
            borderRadius: 14,
            border: selectDeleteHover
              ? '2px solid rgba(220,38,38,0.9)'
              : '1.5px dashed rgba(239,68,68,0.72)',
            background: selectDeleteHover
              ? 'linear-gradient(180deg, rgba(254,226,226,0.92), rgba(254,202,202,0.84))'
              : 'linear-gradient(180deg, rgba(254,242,242,0.76), rgba(254,226,226,0.7))',
            color: selectDeleteHover ? '#991b1b' : '#b91c1c',
            boxShadow: selectDeleteHover
              ? '0 14px 30px rgba(220,38,38,0.24)'
              : '0 10px 22px rgba(220,38,38,0.14)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: '.02em',
            transition: 'all 120ms ease',
          }}
        >
          {selectDeleteHover ? 'Release to delete text' : 'Drag text here to delete'}
        </div>
      )}
      {graphSelectionOverlayActive && (
        <div
          onPointerDown={onGraphSelectionPointerDown}
          onPointerMove={onGraphSelectionPointerMove}
          onPointerUp={onGraphSelectionPointerUp}
          onPointerCancel={onGraphSelectionPointerCancel}
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 1110,
            pointerEvents: 'auto',
            cursor: 'crosshair',
            background: 'transparent',
          }}
        >
          {graphSelectionRectNormalized && graphSelectionRectNormalized.width >= 2 && graphSelectionRectNormalized.height >= 2 && (
            <div
              style={{
                position: 'absolute',
                left: graphSelectionRectNormalized.left,
                top: graphSelectionRectNormalized.top,
                width: graphSelectionRectNormalized.width,
                height: graphSelectionRectNormalized.height,
                border: '1.5px dashed rgba(34,197,94,0.9)',
                background: 'rgba(34,197,94,0.08)',
                borderRadius: 10,
                boxShadow: '0 0 0 1px rgba(255,255,255,0.12) inset',
                pointerEvents: 'none',
              }}
            />
          )}
          <div
            style={{
              position: 'absolute',
              left: 16,
              bottom: 132,
              zIndex: 1111,
              pointerEvents: 'none',
              fontSize: 12,
              color: '#166534',
              background: 'rgba(240,253,244,0.9)',
              border: '1px solid rgba(34,197,94,0.35)',
              borderRadius: 999,
              padding: '6px 10px',
              boxShadow: '0 6px 16px rgba(21,128,61,0.12)',
            }}
          >
            Drag to select fragments for block create/move
          </div>
        </div>
      )}
      {graphInspectorVisible && autoMaintain && pendingVisionGroupOverlays.map((group) => {
        const busy = promoteVisionGroupPending === group.groupId
        return (
          <button
            key={`vision-pending-btn-${group.groupId}`}
            onClick={() => promoteVisionPendingGroup(group.groupId)}
            disabled={busy}
            title={`${group.groupId}${group.readyReason ? ` · ${group.readyReason}` : ''}`}
            style={{
              position: 'absolute',
              left: group.centerScreen.x,
              top: group.centerScreen.y,
              transform: 'translate(-50%, -50%)',
              zIndex: 1150,
              border: '1px solid rgba(245,158,11,0.75)',
              background: busy
                ? 'linear-gradient(180deg, rgba(254,243,199,0.92), rgba(253,230,138,0.86))'
                : 'linear-gradient(180deg, rgba(255,251,235,0.96), rgba(254,240,138,0.9))',
              color: '#78350f',
              borderRadius: 999,
              padding: '6px 12px',
              fontSize: 12,
              fontWeight: 700,
              cursor: busy ? 'wait' : 'pointer',
              boxShadow: '0 8px 18px rgba(217,119,6,0.18)',
              pointerEvents: 'auto',
              whiteSpace: 'nowrap',
            }}
          >
            {busy ? 'Promoting…' : 'Promote to Block'}
          </button>
        )
      })}
      <BottomPanel
        hint={hint}
        plannerNextStepHint={plannerNextStepHint}
        onHintChange={setHint}
        onSubmit={askAI}
        mode={mode}
        onModeCycle={cycleMode}
        showAutoMaintain={mode === 'full'}
        autoMaintainEnabled={autoMaintain}
        autoMaintainPending={autoMaintainPending}
        onToggleAutoMaintain={handleAutoMaintainToggle}
        graphInspectorActive={graphInspectorVisible}
        viewportHeight={size.height}
        graphBlocksDetailed={graphBlockCards}
        onFragmentFocus={focusOnFragment}
        onBlockFocus={focusOnBlock}
        onFragmentHover={handleFragmentHover}
        onBlockHover={handleBlockHover}
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
              Once Auto Maintain is enabled, you will see real-time knowledge graph updates.
            </div>
          )}
          {autoMaintain && (
            <>
              <div style={{ fontSize: 12, color: '#a5b4fc', marginBottom: 8 }}>
                Blocks: {graphSnapshot?.blocks?.length ?? 0} · Fragments: {graphSnapshot?.fragments?.length ?? 0}
                <div
                  style={{
                    marginTop: 10,
                    marginBottom: 12,
                    border: '1px solid rgba(34,197,94,0.28)',
                    borderRadius: 12,
                    padding: 10,
                    background: graphBlockSelectionMode
                      ? 'rgba(34,197,94,0.12)'
                      : 'rgba(15,23,42,0.24)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: graphBlockSelectionMode ? '#bbf7d0' : '#dcfce7' }}>
                      Manual Block Selection
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setGraphBlockSelectionMode((prev) => !prev)
                        setGraphSelectionRectScreen(null)
                        graphSelectionDragRef.current = null
                      }}
                      style={{
                        border: `1px solid ${graphBlockSelectionMode ? 'rgba(34,197,94,0.55)' : 'rgba(148,163,184,0.4)'}`,
                        background: graphBlockSelectionMode ? 'rgba(22,163,74,0.18)' : 'rgba(255,255,255,0.06)',
                        color: graphBlockSelectionMode ? '#dcfce7' : '#e2e8f0',
                        borderRadius: 999,
                        padding: '5px 10px',
                        fontSize: 11,
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      {graphBlockSelectionMode ? 'Exit Select' : 'Box Select'}
                    </button>
                  </div>
                  <div style={{ fontSize: 11, color: '#cbd5e1', marginTop: 6, lineHeight: 1.45 }}>
                    Drag a rectangle on canvas to select fragments, then create a new block or move them into an existing block.
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: '#e2e8f0' }}>
                    Selected: <strong>{graphSelectedFragmentIds.length}</strong> fragment{graphSelectedFragmentIds.length === 1 ? '' : 's'}
                  </div>
                  {graphSelectedFragments.length > 0 && (
                    <div style={{ marginTop: 6, fontSize: 10, color: '#bbf7d0', lineHeight: 1.4 }}>
                      {graphSelectedFragments.slice(0, 4).map((frag) => (
                        <div key={`sel-frag-preview-${frag.id}`}>
                          {frag.type} · {frag.id}
                          {frag.blockLabel ? ` · ${frag.blockLabel}` : frag.blockId ? ` · ${frag.blockId}` : ' · (unassigned)'}
                        </div>
                      ))}
                      {graphSelectedFragments.length > 4 && (
                        <div>… and {graphSelectedFragments.length - 4} more</div>
                      )}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={() => void applyGraphSelectionBlockAction('create_block')}
                      disabled={graphSelectedFragmentIds.length === 0 || graphSelectionActionPending !== null}
                      style={{
                        border: '1px solid rgba(34,197,94,0.45)',
                        background: graphSelectionActionPending === 'create_block'
                          ? 'rgba(34,197,94,0.12)'
                          : 'rgba(34,197,94,0.18)',
                        color: '#dcfce7',
                        borderRadius: 999,
                        padding: '6px 10px',
                        fontSize: 11,
                        fontWeight: 700,
                        cursor: graphSelectedFragmentIds.length === 0 || graphSelectionActionPending !== null ? 'not-allowed' : 'pointer',
                        opacity: graphSelectedFragmentIds.length === 0 ? 0.6 : 1,
                      }}
                    >
                      {graphSelectionActionPending === 'create_block' ? 'Creating…' : 'Create Block'}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setGraphSelectedFragmentIds([])
                        setGraphSelectionRectScreen(null)
                      }}
                      disabled={graphSelectionActionPending !== null}
                      style={{
                        border: '1px solid rgba(148,163,184,0.35)',
                        background: 'rgba(255,255,255,0.04)',
                        color: '#e2e8f0',
                        borderRadius: 999,
                        padding: '6px 10px',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: graphSelectionActionPending !== null ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                    <select
                      value={graphSelectionTargetBlockId}
                      onChange={(e) => setGraphSelectionTargetBlockId(e.target.value)}
                      style={{
                        flex: 1,
                        minWidth: 0,
                        borderRadius: 8,
                        border: '1px solid rgba(148,163,184,0.35)',
                        background: 'rgba(15,23,42,0.45)',
                        color: '#e2e8f0',
                        padding: '6px 8px',
                        fontSize: 11,
                      }}
                    >
                      {(graphSnapshot?.blocks ?? []).map((block) => (
                        <option key={`assign-block-opt-${block.blockId}`} value={block.blockId}>
                          {block.label || block.blockId}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => void applyGraphSelectionBlockAction('assign_block')}
                      disabled={
                        graphSelectedFragmentIds.length === 0
                        || !(graphSelectionTargetBlockId || '').trim()
                        || graphSelectionActionPending !== null
                      }
                      style={{
                        border: '1px solid rgba(59,130,246,0.45)',
                        background: graphSelectionActionPending === 'assign_block'
                          ? 'rgba(59,130,246,0.12)'
                          : 'rgba(59,130,246,0.18)',
                        color: '#dbeafe',
                        borderRadius: 999,
                        padding: '6px 10px',
                        fontSize: 11,
                        fontWeight: 700,
                        cursor: graphSelectedFragmentIds.length === 0 || graphSelectionActionPending !== null ? 'not-allowed' : 'pointer',
                        whiteSpace: 'nowrap',
                        opacity: graphSelectedFragmentIds.length === 0 ? 0.6 : 1,
                      }}
                    >
                      {graphSelectionActionPending === 'assign_block' ? 'Moving…' : 'Move to Block'}
                    </button>
                  </div>
                </div>
                {graphSnapshot?.visionPendingGroups && graphSnapshot.visionPendingGroups.length > 0 && (
                  <div style={{ marginTop: 10, marginBottom: 12 }}>
                    <div style={{ fontSize: 12, color: '#fdba74', marginBottom: 6 }}>
                      Vision Pending Groups ({graphSnapshot.visionPendingGroups.length})
                    </div>
                    {graphSnapshot.visionPendingGroups.map((group) => (
                      <div
                        key={`vision-${group.groupId}`}
                        style={{
                          border: '1px dashed rgba(245,158,11,0.55)',
                          borderRadius: 10,
                          padding: '8px 10px',
                          marginBottom: 6,
                          background: 'rgba(245,158,11,0.08)',
                        }}
                      >
                        <div style={{ fontSize: 12, fontWeight: 600, color: '#fde68a' }}>
                          {group.groupId} · strokes {group.count}
                        </div>
                        <div style={{ fontSize: 11, color: '#fcd34d', marginTop: 2 }}>
                          {group.eligible ? 'eligible for manual promote' : 'collecting more strokes'}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {graphSnapshot?.groups && graphSnapshot.groups.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 12, color: '#fca5a5', marginBottom: 6 }}>
                      Semantic Groups ({graphSnapshot.groups.length})
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
                          {group.members.length > 4 ? ' ...­' : ''}
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
                          {promoteGroupPending === group.groupId ? 'Promoting...' : 'Promote to Block'}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {(!graphSnapshot || graphSnapshot.blocks.length === 0) ? (
                <div style={{ fontSize: 13, color: '#cbd5f5' }}>
                  No named semantic blocks are currently available. Try adding text or wait for the LLM to be aggregated.
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
                      {block.summary || '(No summary available.)'}
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
  relationships: Array<{ target: string; type: string; score?: number | null }>
  updatedAt?: string
  position?: [number, number, number, number] | null
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

type GraphVisionPendingGroup = {
  groupId: string
  bbox?: [number, number, number, number] | null
  count: number
  strokeIds?: string[]
  readyReason?: string | null
  createdAt?: string
  updatedAt?: string
  eligible?: boolean
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
  center: { x: number; y: number } | null
  relationships: Array<{ target: string; type: string; score?: number }>
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
  visionPendingGroups?: GraphVisionPendingGroup[]
}

type ImportEntry = { draft: ShapeDraft; stroke: AIStrokeV11 }

const SHAPE_KIND_VALUES: ShapeDraft['kind'][] = ['pen', 'line', 'rect', 'ellipse', 'poly', 'polyline', 'erase', 'text', 'edit']

function buildImportEntriesFromStrokes(strokes: AIStrokeV11[]): ImportEntry[] {
  if (!strokes.length) return []
  const payload = { version: 1, intent: 'import', strokes }
  const validated = validateAIStrokePayload(payload)
  if (!validated.ok || !validated.payload) {
    throw new Error((validated.errors && validated.errors.join('; ')) || 'invalid stroke payload')
  }
  const normalized = normalizeAIStrokePayload(validated.payload)
  const drafts = planDrafts(normalized)
  const strokeMap = new Map(normalized.strokes.map((s) => [s.id, s]))
  return drafts
    .map((draft) => {
      const stroke = strokeMap.get(draft.id) ?? draftToAIStroke(draft)
      if (!stroke) return null
      return { draft, stroke }
    })
    .filter((entry): entry is ImportEntry => Boolean(entry))
}

function extractStrokesFromImport(data: any): AIStrokeV11[] {
  if (!data) return []
  if (Array.isArray(data)) {
    const asStrokes = coerceStrokeArray(data)
    if (asStrokes.length) return asStrokes
    const fromShapes = shapeArrayToStrokes(data)
    if (fromShapes.length) return fromShapes
  }
  if (Array.isArray(data?.strokes)) {
    const asStrokes = coerceStrokeArray(data.strokes)
    if (asStrokes.length) return asStrokes
  }
  if (Array.isArray(data?.payload?.strokes)) {
    const asStrokes = coerceStrokeArray(data.payload.strokes)
    if (asStrokes.length) return asStrokes
  }
  if (Array.isArray(data?.shapes)) {
    const fromShapes = shapeArrayToStrokes(data.shapes)
    if (fromShapes.length) return fromShapes
  }
  return []
}

function shapeArrayToStrokes(rawShapes: any[]): AIStrokeV11[] {
  if (!Array.isArray(rawShapes)) return []
  const drafts = rawShapes
    .map((raw, index) => sanitizeShapeDraft(raw, index))
    .filter((draft): draft is ShapeDraft => Boolean(draft))
  return drafts
    .map((draft) => draftToAIStroke(draft))
    .filter((stroke): stroke is AIStrokeV11 => Boolean(stroke))
}

function sanitizeShapeDraft(raw: any, index: number): ShapeDraft | null {
  if (!raw || typeof raw !== 'object') return null
  const id = String(raw.id ?? `import_shape_${index}`)
  const rawKind = typeof raw.kind === 'string' ? raw.kind : ''
  const kind = (SHAPE_KIND_VALUES.includes(rawKind as ShapeDraft['kind']) ? rawKind : 'pen') as ShapeDraft['kind']
  const draft: ShapeDraft = {
    id,
    kind,
    x: Number(raw.x) || 0,
    y: Number(raw.y) || 0,
  }
  if (Number.isFinite(raw.w)) draft.w = Number(raw.w)
  if (Number.isFinite(raw.h)) draft.h = Number(raw.h)
  if (Array.isArray(raw.points)) {
    draft.points = raw.points
      .map((pt: any) => {
        if (Array.isArray(pt)) {
          return { x: Number(pt[0]) || 0, y: Number(pt[1]) || 0, pressure: pt.length > 2 ? Number(pt[2]) || undefined : undefined }
        }
        if (pt && typeof pt === 'object') {
          return { x: Number(pt.x ?? pt[0] ?? 0) || 0, y: Number(pt.y ?? pt[1] ?? 0) || 0, pressure: pt.pressure !== undefined ? Number(pt.pressure) : undefined }
        }
        return null
      })
      .filter(
        (pt: { x: number; y: number; pressure?: number } | null): pt is { x: number; y: number; pressure?: number } =>
          Boolean(pt)
      )
  }
  if (typeof raw.text === 'string') draft.text = raw.text
  if (typeof raw.summary === 'string') draft.summary = raw.summary
  if (raw.style && typeof raw.style === 'object') {
    draft.style = {
      size: (raw.style.size ?? 'm') as 's'|'m'|'l'|'xl',
      color: (raw.style.color ?? 'black') as ColorName,
      opacity: Number.isFinite(raw.style.opacity) ? Number(raw.style.opacity) : 1,
    }
  }
  if (raw.meta && typeof raw.meta === 'object') {
    draft.meta = { ...raw.meta }
  }
  return draft
}

function coerceStrokeArray(items: any[]): AIStrokeV11[] {
  if (!Array.isArray(items)) return []
  return items
    .map((raw, index) => sanitizeStroke(raw, index))
    .filter((stroke): stroke is AIStrokeV11 => Boolean(stroke))
}

function sanitizeStroke(raw: any, index: number): AIStrokeV11 | null {
  if (!raw || typeof raw !== 'object') return null
  const tool = String(raw.tool ?? 'pen')
  const rawPoints = Array.isArray(raw.points) ? raw.points : []
  const points = rawPoints
    .map((pt: any) => {
      if (Array.isArray(pt)) {
        return [
          Number(pt[0]) || 0,
          Number(pt[1]) || 0,
          pt.length > 2 ? Number(pt[2]) || undefined : undefined,
          pt.length > 3 ? Number(pt[3]) || undefined : undefined,
        ] as [number, number, number?, number?]
      }
      if (pt && typeof pt === 'object') {
        return [
          Number(pt.x ?? pt[0] ?? 0) || 0,
          Number(pt.y ?? pt[1] ?? 0) || 0,
          pt.t !== undefined ? Number(pt.t) : pt.pressure !== undefined ? Number(pt.pressure) : undefined,
          pt.pressure !== undefined ? Number(pt.pressure) : pt.p !== undefined ? Number(pt.p) : undefined,
        ] as [number, number, number?, number?]
      }
      return null
    })
    .filter(
      (pt: [number, number, number?, number?] | null): pt is [number, number, number?, number?] => Array.isArray(pt)
    )
  if (points.length === 0 && tool !== 'text') return null
  const styleRaw = raw.style && typeof raw.style === 'object' ? raw.style : {}
  const stroke: AIStrokeV11 = {
    id: String(raw.id ?? `import_stroke_${index}`),
    tool: tool as AIStrokeV11['tool'],
    points: points.length > 0 ? points : [
      [Number(raw.x ?? 0), Number(raw.y ?? 0)],
      [Number((raw.x ?? 0) + (raw.w ?? 0)), Number((raw.y ?? 0) + (raw.h ?? 0))],
    ],
    style: {
      size: (styleRaw.size ?? 'm') as 's'|'m'|'l'|'xl',
      color: (styleRaw.color ?? 'black') as ColorName,
      opacity: Number.isFinite(styleRaw.opacity) ? Number(styleRaw.opacity) : 1,
    },
    meta: raw.meta && typeof raw.meta === 'object' ? { ...raw.meta } : undefined,
  }
  if (tool === 'text' && (!stroke.points || stroke.points.length < 2)) {
    const x0 = Number(raw.x ?? 0)
    const y0 = Number(raw.y ?? 0)
    const w = Number(raw.w ?? 240) || 240
    const h = Number(raw.h ?? 160) || 160
    stroke.points = [
      [x0, y0],
      [x0 + w, y0 + h],
    ]
  }
  return stroke
}
