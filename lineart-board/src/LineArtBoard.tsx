import React, { useCallback, useMemo, useRef, useState } from 'react'
import { Stage, Layer, Group, Line as KLine, Rect as KRect, Ellipse as KEllipse, Text as KText } from 'react-konva'
import type { AIStrokePayload, AIStrokeV11, ColorName, PromptMode } from './ai/types'
import { normalizeAIStrokePayload, validateAIStrokePayload } from './ai/normalize'
import type { ShapeDraft } from './ai/plan'
import { planDrafts } from './ai/plan'
import { chaikin, resampleEvenly, geomMaxDeviationFromChord, mergeCollinear, draftToAIStroke } from './ai/draw'
import { TopToolbar, SidePanel, BottomPanel, type AIFeedEntry } from './LineArtUI'

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

// Preview entries keep drafts grouped by payload id
type PreviewEntry = { payloadId: string; drafts: ShapeDraft[] }

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
  const [hint, setHint] = useState<string>('Try to help the user draw line art.')
  // AI generation scale caps point count and informs upload density
  const [aiScale, setAiScale] = useState<number>(16) // adjustable 4-64, defaults to 16
  // Live drawing state with raw float coordinates (world space)
  const [isDrawing, setIsDrawing] = useState(false)
  const [rawPoints, setRawPoints] = useState<number[]>([])  // [x0,y0,x1,y1,...] world coordinates

  // Stack of human strokes for erasing and undo/redo
  type DrawStackEntry = { ai: AIStrokeV11; draft: ShapeDraft }
  const [drawStack, setDrawStack] = useState<DrawStackEntry[]>([])
  // -------- Tool modes: pen / eraser / ellipse / hand --------
  const [toolMode, setToolMode] = useState<'pen' | 'eraser' | 'ellipse' | 'hand'>('pen')
  const [eraserRadius, setEraserRadius] = useState<number>(14) // pixels
  const [boxDraft, setBoxDraft] = useState<ShapeDraft | null>(null)
  // Visual cursor for the eraser radius
  const [eraserCursor, setEraserCursor] = useState<{x:number;y:number}|null>(null)
  // Only push history once per erase gesture (pointer down -> up)
  const eraseGestureStarted = useRef(false)
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
  const autoTimerRef = useRef<number | ReturnType<typeof setTimeout> | null>(null)
  const autoTickerRef = useRef<number | ReturnType<typeof setInterval> | null>(null)

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
  const cycleMode = useCallback(() => {
    setMode((current) => (current === 'light' ? 'full' : current === 'full' ? 'vision' : 'light'))
  }, [setMode])
  const canUndo = past.length > 0
  const canRedo = future.length > 0
  const stageCursor = toolMode === 'hand' ? (isPanning ? 'grabbing' : 'grab') : 'default'

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
      // Record feed entries
      const items = (obj.strokes||[]).map(s=>({ id: s.id, desc: (s.meta as any)?.desc }))
      setAiFeed(prev=>([{ payloadId: 'local_'+Date.now().toString(36), time: Date.now(), items }, ...prev].slice(0,50)))
      const v = validateAIStrokePayload(obj)
      if (!v.ok || !v.payload) { alert('Invalid payload: ' + v.errors.join('; ')); return }
      const norm = normalizeAIStrokePayload(v.payload)
      const drafts = planDrafts(norm)
      setPreviews(prev => ({ ...prev, [norm.payloadId]: { payloadId: norm.payloadId, drafts } }))
      setCurrentPayloadId(norm.payloadId)
      // Stop auto-complete countdown once preview becomes active
      clearAutoTimer()
      alert(`Preview created: ${drafts.length} shapes\nPayloadId: ${norm.payloadId}`)
    } catch { alert('Invalid JSON in localStorage') }
  }, [])

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

  // Accept: merge preview drafts into committed shapes
  const acceptAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    const entry = previews[currentPayloadId]
    if (!entry) { alert('Preview not found'); return }
    // Capture snapshot for undo
    pushHistory()
    // 1) Append drafts to committed shapes
    setShapes(prev => [...prev, ...entry.drafts])
    // 2) Mirror drafts into drawStack so eraser/undo stay coherent
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
    // Treat acceptance as user activity to restart countdown
    noteUserAction({ forceStart: true })
  }, [currentPayloadId, previews, pushHistory, noteUserAction])

  // Dismiss: remove preview without committing
  const dismissAI = useCallback(() => {
    if (!currentPayloadId) { alert('No current payloadId'); return }
    setPreviews((prev) => {
      const { [currentPayloadId]: _omit, ...rest } = prev
      return rest
    })
    setCurrentPayloadId(null)
    // Reset countdown so the next gesture restarts auto-complete if enabled
    clearAutoTimer()
  }, [currentPayloadId])
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
  const DraftNode: React.FC<{ d: ShapeDraft; preview?: boolean }> = ({ d, preview }) => {
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
        const fillColor = colorToStroke(d.style?.color ?? 'black') // 我们已有 color->#hex 的函数
        const fontSize = (d.meta?.fontSize ?? d.meta?.fontsize ?? d.meta?.font_size ?? 16) as number
        const fontFamily = (d.meta?.fontFamily ?? 'sans-serif') as string
        const fontStyle = (d.meta?.fontWeight === 'bold' || d.meta?.fontWeight === '700')
          ? 'bold'
          : 'normal'
        const boxW = d.w ?? 160
        const boxH = d.h ?? 80
        return (
          <Group listening={false}>
            {/* 可选：画一个淡淡的边框，方便可视化文本框范围 */}
            <KRect
              x={d.x}
              y={d.y}
              width={boxW}
              height={boxH}
              stroke={preview ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.4)'}
              strokeWidth={1 / view.scale}
              dash={preview ? [4,4] : undefined}
              opacity={0.6}
            />
            <KText
              x={d.x}
              y={d.y}
              width={boxW}
              height={boxH}
              text={d.text ?? ''}
              fontFamily={fontFamily}
              fontSize={fontSize / view.scale /* 或者不除，看你希望缩放是否影响字号 */}
              fontStyle={fontStyle}
              fill={fillColor}
              opacity={preview ? Math.min(0.35, d.style?.opacity ?? 1) : (d.style?.opacity ?? 1)}
              align="left"
              verticalAlign="top"
              listening={false}
              wrap="word"
            />
          </Group>
        )
      }
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
    if (toolMode === 'hand') return
    const pos = e.target.getStage()?.getPointerPosition()
    if (!pos) return
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
    } else { // eraser
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })
      eraseGestureStarted.current = false // Start a new eraser gesture
      eraseWholeStrokesAt(sx, sy, eraserRadius) // Attempt initial erase immediately
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
    } else {
      // Eraser keeps snapping for more reliable hits
      const wpt = screenToWorld(pos.x, pos.y)
      const [sx, sy] = snapPoint(wpt.x, wpt.y)
      setEraserCursor({ x: sx, y: sy })           // Show eraser cursor indicator
      eraseWholeStrokesAt(sx, sy, eraserRadius)    // Continue erasing whole strokes
    }
  }, [isDrawing, boxDraft, snap, snapPoint, toolMode, eraserRadius, eraseWholeStrokesAt, screenToWorld])

  const onMouseUp = useCallback(() => {
    if (toolMode === 'hand') return
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
    } else {
      // Finish eraser gesture
      setEraserCursor(null)
      eraseGestureStarted.current = false
    }
  }, [isDrawing, rawPoints, boxDraft, snap, snapPoint, toolMode, curveTurns, currentBrush, pushHistory, setShapes, setDrawStack])

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
      />

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
          {shapes.map(d => <DraftNode key={'s:'+d.id} d={d} />)}
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
          {toolMode==='ellipse' && isDrawing && boxDraft && (
            <DraftNode d={boxDraft} />
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

        {/* AI previews */}
        <Layer>
          {Object.values(previews).map(entry => (
            <Group key={'p:'+entry.payloadId} listening={false} name="ai-candidate" id={entry.payloadId}>
              {entry.drafts.map(d => <DraftNode key={'pd:'+entry.payloadId+':'+d.id} d={d} preview />)}
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
      />

    </div>
  )
}














