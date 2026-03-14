import type { ShapeDraft } from './ai/plan'
import type { PromptMode } from './ai/types'

export type BBox = [number, number, number, number]

export type ExperimentShapeSnapshot = {
  id: string
  kind: ShapeDraft['kind']
  bbox: BBox | null
  text: string
  summary: string
  pointCount: number
  sampledPoints: Array<[number, number]>
}

export type ExperimentRequestRound = {
  requestId: string
  phaseId: string
  sentAt: number
  respondedAt?: number | null
  sessionId?: string | null
  requestMode: PromptMode
  status: 'sent' | 'completed' | 'failed'
  promptTokens: number
  completionTokens: number
  totalTokens: number
  planTargetBlockIds: string[]
  activeBlockIds: string[]
  mainBlockId?: string | null
  planAction?: string | null
  previewPayloadId?: string | null
  error?: string | null
}

export type ExperimentPreviewRecord = {
  payloadId: string
  requestId?: string | null
  phaseId: string
  at: number
}

export type ExperimentDismissRecord = {
  payloadId: string
  requestId?: string | null
  phaseId: string
  at: number
  reason?: string | null
}

export type ExperimentAcceptedSuggestion = {
  acceptId: string
  payloadId: string
  requestId?: string | null
  phaseId: string
  acceptedAt: number
  acceptedShapeIds: string[]
  userChangeTrackedShapeIds: string[]
  baselineShapes: Record<string, ExperimentShapeSnapshot>
  usableUnits: number
  targetBBox: BBox | null
  activeBlockIds: string[]
  planTargetBlockIds: string[]
  activeBlockAligned: boolean
  maxPostAcceptEdit?: number
  maxChangedTextChars?: number
  changedEver?: boolean
}

export type ExperimentEvent = {
  type:
    | 'experiment_started'
    | 'experiment_ended'
    | 'phase_changed'
    | 'ai_request_sent'
    | 'ai_request_completed'
    | 'ai_request_failed'
    | 'ai_preview'
    | 'ai_accept'
    | 'ai_dismiss'
  at: number
  data?: Record<string, unknown>
}

export type ExperimentRun = {
  runId: string
  startedAt: number
  endedAt?: number | null
  initialSessionId?: string | null
  currentPhaseId: string
  editThreshold: number
  requestRounds: ExperimentRequestRound[]
  previews: ExperimentPreviewRecord[]
  dismisses: ExperimentDismissRecord[]
  acceptedSuggestions: ExperimentAcceptedSuggestion[]
  events: ExperimentEvent[]
}

export type ExperimentUsageUpdate = {
  promptTokens: number
  completionTokens: number
  totalTokens: number
  planTargetBlockIds: string[]
  activeBlockIds: string[]
  mainBlockId?: string | null
  planAction?: string | null
}

export type ExperimentAcceptedDerived = ExperimentAcceptedSuggestion & {
  postAcceptEdit: number
  acceptedTextChars: number
  changedTextChars: number
  changedTextRatio: number | null
  straightUse: boolean
  userChanged: boolean
}

export type ExperimentPhaseEfficiency = {
  phaseId: string
  invoke: number
  straightUseRate: number | null
  acceptedOutputPer1kToken: number | null
  promptTokens: number
  acceptedUsableUnits: number
  acceptedCount: number
}

export type ExperimentSummary = {
  aiInvokeTimes: number
  previewCount: number
  acceptCount: number
  dismissCount: number
  completedRoundCount: number
  totalPromptTokens: number
  acceptedUsableUnits: number
  acceptedTextChars: number
  changedTextChars: number
  suggestionAcceptanceRate: number | null
  dismissRate: number | null
  straightUseRate: number | null
  userChangedRate: number | null
  promptTokensPerRound: number | null
  acceptedUsableContentPer1kTokens: number | null
  activeBlockAlignmentRate: number | null
  phaseSpecificEfficiency: ExperimentPhaseEfficiency[]
  acceptedSuggestions: ExperimentAcceptedDerived[]
}

const TEXT_UNIT_CHARS = 20

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

const safeRatio = (numerator: number, denominator: number) => (
  denominator > 0 ? numerator / denominator : null
)

const nowId = (prefix: string) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

const normalizeText = (value: string | null | undefined) => String(value ?? '').replace(/\s+/g, ' ').trim()

const snapshotText = (snapshot: ExperimentShapeSnapshot | null | undefined) => normalizeText(snapshot?.text)

const computeLevenshteinDistance = (sourceRaw: string, targetRaw: string): number => {
  if (sourceRaw === targetRaw) return 0
  const source = sourceRaw.length <= targetRaw.length ? sourceRaw : targetRaw
  const target = sourceRaw.length <= targetRaw.length ? targetRaw : sourceRaw
  if (!source.length) return target.length
  if (!target.length) return source.length
  let previous = Array.from({ length: source.length + 1 }, (_, index) => index)
  let current = new Array<number>(source.length + 1)
  for (let targetIndex = 1; targetIndex <= target.length; targetIndex += 1) {
    current[0] = targetIndex
    const targetChar = target.charCodeAt(targetIndex - 1)
    for (let sourceIndex = 1; sourceIndex <= source.length; sourceIndex += 1) {
      const cost = source.charCodeAt(sourceIndex - 1) === targetChar ? 0 : 1
      current[sourceIndex] = Math.min(
        current[sourceIndex - 1] + 1,
        previous[sourceIndex] + 1,
        previous[sourceIndex - 1] + cost,
      )
    }
    const swap = previous
    previous = current
    current = swap
  }
  return previous[source.length]
}

const computeChangedTextChars = (
  baseline: ExperimentShapeSnapshot | null | undefined,
  current: ExperimentShapeSnapshot | null | undefined,
): number => {
  const baselineText = snapshotText(baseline)
  if (!baselineText) return 0
  const currentText = snapshotText(current)
  return Math.min(baselineText.length, computeLevenshteinDistance(baselineText, currentText))
}

const computeAcceptedTextStats = (
  record: ExperimentAcceptedSuggestion,
  shapeMap: Map<string, ExperimentShapeSnapshot> | null,
) => {
  const trackedShapeIds = record.userChangeTrackedShapeIds?.length
    ? record.userChangeTrackedShapeIds
    : record.acceptedShapeIds
  let acceptedTextChars = 0
  let liveChangedTextChars = 0
  for (const shapeId of trackedShapeIds) {
    const baseline = record.baselineShapes[shapeId]
    const baselineText = snapshotText(baseline)
    if (!baselineText) continue
    acceptedTextChars += baselineText.length
    if (shapeMap) {
      liveChangedTextChars += computeChangedTextChars(baseline, shapeMap.get(shapeId))
    }
  }
  return {
    acceptedTextChars,
    liveChangedTextChars,
  }
}

const sampleAbsolutePoints = (shape: ShapeDraft, sampleCount = 10): Array<[number, number]> => {
  const points = shape.points ?? []
  if (!points.length) return []
  if (points.length <= sampleCount) {
    return points.map((point) => [round3(shape.x + point.x), round3(shape.y + point.y)])
  }
  const sampled: Array<[number, number]> = []
  for (let index = 0; index < sampleCount; index += 1) {
    const ratio = sampleCount === 1 ? 0 : index / (sampleCount - 1)
    const pointIndex = Math.min(points.length - 1, Math.round(ratio * (points.length - 1)))
    const point = points[pointIndex]
    sampled.push([round3(shape.x + point.x), round3(shape.y + point.y)])
  }
  return sampled
}

const round3 = (value: number) => {
  if (!Number.isFinite(value)) return 0
  return Math.round(value * 1000) / 1000
}

export const computeShapeBBox = (shape: ShapeDraft | null | undefined): BBox | null => {
  if (!shape) return null
  const baseX = Number(shape.x) || 0
  const baseY = Number(shape.y) || 0
  if (Number.isFinite(shape.w) && Number.isFinite(shape.h)) {
    const width = Math.max(1, Number(shape.w))
    const height = Math.max(1, Number(shape.h))
    return [baseX, baseY, baseX + width, baseY + height]
  }
  const points = shape.points ?? []
  if (!points.length) return null
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const point of points) {
    const px = baseX + (Number(point.x) || 0)
    const py = baseY + (Number(point.y) || 0)
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

export const mergeBBox = (a: BBox | null, b: BBox | null): BBox | null => {
  if (!a) return b ? [...b] as BBox : null
  if (!b) return [...a] as BBox
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[2], b[2]),
    Math.max(a[3], b[3]),
  ]
}

export const bboxIntersects = (a: BBox | null, b: BBox | null): boolean => {
  if (!a || !b) return false
  const x0 = Math.max(a[0], b[0])
  const y0 = Math.max(a[1], b[1])
  const x1 = Math.min(a[2], b[2])
  const y1 = Math.min(a[3], b[3])
  return x1 > x0 && y1 > y0
}

export const shapeToSnapshot = (shape: ShapeDraft): ExperimentShapeSnapshot => ({
  id: String(shape.id),
  kind: shape.kind,
  bbox: computeShapeBBox(shape),
  text: normalizeText(shape.text),
  summary: normalizeText(shape.summary),
  pointCount: (shape.points ?? []).length,
  sampledPoints: sampleAbsolutePoints(shape),
})

const bboxDistance = (a: BBox | null, b: BBox | null): number => {
  if (!a && !b) return 0
  if (!a || !b) return 1
  const aw = Math.max(1, a[2] - a[0])
  const ah = Math.max(1, a[3] - a[1])
  const bw = Math.max(1, b[2] - b[0])
  const bh = Math.max(1, b[3] - b[1])
  const acx = (a[0] + a[2]) / 2
  const acy = (a[1] + a[3]) / 2
  const bcx = (b[0] + b[2]) / 2
  const bcy = (b[1] + b[3]) / 2
  const scale = Math.max(aw, ah, bw, bh, 1)
  const center = clamp(Math.hypot(acx - bcx, acy - bcy) / scale, 0, 1)
  const size = clamp((Math.abs(aw - bw) + Math.abs(ah - bh)) / Math.max(aw + ah, bw + bh, 1), 0, 1)
  return clamp(center * 0.6 + size * 0.4, 0, 1)
}

const pointDistance = (a: ExperimentShapeSnapshot, b: ExperimentShapeSnapshot): number => {
  const countMax = Math.max(a.pointCount, b.pointCount)
  if (countMax <= 0) return 0
  const scale = Math.max(
    Math.abs((a.bbox?.[2] ?? 0) - (a.bbox?.[0] ?? 0)),
    Math.abs((a.bbox?.[3] ?? 0) - (a.bbox?.[1] ?? 0)),
    Math.abs((b.bbox?.[2] ?? 0) - (b.bbox?.[0] ?? 0)),
    Math.abs((b.bbox?.[3] ?? 0) - (b.bbox?.[1] ?? 0)),
    1,
  )
  const sampleCount = Math.min(a.sampledPoints.length, b.sampledPoints.length)
  let sampleDelta = 0
  if (sampleCount > 0) {
    for (let index = 0; index < sampleCount; index += 1) {
      const pa = a.sampledPoints[index]
      const pb = b.sampledPoints[index]
      sampleDelta += Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / scale
    }
    sampleDelta = clamp(sampleDelta / sampleCount, 0, 1)
  }
  const countDelta = clamp(Math.abs(a.pointCount - b.pointCount) / countMax, 0, 1)
  return clamp(sampleDelta * 0.7 + countDelta * 0.3, 0, 1)
}

const textDistance = (a: ExperimentShapeSnapshot, b: ExperimentShapeSnapshot): number => {
  const ta = normalizeText(a.text || a.summary)
  const tb = normalizeText(b.text || b.summary)
  if (!ta && !tb) return 0
  if (ta === tb) return 0
  const maxLen = Math.max(ta.length, tb.length, 1)
  const minLen = Math.min(ta.length, tb.length)
  let commonPrefix = 0
  while (commonPrefix < minLen && ta[commonPrefix] === tb[commonPrefix]) {
    commonPrefix += 1
  }
  const similarity = commonPrefix / maxLen
  const lenDelta = Math.abs(ta.length - tb.length) / maxLen
  return clamp(0.55 + lenDelta * 0.35 - similarity * 0.4, 0, 1)
}

export const computeSnapshotDistance = (
  baseline: ExperimentShapeSnapshot,
  current: ExperimentShapeSnapshot | null | undefined,
): number => {
  if (!current) return 1
  if (baseline.kind !== current.kind) return 1
  let weighted = 0
  let weightTotal = 0
  const bboxDelta = bboxDistance(baseline.bbox, current.bbox)
  weighted += bboxDelta * 0.5
  weightTotal += 0.5
  if (baseline.pointCount > 0 || current.pointCount > 0) {
    weighted += pointDistance(baseline, current) * 0.25
    weightTotal += 0.25
  }
  if (baseline.text || baseline.summary || current.text || current.summary) {
    weighted += textDistance(baseline, current) * 0.4
    weightTotal += 0.4
  }
  return clamp(weighted / Math.max(weightTotal, 1e-6), 0, 1)
}

export const estimateDraftUsableUnits = (drafts: ShapeDraft[]): number => {
  let units = 0
  for (const draft of drafts) {
    if (draft.kind === 'erase') continue
    if (draft.kind === 'text' || draft.kind === 'edit') {
      const compact = normalizeText(draft.text || draft.summary)
      units += compact ? Math.max(1, Math.ceil(compact.length / TEXT_UNIT_CHARS)) : 1
      continue
    }
    units += 1
  }
  return units
}

export const extractUsageUpdate = (raw: any): ExperimentUsageUpdate => {
  const promptTokens = Math.max(0, Number(raw?.prompt_tokens ?? raw?.promptTokens ?? 0) || 0)
  const completionTokens = Math.max(0, Number(raw?.completion_tokens ?? raw?.completionTokens ?? 0) || 0)
  const totalTokensRaw = Number(raw?.total_tokens ?? raw?.totalTokens ?? 0) || 0
  const totalTokens = Math.max(0, totalTokensRaw || (promptTokens + completionTokens))
  const toIdArray = (value: unknown) => (
    Array.isArray(value)
      ? value.map((item) => String(item ?? '').trim()).filter(Boolean)
      : []
  )
  return {
    promptTokens,
    completionTokens,
    totalTokens,
    planTargetBlockIds: toIdArray(raw?.plan_target_block_ids ?? raw?.planTargetBlockIds),
    activeBlockIds: toIdArray(raw?.active_block_ids ?? raw?.activeBlockIds),
    mainBlockId: raw?.main_block_id ? String(raw.main_block_id) : raw?.mainBlockId ? String(raw.mainBlockId) : null,
    planAction: raw?.plan_action ? String(raw.plan_action) : raw?.planAction ? String(raw.planAction) : null,
  }
}

export const createExperimentRun = (phaseId: string, sessionId?: string | null): ExperimentRun => {
  const normalizedPhase = normalizePhaseId(phaseId)
  const startedAt = Date.now()
  return {
    runId: nowId('run'),
    startedAt,
    endedAt: null,
    initialSessionId: sessionId ?? null,
    currentPhaseId: normalizedPhase,
    editThreshold: 0.2,
    requestRounds: [],
    previews: [],
    dismisses: [],
    acceptedSuggestions: [],
    events: [
      {
        type: 'experiment_started',
        at: startedAt,
        data: {
          phaseId: normalizedPhase,
          sessionId: sessionId ?? null,
        },
      },
    ],
  }
}

export const normalizePhaseId = (value: string | null | undefined): string => {
  const compact = String(value ?? '').trim()
  return compact || 'phase-1'
}

export const updateExperimentPhase = (run: ExperimentRun, phaseId: string): ExperimentRun => {
  const normalized = normalizePhaseId(phaseId)
  if (run.currentPhaseId === normalized) return run
  return {
    ...run,
    currentPhaseId: normalized,
    events: [
      ...run.events,
      {
        type: 'phase_changed',
        at: Date.now(),
        data: { phaseId: normalized },
      },
    ],
  }
}

export const addRequestSent = (
  run: ExperimentRun,
  params: {
    phaseId: string
    sessionId?: string | null
    requestMode: PromptMode
  },
): { run: ExperimentRun; requestId: string } => {
  const requestId = nowId('req')
  const at = Date.now()
  const round: ExperimentRequestRound = {
    requestId,
    phaseId: normalizePhaseId(params.phaseId),
    sentAt: at,
    respondedAt: null,
    sessionId: params.sessionId ?? null,
    requestMode: params.requestMode,
    status: 'sent',
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    planTargetBlockIds: [],
    activeBlockIds: [],
    mainBlockId: null,
    planAction: null,
    previewPayloadId: null,
    error: null,
  }
  return {
    requestId,
    run: {
      ...run,
      requestRounds: [...run.requestRounds, round],
      events: [
        ...run.events,
        {
          type: 'ai_request_sent',
          at,
          data: {
            requestId,
            phaseId: round.phaseId,
            requestMode: round.requestMode,
            sessionId: round.sessionId ?? null,
          },
        },
      ],
    },
  }
}

export const addRequestCompleted = (
  run: ExperimentRun,
  requestId: string,
  update: ExperimentUsageUpdate,
): ExperimentRun => {
  const at = Date.now()
  return {
    ...run,
    requestRounds: run.requestRounds.map((round) => (
      round.requestId !== requestId
        ? round
        : {
            ...round,
            status: 'completed',
            respondedAt: at,
            promptTokens: update.promptTokens,
            completionTokens: update.completionTokens,
            totalTokens: update.totalTokens,
            activeBlockIds: [...update.activeBlockIds],
            planTargetBlockIds: [...update.planTargetBlockIds],
            mainBlockId: update.mainBlockId ?? null,
            planAction: update.planAction ?? null,
          }
    )),
    events: [
      ...run.events,
      {
        type: 'ai_request_completed',
        at,
        data: {
          requestId,
          promptTokens: update.promptTokens,
          completionTokens: update.completionTokens,
          totalTokens: update.totalTokens,
          activeBlockIds: update.activeBlockIds,
          planTargetBlockIds: update.planTargetBlockIds,
        },
      },
    ],
  }
}

export const addRequestFailed = (
  run: ExperimentRun,
  requestId: string,
  error: string,
): ExperimentRun => {
  const at = Date.now()
  return {
    ...run,
    requestRounds: run.requestRounds.map((round) => (
      round.requestId !== requestId
        ? round
        : {
            ...round,
            status: 'failed',
            respondedAt: at,
            error,
          }
    )),
    events: [
      ...run.events,
      {
        type: 'ai_request_failed',
        at,
        data: {
          requestId,
          error,
        },
      },
    ],
  }
}

export const addPreviewRecord = (
  run: ExperimentRun,
  preview: {
    payloadId: string
    requestId?: string | null
    phaseId: string
  },
): ExperimentRun => {
  const at = Date.now()
  return {
    ...run,
    previews: [
      ...run.previews,
      {
        payloadId: preview.payloadId,
        requestId: preview.requestId ?? null,
        phaseId: normalizePhaseId(preview.phaseId),
        at,
      },
    ],
    requestRounds: run.requestRounds.map((round) => (
      round.requestId === preview.requestId
        ? { ...round, previewPayloadId: preview.payloadId }
        : round
    )),
    events: [
      ...run.events,
      {
        type: 'ai_preview',
        at,
        data: {
          payloadId: preview.payloadId,
          requestId: preview.requestId ?? null,
          phaseId: normalizePhaseId(preview.phaseId),
        },
      },
    ],
  }
}

export const addDismissRecord = (
  run: ExperimentRun,
  dismiss: {
    payloadId: string
    requestId?: string | null
    phaseId: string
    reason?: string | null
  },
): ExperimentRun => {
  const at = Date.now()
  return {
    ...run,
    dismisses: [
      ...run.dismisses,
      {
        payloadId: dismiss.payloadId,
        requestId: dismiss.requestId ?? null,
        phaseId: normalizePhaseId(dismiss.phaseId),
        at,
        reason: dismiss.reason ?? null,
      },
    ],
    events: [
      ...run.events,
      {
        type: 'ai_dismiss',
        at,
        data: {
          payloadId: dismiss.payloadId,
          requestId: dismiss.requestId ?? null,
          phaseId: normalizePhaseId(dismiss.phaseId),
          reason: dismiss.reason ?? null,
        },
      },
    ],
  }
}

export const addAcceptedSuggestion = (
  run: ExperimentRun,
  accepted: Omit<ExperimentAcceptedSuggestion, 'acceptId' | 'acceptedAt'>,
): ExperimentRun => {
  const acceptId = nowId('accept')
  const acceptedAt = Date.now()
  return {
    ...run,
    acceptedSuggestions: [
      ...run.acceptedSuggestions,
      {
        ...accepted,
        acceptId,
        acceptedAt,
        maxPostAcceptEdit: Math.max(0, Number(accepted.maxPostAcceptEdit ?? 0) || 0),
        maxChangedTextChars: Math.max(0, Number(accepted.maxChangedTextChars ?? 0) || 0),
        changedEver: Boolean(accepted.changedEver),
      },
    ],
    events: [
      ...run.events,
      {
        type: 'ai_accept',
        at: acceptedAt,
        data: {
          acceptId,
          payloadId: accepted.payloadId,
          requestId: accepted.requestId ?? null,
          phaseId: normalizePhaseId(accepted.phaseId),
          usableUnits: accepted.usableUnits,
          activeBlockAligned: accepted.activeBlockAligned,
        },
      },
    ],
  }
}

export const endExperimentRun = (run: ExperimentRun): ExperimentRun => {
  if (run.endedAt) return run
  const endedAt = Date.now()
  return {
    ...run,
    endedAt,
    events: [
      ...run.events,
      {
        type: 'experiment_ended',
        at: endedAt,
      },
    ],
  }
}

const deriveAcceptedSuggestion = (
  record: ExperimentAcceptedSuggestion,
  shapeMap: Map<string, ExperimentShapeSnapshot> | null,
  editThreshold: number,
): ExperimentAcceptedDerived => {
  const { acceptedTextChars, liveChangedTextChars } = computeAcceptedTextStats(record, shapeMap)
  let livePostAcceptEdit = Number(record.maxPostAcceptEdit ?? 0) || 0
  if (shapeMap) {
    const distances = record.acceptedShapeIds.map((shapeId) => {
      const baseline = record.baselineShapes[shapeId]
      if (!baseline) return 1
      return computeSnapshotDistance(baseline, shapeMap.get(shapeId))
    })
    livePostAcceptEdit = distances.length
      ? clamp(distances.reduce((sum, value) => sum + value, 0) / distances.length, 0, 1)
      : 1
  }
  const maxPostAcceptEdit = Math.max(livePostAcceptEdit, Number(record.maxPostAcceptEdit ?? 0) || 0)
  const changedTextChars = Math.max(liveChangedTextChars, Number(record.maxChangedTextChars ?? 0) || 0)
  const userChanged = Boolean(record.changedEver) || maxPostAcceptEdit > editThreshold
  return {
    ...record,
    postAcceptEdit: maxPostAcceptEdit,
    acceptedTextChars,
    changedTextChars,
    changedTextRatio: safeRatio(changedTextChars, acceptedTextChars),
    straightUse: !userChanged && maxPostAcceptEdit <= editThreshold,
    userChanged,
  }
}

export const refreshAcceptedSuggestionMutations = (
  run: ExperimentRun,
  currentShapes: ShapeDraft[],
): ExperimentRun => {
  if (!run.acceptedSuggestions.length) return run
  const shapeMap = new Map<string, ExperimentShapeSnapshot>()
  for (const shape of currentShapes) {
    shapeMap.set(String(shape.id), shapeToSnapshot(shape))
  }
  let changed = false
  const nextAccepted = run.acceptedSuggestions.map((record) => {
    const { liveChangedTextChars } = computeAcceptedTextStats(record, shapeMap)
    const distances = record.acceptedShapeIds.map((shapeId) => {
      const baseline = record.baselineShapes[shapeId]
      if (!baseline) return 1
      return computeSnapshotDistance(baseline, shapeMap.get(shapeId))
    })
    const currentPostAcceptEdit = distances.length
      ? clamp(distances.reduce((sum, value) => sum + value, 0) / distances.length, 0, 1)
      : 1
    const nextMax = Math.max(Number(record.maxPostAcceptEdit ?? 0) || 0, currentPostAcceptEdit)
    const nextMaxChangedTextChars = Math.max(Number(record.maxChangedTextChars ?? 0) || 0, liveChangedTextChars)
    const nextChangedEver = Boolean(record.changedEver) || nextMax > run.editThreshold
    if (
      nextMax !== (Number(record.maxPostAcceptEdit ?? 0) || 0)
      || nextMaxChangedTextChars !== (Number(record.maxChangedTextChars ?? 0) || 0)
      || nextChangedEver !== Boolean(record.changedEver)
    ) {
      changed = true
      return {
        ...record,
        maxPostAcceptEdit: nextMax,
        maxChangedTextChars: nextMaxChangedTextChars,
        changedEver: nextChangedEver,
      }
    }
    return record
  })
  if (!changed) return run
  return {
    ...run,
    acceptedSuggestions: nextAccepted,
  }
}

export const summarizeExperimentRun = (
  run: ExperimentRun,
  currentShapes: ShapeDraft[],
): ExperimentSummary => {
  const shapeMap = run.endedAt ? null : new Map<string, ExperimentShapeSnapshot>()
  if (shapeMap) {
    for (const shape of currentShapes) {
      shapeMap.set(String(shape.id), shapeToSnapshot(shape))
    }
  }
  const acceptedSuggestions = run.acceptedSuggestions.map((record) => (
    deriveAcceptedSuggestion(record, shapeMap, run.editThreshold)
  ))
  const aiInvokeTimes = run.requestRounds.length
  const previewCount = run.previews.length
  const acceptCount = acceptedSuggestions.length
  const dismissCount = run.dismisses.length
  const completedRounds = run.requestRounds.filter((round) => round.status === 'completed')
  const completedRoundCount = completedRounds.length
  const totalPromptTokens = completedRounds.reduce((sum, round) => sum + Math.max(0, round.promptTokens || 0), 0)
  const acceptedUsableUnits = acceptedSuggestions.reduce((sum, record) => sum + Math.max(0, record.usableUnits || 0), 0)
  const acceptedTextChars = acceptedSuggestions.reduce((sum, record) => sum + Math.max(0, record.acceptedTextChars || 0), 0)
  const changedTextChars = acceptedSuggestions.reduce((sum, record) => sum + Math.max(0, record.changedTextChars || 0), 0)
  const straightUseCount = acceptedSuggestions.filter((record) => record.straightUse).length
  const activeAlignmentCount = acceptedSuggestions.filter((record) => record.activeBlockAligned).length

  const phaseIds: string[] = []
  const seenPhases = new Set<string>()
  for (const round of run.requestRounds) {
    if (!seenPhases.has(round.phaseId)) {
      phaseIds.push(round.phaseId)
      seenPhases.add(round.phaseId)
    }
  }
  for (const accepted of acceptedSuggestions) {
    if (!seenPhases.has(accepted.phaseId)) {
      phaseIds.push(accepted.phaseId)
      seenPhases.add(accepted.phaseId)
    }
  }

  const phaseSpecificEfficiency = phaseIds.map((phaseId) => {
    const phaseRounds = run.requestRounds.filter((round) => round.phaseId === phaseId)
    const phaseAccepted = acceptedSuggestions.filter((record) => record.phaseId === phaseId)
    const phasePromptTokens = phaseRounds
      .filter((round) => round.status === 'completed')
      .reduce((sum, round) => sum + Math.max(0, round.promptTokens || 0), 0)
    const phaseUsableUnits = phaseAccepted.reduce((sum, record) => sum + Math.max(0, record.usableUnits || 0), 0)
    return {
      phaseId,
      invoke: phaseRounds.length,
      straightUseRate: safeRatio(phaseAccepted.filter((record) => record.straightUse).length, phaseAccepted.length),
      acceptedOutputPer1kToken: phasePromptTokens > 0 ? (phaseUsableUnits / phasePromptTokens) * 1000 : null,
      promptTokens: phasePromptTokens,
      acceptedUsableUnits: phaseUsableUnits,
      acceptedCount: phaseAccepted.length,
    }
  })

  return {
    aiInvokeTimes,
    previewCount,
    acceptCount,
    dismissCount,
    completedRoundCount,
    totalPromptTokens,
    acceptedUsableUnits,
    acceptedTextChars,
    changedTextChars,
    suggestionAcceptanceRate: safeRatio(acceptCount, previewCount),
    dismissRate: safeRatio(dismissCount, previewCount),
    straightUseRate: safeRatio(straightUseCount, acceptCount),
    userChangedRate: safeRatio(changedTextChars, acceptedTextChars),
    promptTokensPerRound: safeRatio(totalPromptTokens, completedRoundCount),
    acceptedUsableContentPer1kTokens: totalPromptTokens > 0 ? (acceptedUsableUnits / totalPromptTokens) * 1000 : null,
    activeBlockAlignmentRate: safeRatio(activeAlignmentCount, acceptCount),
    phaseSpecificEfficiency,
    acceptedSuggestions,
  }
}
