import React from 'react'
import type { PromptMode, ColorName } from './ai/types'
import { COLORS } from './ai/normalize'

const UI_THEME = {
  ink: '#0f172a',
  inkSoft: '#334155',
  inkMuted: '#64748b',
  line: 'rgba(148, 163, 184, 0.28)',
  lineStrong: 'rgba(100, 116, 139, 0.34)',
  panel: 'rgba(255,255,255,0.82)',
  panelDeep: 'rgba(248,250,252,0.92)',
  panelHover: 'rgba(255,255,255,0.94)',
  glow: '0 22px 50px rgba(15,23,42,0.12), 0 4px 14px rgba(15,23,42,0.06)',
  glowStrong: '0 24px 60px rgba(15,23,42,0.16), 0 6px 18px rgba(15,23,42,0.1)',
  accent: '#2563eb',
  accentSoft: 'rgba(37,99,235,0.1)',
}

type UiIconName =
  | 'grid'
  | 'snap'
  | 'curve'
  | 'spark'
  | 'ask'
  | 'check'
  | 'close'
  | 'hand'
  | 'pen'
  | 'eraser'
  | 'ellipse'
  | 'text'
  | 'select'

function UiIcon({
  name,
  size = 14,
  stroke = 'currentColor',
}: {
  name: UiIconName
  size?: number
  stroke?: string
}) {
  const common = {
    fill: 'none',
    stroke,
    strokeWidth: 1.7,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  }
  return (
    <svg
      aria-hidden
      width={size}
      height={size}
      viewBox="0 0 16 16"
      style={{ flexShrink: 0, display: 'block' }}
    >
      {name === 'grid' && (
        <g {...common}>
          <rect x="2.2" y="2.2" width="11.6" height="11.6" rx="1.6" />
          <path d="M7.5 2.2v11.6M11 2.2v11.6M2.2 7.5h11.6M2.2 11h11.6" />
        </g>
      )}
      {name === 'snap' && (
        <g {...common}>
          <circle cx="8" cy="8" r="4.5" />
          <path d="M8 1.5v2.2M8 12.3v2.2M1.5 8h2.2M12.3 8h2.2M8 6.1v3.8M6.1 8h3.8" />
        </g>
      )}
      {name === 'curve' && (
        <g {...common}>
          <path d="M2.3 11.8C4.5 4.1 7.4 4.1 9.8 8.2c1.1 1.9 2.1 2.2 3.9-.2" />
          <circle cx="2.3" cy="11.8" r="1.1" />
          <circle cx="9.8" cy="8.2" r="1.1" />
          <circle cx="13.7" cy="8" r="1.1" />
        </g>
      )}
      {name === 'spark' && (
        <g {...common}>
          <path d="M8 1.7l1.2 3.1 3.1 1.2-3.1 1.2L8 10.3 6.8 7.2 3.7 6l3.1-1.2L8 1.7Z" />
          <path d="M12.5 10.7v3M11 12.2h3" />
        </g>
      )}
      {name === 'ask' && (
        <g {...common}>
          <path d="M3 4.1h10v6.4a1.2 1.2 0 0 1-1.2 1.2H7.2L4 14.2v-2.5H4A1 1 0 0 1 3 10.7z" />
          <path d="M6 7.7h4M6 5.8h6" />
        </g>
      )}
      {name === 'check' && <path {...common} d="M2.8 8.5 6.4 12l6.8-7.4" />}
      {name === 'close' && <path {...common} d="M3.3 3.3l9.4 9.4M12.7 3.3 3.3 12.7" />}
      {name === 'hand' && (
        <g {...common}>
          <path d="M6 13.3c-1.5 0-2.7-1.2-2.7-2.7V7.2c0-.6.5-1.1 1.1-1.1s1.1.5 1.1 1.1V5.4c0-.6.5-1.1 1.1-1.1s1.1.5 1.1 1.1V4.8c0-.6.5-1.1 1.1-1.1s1.1.5 1.1 1.1v.9c0-.5.4-.9.9-.9s.9.4.9.9v4.9c0 1.5-1.2 2.7-2.7 2.7H6z" />
        </g>
      )}
      {name === 'pen' && (
        <g {...common}>
          <path d="M3 13l2.8-.6 6.7-6.7-2.2-2.2-6.7 6.7L3 13z" />
          <path d="M9.4 3.5l2.2 2.2" />
        </g>
      )}
      {name === 'eraser' && (
        <g {...common}>
          <path d="M6.2 3.2 13 10a1.2 1.2 0 0 1 0 1.7l-1.2 1.2a1.2 1.2 0 0 1-1.7 0L3.3 6.1a1.2 1.2 0 0 1 0-1.7l1.2-1.2a1.2 1.2 0 0 1 1.7 0Z" />
          <path d="M7.7 13.4h5" />
        </g>
      )}
      {name === 'ellipse' && (
        <g {...common}>
          <ellipse cx="8" cy="8" rx="5.6" ry="4.1" />
        </g>
      )}
      {name === 'text' && (
        <g {...common}>
          <path d="M2.5 3.3h11M8 3.3v9.4M5.8 12.7h4.4" />
        </g>
      )}
      {name === 'select' && (
        <g {...common}>
          <rect x="2.4" y="2.4" width="8.5" height="8.5" rx="1.2" strokeDasharray="2 1.4" />
          <path d="M10.8 10.2l2.8 3.1M10.6 7.8l2.6 1-.9 2.6" />
        </g>
      )}
    </svg>
  )
}

function IconLabel({
  icon,
  children,
  tone,
}: {
  icon: UiIconName
  children: React.ReactNode
  tone?: string
}) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <UiIcon name={icon} size={14} stroke={tone ?? 'currentColor'} />
      <span>{children}</span>
    </span>
  )
}

const BTN_BASE: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 6,
  padding: '7px 12px',
  borderRadius: 999,
  border: `1px solid ${UI_THEME.lineStrong}`,
  background: `linear-gradient(180deg, ${UI_THEME.panelHover}, rgba(241,245,249,0.92))`,
  color: UI_THEME.ink,
  cursor: 'pointer',
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: '0.01em',
  boxShadow: '0 1px 0 rgba(255,255,255,0.9) inset, 0 6px 14px rgba(15,23,42,0.06)',
  transition: 'transform 120ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease',
}

const CARD: React.CSSProperties = {
  padding: 14,
  border: `1px solid ${UI_THEME.line}`,
  borderRadius: 16,
  background: `linear-gradient(180deg, ${UI_THEME.panelHover}, rgba(248,250,252,0.9))`,
  boxShadow: '0 10px 24px rgba(15,23,42,0.05)',
  backdropFilter: 'blur(10px)',
}

const CARD_TITLE: React.CSSProperties = {
  fontSize: 12,
  color: UI_THEME.inkMuted,
  marginBottom: 8,
  letterSpacing: '.08em',
  textTransform: 'uppercase',
  fontWeight: 700,
}

const SEL: React.CSSProperties = {
  padding: '7px 10px',
  borderRadius: 10,
  border: `1px solid ${UI_THEME.lineStrong}`,
  background: 'rgba(255,255,255,0.92)',
  color: UI_THEME.ink,
  boxShadow: '0 1px 0 rgba(255,255,255,0.95) inset',
  outline: 'none',
}

type ButtonLike = React.ButtonHTMLAttributes<HTMLButtonElement>

const Btn = React.forwardRef<HTMLButtonElement, ButtonLike>(({ style, ...props }, ref) => (
  <button
    ref={ref}
    {...props}
    style={{
      ...BTN_BASE,
      ...(props.disabled ? { opacity: 0.52, cursor: 'not-allowed', boxShadow: 'none' } : null),
      ...style,
    }}
  />
))
Btn.displayName = 'Btn'

export type TopToolbarProps = {
  showGrid: boolean
  snap: boolean
  curveTurns: boolean
  onToggleGrid: () => void
  onToggleSnap: () => void
  onToggleCurve: () => void
  onAskAI: () => void
  onAcceptAI: () => void
  onDismissAI: () => void
  autoComplete: boolean
  autoCountdown: number | null
  hasActivePreview: boolean
  onToggleAutoComplete: (enabled: boolean) => void
}

export function TopToolbar(props: TopToolbarProps) {
  const {
    showGrid,
    snap,
    curveTurns,
    onToggleGrid,
    onToggleSnap,
    onToggleCurve,
    onAskAI,
    onAcceptAI,
    onDismissAI,
    autoComplete,
    autoCountdown,
    hasActivePreview,
    onToggleAutoComplete,
  } = props
  const autoCompleteState = hasActivePreview
    ? 'Paused'
    : autoCountdown != null
      ? `Next ${autoCountdown}s`
      : 'Idle'

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        transform: 'translateX(-50%)',
        top: 14,
        zIndex: 1000,
        display: 'flex',
        gap: 10,
        alignItems: 'center',
        flexWrap: 'wrap',
        justifyContent: 'center',
        maxWidth: 'calc(100vw - 28px)',
        background: 'linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.84))',
        backdropFilter: 'blur(14px) saturate(120%)',
        padding: '10px 14px',
        borderRadius: 18,
        boxShadow: UI_THEME.glow,
        border: `1px solid ${UI_THEME.line}`,
      }}
    >
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '5px 10px 5px 6px',
          borderRadius: 999,
          border: `1px solid ${UI_THEME.line}`,
          background: 'rgba(255,255,255,0.7)',
          boxShadow: '0 1px 0 rgba(255,255,255,0.9) inset',
          marginRight: 2,
        }}
      >
        <span
          aria-hidden
          style={{
            width: 18,
            height: 18,
            borderRadius: '50%',
            background: 'conic-gradient(from 210deg, #0ea5e9, #2563eb, #14b8a6, #0ea5e9)',
            boxShadow: '0 0 0 2px rgba(255,255,255,0.9) inset, 0 6px 12px rgba(37,99,235,0.22)',
          }}
        />
        <span style={{ fontSize: 12, fontWeight: 700, color: UI_THEME.ink, letterSpacing: '.06em' }}>AIPAD</span>
        <span style={{ fontSize: 10, color: UI_THEME.inkMuted }}>Canvas Lab</span>
      </div>
      <Btn onClick={onToggleGrid}>
        <IconLabel icon="grid">{showGrid ? 'Grid: ON' : 'Grid: OFF'}</IconLabel>
      </Btn>
      <Btn onClick={onToggleSnap}>
        <IconLabel icon="snap">{snap ? 'Snap: ON' : 'Snap: OFF'}</IconLabel>
      </Btn>
      <Btn onClick={onToggleCurve}>
        <IconLabel icon="curve">{curveTurns ? 'Curve: ON' : 'Curve: OFF'}</IconLabel>
      </Btn>
      <button
        onClick={() => onToggleAutoComplete(!autoComplete)}
        title={`Auto Complete (${autoCompleteState})`}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '7px 10px',
          borderRadius: 999,
          border: `1px solid ${autoComplete ? 'rgba(34,197,94,0.45)' : UI_THEME.lineStrong}`,
          background: autoComplete
            ? 'linear-gradient(180deg, rgba(240,253,244,0.95), rgba(220,252,231,0.92))'
            : 'linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.9))',
          cursor: 'pointer',
          color: UI_THEME.ink,
          fontSize: 12,
          fontWeight: 600,
          boxShadow: autoComplete
            ? '0 8px 18px rgba(34,197,94,0.12)'
            : '0 6px 14px rgba(15,23,42,0.05)',
        }}
      >
        <IconLabel icon="spark">
          Auto Complete
        </IconLabel>
        <span
          style={{
            position: 'relative',
            width: 34,
            height: 20,
            borderRadius: 999,
            background: autoComplete ? '#22c55e' : '#cbd5e1',
            transition: 'background 0.2s ease',
            flexShrink: 0,
            boxShadow: 'inset 0 1px 2px rgba(15,23,42,0.14)',
          }}
        >
          <span
            style={{
              position: 'absolute',
              top: 2,
              left: autoComplete ? 16 : 2,
              width: 16,
              height: 16,
              borderRadius: '50%',
              background: '#fff',
              boxShadow: '0 1px 3px rgba(15,23,42,0.25)',
              transition: 'left 0.2s ease',
            }}
          />
        </span>
        <span style={{ fontSize: 11, color: UI_THEME.inkMuted, minWidth: 56 }}>{autoCompleteState}</span>
      </button>

      <div style={{ width: 10, height: 24, borderLeft: `1px solid ${UI_THEME.line}`, margin: '0 2px' }} />

      <Btn
        onClick={onAskAI}
        style={{
          borderColor: 'rgba(59,130,246,0.34)',
          background: 'linear-gradient(180deg, rgba(239,246,255,0.95), rgba(219,234,254,0.92))',
          color: '#1d4ed8',
        }}
      >
        <IconLabel icon="ask" tone="#1d4ed8">Ask AI</IconLabel>
      </Btn>
      <Btn
        onClick={onAcceptAI}
        style={{
          borderColor: 'rgba(22,163,74,0.32)',
          background: 'linear-gradient(180deg, rgba(240,253,244,0.96), rgba(220,252,231,0.92))',
          color: '#166534',
        }}
      >
        <IconLabel icon="check" tone="#166534">Accept</IconLabel>
      </Btn>
      <Btn
        onClick={onDismissAI}
        style={{
          borderColor: 'rgba(239,68,68,0.28)',
          background: 'linear-gradient(180deg, rgba(254,242,242,0.96), rgba(254,226,226,0.92))',
          color: '#b91c1c',
        }}
      >
        <IconLabel icon="close" tone="#b91c1c">Dismiss</IconLabel>
      </Btn>
    </div>
  )
}

export type SettingsButtonProps = {
  open: boolean
  onToggle: () => void
}

export function SettingsButton(props: SettingsButtonProps) {
  const { open, onToggle } = props
  const [hovered, setHovered] = React.useState(false)
  const [pressed, setPressed] = React.useState(false)
  const translateY = pressed ? 1 : hovered ? -1 : 0
  const scale = pressed ? 0.97 : hovered ? 1.04 : 1
  const shellBorder = open ? '1px solid rgba(37,99,235,0.46)' : `1px solid ${UI_THEME.lineStrong}`
  const shellBackground = open
    ? 'linear-gradient(150deg, rgba(29,78,216,0.98), rgba(14,165,233,0.92))'
    : 'linear-gradient(150deg, rgba(255,255,255,0.98), rgba(241,245,249,0.94))'
  const shellShadow = open
    ? '0 14px 30px rgba(37,99,235,0.3), 0 4px 10px rgba(14,165,233,0.22)'
    : hovered
      ? '0 12px 24px rgba(59,130,246,0.18), 0 3px 8px rgba(15,23,42,0.1)'
      : '0 6px 12px rgba(15,23,42,0.12)'

  return (
    <button
      type="button"
      title={open ? 'Close settings' : 'Open settings'}
      onClick={onToggle}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => {
        setHovered(false)
        setPressed(false)
      }}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      style={{
        position: 'absolute',
        top: 76,
        right: 364,
        zIndex: 1100,
        width: 46,
        height: 46,
        borderRadius: '50%',
        border: shellBorder,
        background: shellBackground,
        color: open ? '#eff6ff' : UI_THEME.ink,
        fontSize: 19,
        fontWeight: 700,
        lineHeight: 1,
        cursor: 'pointer',
        boxShadow: shellShadow,
        transform: `translateY(${translateY}px) scale(${scale})`,
        transition: 'transform 140ms ease, box-shadow 180ms ease, background 180ms ease, border-color 180ms ease',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      <span
        aria-hidden
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '50%',
          background: open
            ? 'radial-gradient(circle at 30% 25%, rgba(255,255,255,0.34), transparent 58%)'
            : 'radial-gradient(circle at 30% 25%, rgba(148,163,184,0.15), transparent 58%)',
        }}
      />
      <span
        style={{
          position: 'relative',
          zIndex: 1,
          transform: open ? 'rotate(24deg)' : 'rotate(0deg)',
          transition: 'transform 220ms ease',
        }}
      >
        {'\u2699'}
      </span>
    </button>
  )
}

type GrowDir = 'down' | 'up' | 'left' | 'right' | 'right-down'
type GroupPromoteMode = 'heuristic' | 'hybrid' | 'llm'
type VisionImageMode = 'off' | 'auto' | 'always'

export type SidePanelProps = {
  toolMode: 'pen' | 'eraser' | 'ellipse' | 'hand' | 'text' | 'select'
  onToolModeChange: (mode: 'pen' | 'eraser' | 'ellipse' | 'hand' | 'text' | 'select') => void
  eraserRadius: number
  onEraserRadiusChange: (radius: number) => void
  brushSize: 's' | 'm' | 'l' | 'xl'
  onBrushSizeChange: (size: 's' | 'm' | 'l' | 'xl') => void
  brushColor: ColorName
  onBrushColorChange: (color: ColorName) => void
  aiScale: number
  onAiScaleChange: (scale: number) => void
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
  onExportJSON: () => void
  onImportJSON: (file: File) => void
  fileInputRef: React.RefObject<HTMLInputElement | null>
  onExportAI: () => void
  onApplyAIStub: () => void
  onPreviewAI: () => void
  llmModel: string
  llmTemperature: number
  llmTopP: number
  llmMaxTokens: number
  onLlmModelChange: (value: string) => void
  onLlmTemperatureChange: (value: number) => void
  onLlmTopPChange: (value: number) => void
  onLlmMaxTokensChange: (value: number) => void
  onResetLLMSettings: () => void
  groupPromoteMode: GroupPromoteMode
  onGroupPromoteModeChange: (value: GroupPromoteMode) => void
  visionImageMode: VisionImageMode
  onVisionImageModeChange: (value: VisionImageMode) => void
  settingsOpen: boolean
  onCloseSettings: () => void
  promptMode: PromptMode
  visionVersion: number
  onVisionVersionChange: (value: number) => void
  textSettings: {
    fontFamily: string
    fontSize: number
    fontWeight: string
    growDir: GrowDir
  }
  onTextSettingsChange: (next: Partial<{ fontFamily: string; fontSize: number; fontWeight: string; growDir: GrowDir }>) => void
  onToggleGraphInspector: () => void
  graphInspectorActive: boolean
}

export function SidePanel(props: SidePanelProps) {
  const {
    toolMode,
    onToolModeChange,
    eraserRadius,
    onEraserRadiusChange,
    brushSize,
    onBrushSizeChange,
    brushColor,
    onBrushColorChange,
    aiScale,
    onAiScaleChange,
    canUndo,
    canRedo,
    onUndo,
    onRedo,
    onExportJSON,
    onImportJSON,
    fileInputRef,
    onExportAI,
    onApplyAIStub,
    onPreviewAI,
    llmModel,
    llmTemperature,
    llmTopP,
    llmMaxTokens,
    onLlmModelChange,
    onLlmTemperatureChange,
    onLlmTopPChange,
    onLlmMaxTokensChange,
    onResetLLMSettings,
    groupPromoteMode,
    onGroupPromoteModeChange,
    visionImageMode,
    onVisionImageModeChange,
    settingsOpen,
    onCloseSettings,
    promptMode,
    visionVersion,
    onVisionVersionChange,
    textSettings,
    onTextSettingsChange,
    onToggleGraphInspector,
    graphInspectorActive,
  } = props
  const toolButtonsPrimary = [
    { id: 'hand', label: 'Hand', icon: 'hand' },
    { id: 'pen', label: 'Pen', icon: 'pen' },
    { id: 'eraser', label: 'Eraser', icon: 'eraser' },
    { id: 'ellipse', label: 'Ellipse', icon: 'ellipse' },
  ] as const
  const toolButtonsSecondary = [
    { id: 'text', label: 'Text', icon: 'text' },
    { id: 'select', label: 'Select', icon: 'select' },
  ] as const
  return (
    <div
      style={{
        position: 'absolute',
        top: 74,
        right: 14,
        bottom: 176,
        width: 'min(360px, calc(100vw - 28px))',
        minWidth: 300,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        overflow: 'auto',
        padding: 14,
        background:
          'linear-gradient(180deg, rgba(255,255,255,0.78), rgba(248,250,252,0.72))',
        backdropFilter: 'blur(16px) saturate(125%)',
        border: `1px solid ${UI_THEME.line}`,
        borderRadius: 20,
        boxShadow: UI_THEME.glowStrong,
      }}
    >
      {settingsOpen && (
        <section
          style={{
            ...CARD,
            background: 'linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,0.96))',
            borderColor: 'rgba(148,163,184,0.28)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ ...CARD_TITLE, marginBottom: 0 }}>Settings</div>
            <Btn onClick={onCloseSettings} style={{ padding: '4px 10px', fontSize: 12 }}>
              Close
            </Btn>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8 }}>
            <Btn onClick={onUndo} disabled={!canUndo} style={{ opacity: canUndo ? 1 : 0.6 }}>
              Undo
            </Btn>
            <Btn onClick={onRedo} disabled={!canRedo} style={{ opacity: canRedo ? 1 : 0.6 }}>
              Redo
            </Btn>
            <Btn onClick={onExportJSON}>Export JSON</Btn>
            <Btn onClick={() => fileInputRef.current?.click()}>Import JSON</Btn>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json,.json"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) onImportJSON(file)
              }}
            />
            <Btn onClick={onExportAI}>Export Strokes (AI)</Btn>
            <Btn onClick={onApplyAIStub}>Apply AI (stub)</Btn>
            <Btn onClick={onPreviewAI}>Preview AI</Btn>
          </div>

          <div style={{ marginTop: 12, fontSize: 11, color: '#64748b', letterSpacing: '.3px', textTransform: 'uppercase' }}>
            Runtime LLM
          </div>
          <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#475569' }}>Model</span>
              <input
                style={{ ...SEL, borderRadius: 10, width: '100%' }}
                type="text"
                value={llmModel}
                onChange={(e) => onLlmModelChange(e.target.value)}
                placeholder="Use backend default when empty"
              />
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 12, color: '#475569' }}>Temp</span>
                <input
                  style={{ ...SEL, borderRadius: 10, width: '100%' }}
                  type="number"
                  step="0.05"
                  min={0}
                  max={2}
                  value={llmTemperature}
                  onChange={(e) => onLlmTemperatureChange(Number(e.target.value))}
                />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 12, color: '#475569' }}>Top P</span>
                <input
                  style={{ ...SEL, borderRadius: 10, width: '100%' }}
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={llmTopP}
                  onChange={(e) => onLlmTopPChange(Number(e.target.value))}
                />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 12, color: '#475569' }}>Max Tokens</span>
                <input
                  style={{ ...SEL, borderRadius: 10, width: '100%' }}
                  type="number"
                  step={128}
                  min={256}
                  max={32768}
                  value={llmMaxTokens}
                  onChange={(e) => onLlmMaxTokensChange(Number(e.target.value))}
                />
              </label>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Btn onClick={onResetLLMSettings} style={{ padding: '5px 12px', fontSize: 12 }}>
                Reset to Env
              </Btn>
            </div>
          </div>

          <div style={{ marginTop: 12, fontSize: 11, color: '#64748b', letterSpacing: '.3px', textTransform: 'uppercase' }}>
            Auto Maintain
          </div>
          <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#475569' }}>Group Promote Mode</span>
              <select
                value={groupPromoteMode}
                onChange={(e) => onGroupPromoteModeChange(e.target.value as GroupPromoteMode)}
                style={{ ...SEL, borderRadius: 10, width: '100%' }}
                title="heuristic: rules only, hybrid: rules+LLM review on boundary cases, llm: always review"
              >
                <option value="heuristic">Heuristic</option>
                <option value="hybrid">Hybrid</option>
                <option value="llm">LLM Review</option>
              </select>
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 12, color: '#475569' }}>Vision Image Mode</span>
              <select
                value={visionImageMode}
                onChange={(e) => onVisionImageModeChange(e.target.value as VisionImageMode)}
                style={{ ...SEL, borderRadius: 10, width: '100%' }}
                title="off: text-only, auto: attach snapshot only for complex/ambiguous groups, always: always attach snapshot"
              >
                <option value="off">Off (text only)</option>
                <option value="auto">Auto (recommended)</option>
                <option value="always">Always attach snapshot</option>
              </select>
            </label>
          </div>

          {promptMode === 'vision' && (
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: '#334155', width: 120 }}>Vision version</label>
              <input
                type="number"
                step="0.1"
                min={1.0}
                value={visionVersion}
                onChange={(e) => onVisionVersionChange(Number(e.target.value) || 2.0)}
                style={{ ...SEL, width: 120, height: 32, borderRadius: 8, padding: '0 8px' }}
                title="Vision protocol version (2.0 is two-phase)"
              />
            </div>
          )}
        </section>
      )}

      <section style={CARD}>
        <div style={CARD_TITLE}>Tools</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
          {toolButtonsPrimary.map(({ id, label, icon }) => (
            <Btn
              key={id}
              onClick={() => onToolModeChange(id)}
              style={{
                padding: '8px 10px',
                borderRadius: 12,
                flexDirection: 'column',
                gap: 4,
                minHeight: 54,
                ...(toolMode === id
                  ? {
                      outline: '2px solid rgba(59,130,246,0.38)',
                      borderColor: 'rgba(59,130,246,0.34)',
                      background: 'linear-gradient(180deg, rgba(239,246,255,0.96), rgba(219,234,254,0.92))',
                      color: '#1d4ed8',
                      boxShadow: '0 8px 18px rgba(37,99,235,0.12)',
                    }
                  : {}),
              }}
              title={label}
            >
              <UiIcon name={icon} size={15} />
              <span style={{ fontSize: 11, fontWeight: 700, lineHeight: 1 }}>{label}</span>
            </Btn>
          ))}
        </div>
        <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8 }}>
          {toolButtonsSecondary.map(({ id, label, icon }) => (
            <Btn
              key={id}
              onClick={() => onToolModeChange(id)}
              style={{
                padding: '8px 10px',
                borderRadius: 12,
                justifyContent: 'flex-start',
                ...(toolMode === id
                  ? {
                      outline: '2px solid rgba(59,130,246,0.38)',
                      borderColor: 'rgba(59,130,246,0.34)',
                      background: 'linear-gradient(180deg, rgba(239,246,255,0.96), rgba(219,234,254,0.92))',
                      color: '#1d4ed8',
                    }
                  : {}),
              }}
              title={label}
            >
              <UiIcon name={icon} size={14} />
              <span>{label}</span>
            </Btn>
          ))}
        </div>

        {toolMode === 'eraser' && (
          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#555' }}>Radius</span>
            <input
              style={{ ...SEL, width: 90 }}
              type="number"
              min={4}
              max={64}
              step={2}
              value={eraserRadius}
              onChange={(e) => {
                const next = Math.max(4, Math.min(64, Number(e.target.value) || 14))
                onEraserRadiusChange(next)
              }}
              title="Eraser radius (px)"
            />
          </div>
        )}

        {toolMode === 'text' && (
          <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
            <div>
              <span style={{ fontSize: 12, color: '#555', display: 'block', marginBottom: 4 }}>Font family</span>
              <select
                value={textSettings.fontFamily}
                onChange={(e) => onTextSettingsChange({ fontFamily: e.target.value })}
                style={{ ...SEL, width: '100%' }}
              >
                {['sans-serif', 'serif', 'monospace', 'cursive'].map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <label style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                <span style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>Font size (px)</span>
                <input
                  style={{ ...SEL, width: '100%' }}
                  type="number"
                  min={8}
                  max={96}
                  value={textSettings.fontSize}
                  onChange={(e) => {
                    const v = Number(e.target.value) || 16
                    onTextSettingsChange({ fontSize: Math.max(8, Math.min(96, v)) })
                  }}
                />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                <span style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>Weight</span>
                <select
                  style={{ ...SEL, width: '100%' }}
                  value={textSettings.fontWeight}
                  onChange={(e) => onTextSettingsChange({ fontWeight: e.target.value })}
                >
                  {['300', '400', '500', '600', '700'].map((w) => (
                    <option key={w} value={w}>{w}</option>
                  ))}
                </select>
              </label>
            </div>
            <div>
              <span style={{ fontSize: 12, color: '#555', display: 'block', marginBottom: 4 }}>Grow direction</span>
              <select
                value={textSettings.growDir}
                onChange={(e) => onTextSettingsChange({ growDir: e.target.value as GrowDir })}
                style={{ ...SEL, width: '100%' }}
              >
                {(['right-down', 'down', 'right', 'up', 'left'] as const).map((dir) => (
                  <option key={dir} value={dir}>{dir}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </section>

      <section style={CARD}>
        <div style={CARD_TITLE}>Knowledge Graph</div>
        <Btn
          onClick={onToggleGraphInspector}
          style={{
            width: '100%',
            justifyContent: 'center',
            background: graphInspectorActive ? 'linear-gradient(120deg, rgba(59,130,246,0.25), rgba(236,72,153,0.2))' : 'rgba(59,130,246,0.12)',
            borderColor: graphInspectorActive ? 'rgba(236,72,153,0.7)' : 'rgba(59,130,246,0.6)',
            color: graphInspectorActive ? '#1d4ed8' : '#2563eb',
            fontWeight: 600,
          }}
        >
          {graphInspectorActive ? 'Hide Graph View' : 'Show Graph View'}
        </Btn>
        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>
          Visualize the semantic blocks, summaries, and recent fragments generated by Auto Maintain.
        </div>
      </section>

      <section style={CARD}>
        <div style={CARD_TITLE}>Brush</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 12, color: '#555' }}>Size</span>
          <select
            style={{ ...SEL, width: 90 }}
            value={brushSize}
            onChange={(e) => onBrushSizeChange(e.target.value as 's' | 'm' | 'l' | 'xl')}
          >
            <option value="s">S</option>
            <option value="m">M</option>
            <option value="l">L</option>
            <option value="xl">XL</option>
          </select>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 8 }}>
          {COLORS.map((c) => (
            <button
              key={c}
              title={c}
              onClick={() => onBrushColorChange(c as ColorName)}
              style={{
                width: 28,
                height: 28,
                borderRadius: 8,
                border: `2px solid ${brushColor === c ? '#4aa3ff' : '#e5e7eb'}`,
                background: c === 'white' ? '#fff' : c.replace('light-', 'light'),
                boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.04)',
              }}
            />
          ))}
        </div>
      </section>

      <section style={CARD}>
        <div style={CARD_TITLE}>AI Scale</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input
            type="range"
            min={4}
            max={64}
            step={1}
            value={aiScale}
            onChange={(e) => onAiScaleChange(Number(e.target.value) || 16)}
            title="Max points for AI stroke (model is asked to keep under this)"
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 12, color: '#333', width: 32, textAlign: 'right' }}>{aiScale}</span>
        </div>
      </section>


    </div>
  )
}

export type AIFeedEntry = {
  payloadId: string
  time: number
  items: { id: string; desc?: string }[]
}

type BottomPanelGraphFragment = {
  id: string
  type: string
  text: string
  bbox: [number, number, number, number] | null
}

type BottomPanelGraphBlock = {
  blockId: string
  label: string
  summary: string
  updatedAt?: string
  color: string
  bbox: [number, number, number, number] | null
  fragments: BottomPanelGraphFragment[]
  relationships: Array<{ target: string; type: string; score?: number }>
}

export type BottomPanelProps = {
  hint: string
  plannerNextStepHint?: string
  onHintChange: (value: string) => void
  onSubmit: () => void
  mode: PromptMode
  onModeCycle: () => void
  aiFeed: AIFeedEntry[]
  showAutoMaintain: boolean
  autoMaintainEnabled: boolean
  autoMaintainPending: boolean
  onToggleAutoMaintain: () => void
  graphInspectorActive: boolean
  viewportHeight: number
  graphBlocksDetailed: BottomPanelGraphBlock[]
  onFragmentFocus: (fragmentId: string) => void
  onBlockFocus: (blockId: string) => void
  onFragmentHover?: (fragmentId: string | null, blockId: string | null) => void
  onBlockHover?: (blockId: string | null) => void
  graphBlocks: Array<{ blockId: string; label: string; summary: string; updatedAt?: string }>
}

export function BottomPanel(props: BottomPanelProps) {
  const {
    hint,
    plannerNextStepHint,
    onHintChange,
    onSubmit,
    mode,
    onModeCycle,
    aiFeed,
    showAutoMaintain,
    autoMaintainEnabled,
    autoMaintainPending,
    onToggleAutoMaintain,
    graphInspectorActive,
    viewportHeight,
    graphBlocksDetailed,
    onFragmentFocus,
    onBlockFocus,
    onFragmentHover,
    onBlockHover,
    graphBlocks,
  } = props
  const resolvedFragmentHover = onFragmentHover ?? (() => {})
  const resolvedBlockHover = onBlockHover ?? (() => {})
  const expandedHeight = Math.max(viewportHeight * 0.5, 360)
  const panelMaxHeight = graphInspectorActive
    ? Math.min(expandedHeight, viewportHeight - 120)
    : 220
  const showDetailedBlocks =
    graphInspectorActive && autoMaintainEnabled && graphBlocksDetailed.length > 0
  const toRgba = (hex: string, alpha: number) => {
    const normalized = hex.replace('#', '')
    if (normalized.length !== 6) return `rgba(148, 163, 184, ${alpha})`
    const value = parseInt(normalized, 16)
    const r = (value >> 16) & 255
    const g = (value >> 8) & 255
    const b = value & 255
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }

  const modeConfig = {
    light: {
      title: '轻量补全：仅预测下一笔，快速响应',
      label: 'LIGHT',
      borderColor: 'rgba(37,99,235,0.32)',
      background: 'linear-gradient(135deg, rgba(239,246,255,0.95), rgba(224,242,254,0.9))',
      color: '#1d4ed8',
      boxShadow: '0 8px 18px rgba(37,99,235,0.12)',
      textShadow: 'none',
    },
    full: {
      title: '常规补全：可多笔',
      label: 'FULL',
      borderColor: 'rgba(217,119,6,0.28)',
      background: 'linear-gradient(135deg, rgba(255,251,235,0.97), rgba(254,243,199,0.9))',
      color: '#b45309',
      boxShadow: '0 8px 18px rgba(217,119,6,0.1)',
      textShadow: 'none',
    },
    vision: {
      title: '视觉增强：AI 视觉理解与创意绘制',
      label: 'VISION',
      borderColor: 'rgba(13,148,136,0.32)',
      background: 'linear-gradient(135deg, rgba(240,253,250,0.97), rgba(204,251,241,0.9))',
      color: '#0f766e',
      boxShadow: '0 8px 20px rgba(13,148,136,0.14)',
      textShadow: 'none',
    },
  } as const

  const styles = modeConfig[mode]

  return (
    <div
      style={{
        position: 'absolute',
        left: 14,
        right: 14,
        bottom: 14,
        zIndex: 1000,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.9), rgba(248,250,252,0.82))',
        backdropFilter: 'blur(16px) saturate(125%)',
        border: `1px solid ${UI_THEME.line}`,
        borderRadius: 18,
        padding: '12px 14px',
        boxShadow: UI_THEME.glowStrong,
        maxHeight: panelMaxHeight,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <input
          style={{
            ...SEL,
            width: '100%',
            borderRadius: 12,
            height: 42,
            padding: '0 14px',
            flex: '1 1 320px',
            background: 'rgba(255,255,255,0.92)',
            borderColor: 'rgba(148,163,184,0.32)',
            boxShadow: '0 1px 0 rgba(255,255,255,0.95) inset, 0 8px 18px rgba(15,23,42,0.04)',
          }}
          type="text"
          placeholder="hint for AI, e.g. clean curves / refine hair"
          value={hint}
          onChange={(e) => onHintChange(e.target.value)}
          title="Hint sent to backend /suggest"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSubmit()
            }
          }}
        />
        {showAutoMaintain && (
          <button
            title={autoMaintainEnabled ? '自动维护运行中，实时生成语义块与关系' : '启动自动维护：实时聚合文本与线稿'}
            onClick={onToggleAutoMaintain}
            disabled={autoMaintainPending}
            style={{
              padding: '10px 16px',
              borderRadius: 999,
              border: '1px solid',
              borderColor: autoMaintainEnabled ? 'rgba(20,184,166,0.38)' : 'rgba(37,99,235,0.34)',
              background: autoMaintainEnabled
                ? 'linear-gradient(135deg, rgba(240,253,250,0.98), rgba(204,251,241,0.92))'
                : 'linear-gradient(135deg, rgba(239,246,255,0.98), rgba(219,234,254,0.92))',
              color: autoMaintainEnabled ? '#0f766e' : '#1d4ed8',
              fontWeight: 700,
              letterSpacing: '0.04em',
              boxShadow: autoMaintainEnabled
                ? '0 12px 22px rgba(20,184,166,0.12)'
                : '0 12px 22px rgba(37,99,235,0.08)',
              cursor: autoMaintainPending ? 'wait' : 'pointer',
              opacity: autoMaintainPending ? 0.75 : 1,
              transition: 'all 0.25s ease',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <span style={{ position: 'relative', zIndex: 1 }}>
              {autoMaintainPending
                ? 'Engaging...'
                : autoMaintainEnabled
                  ? 'Auto Maintain ON'
                  : 'Auto Maintain'}
            </span>
            <span
              aria-hidden
              style={{
                content: '""',
                position: 'absolute',
                inset: 0,
                background: autoMaintainEnabled
                  ? 'radial-gradient(circle at 20% 20%, rgba(20,184,166,0.15), transparent 55%)'
                  : 'radial-gradient(circle at 20% 20%, rgba(37,99,235,0.12), transparent 52%)',
                mixBlendMode: 'screen',
              }}
            />
          </button>
        )}
        <button
          title={styles.title}
          onClick={onModeCycle}
          style={{
            padding: '9px 14px',
            borderRadius: 12,
            border: '1px solid',
            fontSize: '14px',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.25s ease',
            borderColor: styles.borderColor,
            background: styles.background,
            color: styles.color,
            boxShadow: styles.boxShadow,
            textShadow: styles.textShadow,
            letterSpacing: '.08em',
          }}
        >
          {styles.label}
        </button>
        <Btn
          onClick={onSubmit}
          style={{
            borderColor: 'rgba(37,99,235,0.34)',
            background: 'linear-gradient(180deg, rgba(239,246,255,0.95), rgba(219,234,254,0.9))',
            color: '#1d4ed8',
            minWidth: 76,
          }}
        >
          Send
        </Btn>
      </div>
      {showAutoMaintain && autoMaintainEnabled && (
        <div
          title="Planner Hint"
          style={{
            marginBottom: 10,
            borderRadius: 14,
            border: `1px solid ${UI_THEME.line}`,
            background: 'linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,0.9))',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.95), 0 10px 18px rgba(15,23,42,0.03)',
            padding: '11px 12px',
          }}
        >
          <div style={{ fontSize: 11, color: UI_THEME.inkMuted, letterSpacing: '.08em', marginBottom: 6, textTransform: 'uppercase', fontWeight: 700 }}>
            Assistant Planner Focus
          </div>
          <div
            style={{
              fontSize: 12,
              color: UI_THEME.ink,
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {plannerNextStepHint || 'No planner next-step hint yet.'}
          </div>
        </div>
      )}

      <div
        style={{
          flex: '1 1 auto',
          overflowY: 'auto',
          marginTop: 6,
          paddingRight: showDetailedBlocks ? 8 : 0,
        }}
      >
        <div style={{ fontSize: 12, color: UI_THEME.inkMuted, marginBottom: 8, fontWeight: 700, letterSpacing: '.06em' }}>AI Feed (latest)</div>
        {aiFeed.length === 0 ? (
          <div style={{ fontSize: 12, color: '#94a3b8', border: `1px dashed ${UI_THEME.line}`, borderRadius: 12, padding: '10px 12px', background: 'rgba(255,255,255,0.55)' }}>No AI packages yet.</div>
        ) : (
          aiFeed.map((entry) => (
            <div key={entry.payloadId} style={{ marginBottom: 8, border: `1px solid ${UI_THEME.line}`, borderRadius: 12, padding: '8px 10px', background: 'rgba(255,255,255,0.6)' }}>
              <div style={{ fontSize: 12, color: UI_THEME.inkSoft }}>
                <b>{new Date(entry.time).toLocaleTimeString()}</b> AIpayload <code>{entry.payloadId}</code>
              </div>
              <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                {entry.items.map((item, idx) => (
                  <li
                    key={`${item.id}_${idx}`}
                    style={{ fontSize: 12, color: UI_THEME.ink, listStyle: 'disc' }}
                  >
                    <code>{item.id}</code>
                    {item.desc ? ` · ${item.desc}` : ''}
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
        {showDetailedBlocks ? (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 13, color: '#1f2937', marginBottom: 12, fontWeight: 600 }}>
              Graph Blocks Overview · {graphBlocksDetailed.length} blocks
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
                gap: 16,
                alignItems: 'stretch',
              }}
            >
              {graphBlocksDetailed.map((block) => (
                <div
                  key={block.blockId}
                  style={{
                    border: `1px solid ${toRgba(block.color, 0.55)}`,
                    borderRadius: 12,
                    padding: '12px 14px',
                    background: `linear-gradient(135deg, ${toRgba(block.color, 0.14)}, ${toRgba(block.color, 0.05)})`,
                    boxShadow: `0 12px 32px ${toRgba(block.color, 0.22)}`,
                    minHeight: 220,
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                  onMouseEnter={() => resolvedBlockHover(block.blockId)}
                  onMouseLeave={() => resolvedBlockHover(null)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <button
                      type="button"
                      onClick={() => onBlockFocus(block.blockId)}
                      style={{
                        border: 'none',
                        background: 'none',
                        padding: 0,
                        margin: 0,
                        fontSize: 13,
                        fontWeight: 600,
                        color: block.color,
                        cursor: 'pointer',
                      }}
                    >
                      {block.label || block.blockId}
                    </button>
                    <span style={{ fontSize: 11, color: '#475569' }}>
                      {block.fragments.length} fragment{block.fragments.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: '#1f2937', lineHeight: 1.5 }}>
                    {block.summary || '暂无摘要'}
                  </div>
                  {block.updatedAt && (
                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 6 }}>
                      {new Date(block.updatedAt).toLocaleTimeString()}
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: '#475569', marginTop: 10, marginBottom: 6 }}>
                    Fragments
                  </div>
                  <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {block.fragments.length === 0 ? (
                      <li style={{ fontSize: 12, color: '#6b7280' }}>暂无 fragment</li>
                    ) : (
                      block.fragments.map((frag) => (
                        <li key={frag.id}>
                          <button
                            type="button"
                            onClick={() => onFragmentFocus(frag.id)}
                            style={{
                              width: '100%',
                              textAlign: 'left',
                              border: `1px solid ${toRgba(block.color, 0.45)}`,
                              background: toRgba(block.color, 0.12),
                              color: '#0f172a',
                              borderRadius: 10,
                              padding: '6px 8px',
                              cursor: 'pointer',
                              transition: 'background 0.2s ease',
                            }}
                            onMouseEnter={() => resolvedFragmentHover(frag.id, block.blockId)}
                            onMouseLeave={() => resolvedFragmentHover(null, null)}
                          >
                            <div style={{ fontSize: 11, color: block.color, fontWeight: 600 }}>
                              #{frag.type || 'fragment'}
                            </div>
                            <div style={{ fontSize: 12, color: '#0f172a', lineHeight: 1.45 }}>
                              {frag.text || 'No summary yet'}
                            </div>
                          </button>
                        </li>
                      ))
                    )}
                  </ul>
                  {block.relationships?.length ? (
                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>Relationships</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {block.relationships.map((rel, idx) => (
                          <span
                            key={`${rel.target}_${idx}`}
                            style={{
                              fontSize: 11,
                              padding: '4px 8px',
                              borderRadius: 999,
                              background: toRgba(block.color, 0.15),
                              border: `1px solid ${toRgba(block.color, 0.35)}`,
                              color: '#0f172a',
                            }}
                          >
                            {rel.type} → {rel.target}
                            {typeof rel.score === 'number' ? ` (${rel.score.toFixed(2)})` : ''}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : (
          showAutoMaintain && autoMaintainEnabled && graphBlocks.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>Graph Blocks</div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                  gap: 8,
                }}
              >
                {graphBlocks.map((block) => (
                  <div
                    key={block.blockId}
                    style={{
                      border: '1px solid rgba(99,102,241,0.25)',
                      borderRadius: 10,
                      padding: '8px 10px',
                      background: 'linear-gradient(135deg, rgba(79,70,229,0.08), rgba(14,165,233,0.05))',
                      boxShadow: '0 6px 16px rgba(79,70,229,0.12)',
                    }}
                  >
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#4338ca', marginBottom: 4 }}>
                      {block.label || block.blockId}
                    </div>
                    <div style={{ fontSize: 12, color: '#1f2937', lineHeight: 1.4 }}>
                      {block.summary || '暂无摘要'}
                    </div>
                    {block.updatedAt && (
                      <div style={{ fontSize: 10, color: '#6b7280', marginTop: 6 }}>
                        {new Date(block.updatedAt).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )
        )}
      </div>
    </div>
  )
}
