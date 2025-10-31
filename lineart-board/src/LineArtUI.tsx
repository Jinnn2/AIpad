import React from 'react'
import type { PromptMode, ColorName } from './ai/types'
import { COLORS } from './ai/normalize'

const BTN_BASE: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: 999,
  border: '1px solid #ccc',
  background: '#fff',
  cursor: 'pointer',
}

const CARD: React.CSSProperties = {
  padding: 12,
  border: '1px solid #e5e7eb',
  borderRadius: 12,
  background: 'rgba(255,255,255,0.6)',
}

const CARD_TITLE: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  marginBottom: 8,
  letterSpacing: '.3px',
  textTransform: 'uppercase',
}

const SEL: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: 999,
  border: '1px solid #ccc',
  background: '#fff',
}

type ButtonLike = React.ButtonHTMLAttributes<HTMLButtonElement>

const Btn = React.forwardRef<HTMLButtonElement, ButtonLike>(({ style, ...props }, ref) => (
  <button ref={ref} {...props} style={{ ...BTN_BASE, ...style }} />
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
  } = props

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        transform: 'translateX(-50%)',
        top: 16,
        zIndex: 1000,
        display: 'flex',
        gap: 8,
        alignItems: 'center',
        background: 'rgba(255,255,255,0.85)',
        backdropFilter: 'blur(6px)',
        padding: '8px 12px',
        borderRadius: 16,
        boxShadow: '0 6px 18px rgba(0,0,0,0.12)',
        border: '1px solid #e5e7eb',
      }}
    >
      <Btn onClick={onToggleGrid}>{showGrid ? 'Grid: ON' : 'Grid: OFF'}</Btn>
      <Btn onClick={onToggleSnap}>{snap ? 'Snap: ON' : 'Snap: OFF'}</Btn>
      <Btn onClick={onToggleCurve}>{curveTurns ? 'Curve: ON' : 'Curve: OFF'}</Btn>

      <div style={{ width: 10, height: 24, borderLeft: '1px solid #e5e7eb', margin: '0 4px' }} />

      <Btn
        onClick={onAskAI}
        style={{ borderColor: '#4aa3ff', background: 'rgba(74,163,255,0.14)' }}
      >
        Ask AI
      </Btn>
      <Btn
        onClick={onAcceptAI}
        style={{ borderColor: '#52d273', background: 'rgba(82,210,115,0.14)' }}
      >
        Accept
      </Btn>
      <Btn
        onClick={onDismissAI}
        style={{ borderColor: '#ff6b6b', background: 'rgba(255,107,107,0.14)' }}
      >
        Dismiss
      </Btn>
    </div>
  )
}

type GrowDir = 'down' | 'up' | 'left' | 'right'

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
  autoComplete: boolean
  onAutoCompleteToggle: (enabled: boolean) => void
  autoCountdown: number | null
  hasActivePreview: boolean
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
    autoComplete,
    onAutoCompleteToggle,
    autoCountdown,
    hasActivePreview,
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
    promptMode,
    visionVersion,
    onVisionVersionChange,
    textSettings,
    onTextSettingsChange,
    onToggleGraphInspector,
    graphInspectorActive,
  } = props
  return (
    <div
      style={{
        position: 'absolute',
        top: 80,
        right: 16,
        bottom: 180,
        width: 340,
        minWidth: 340,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        overflow: 'auto',
        padding: 12,
        background: 'rgba(255,255,255,0.75)',
        backdropFilter: 'blur(8px)',
        border: '1px solid #e5e7eb',
        borderRadius: 16,
        boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
      }}
    >
      <section style={CARD}>
        <div style={CARD_TITLE}>Tools</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
          {(['hand', 'pen', 'eraser', 'ellipse'] as const).map((t) => (
            <Btn
              key={t}
              onClick={() => onToolModeChange(t)}
              style={{
                padding: '8px 10px',
                ...(toolMode === t
                  ? { outline: '2px solid #4aa3ff', background: 'rgba(74,163,255,0.12)' }
                  : {}),
              }}
              title={t}
            >
              {t}
            </Btn>
          ))}
        </div>
        <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 8 }}>
          {(['text', 'select'] as const).map((t) => (
            <Btn
              key={t}
              onClick={() => onToolModeChange(t)}
              style={{
                padding: '8px 10px',
                ...(toolMode === t
                  ? { outline: '2px solid #4aa3ff', background: 'rgba(74,163,255,0.12)' }
                  : {}),
              }}
              title={t}
            >
              {t}
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
                {(['down', 'right', 'up', 'left'] as const).map((dir) => (
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
          可视化 Auto Maintain 生成的语义块、摘要与最近片段。
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

      <section style={CARD}>
        <div style={CARD_TITLE}>Auto Complete</div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
          <label style={{ fontSize: 13, color: '#333' }}>
            自动补全：5秒无操作触发 askAI。
          </label>
          <input
            type="checkbox"
            checked={autoComplete}
            onChange={(e) => onAutoCompleteToggle(e.target.checked)}
            title="开启后：无预览且 5 秒无新操作自动发送"
          />
        </div>
        <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>
          状态：
          {hasActivePreview
            ? '有预览，暂停自动发送'
            : autoCountdown != null
            ? `倒计时 ${autoCountdown}s`
            : '空闲'}
        </div>
      </section>

      <section style={CARD}>
        <div style={CARD_TITLE}>Actions</div>
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

        {promptMode === 'vision' && (
          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ fontSize: 12, color: '#333', width: 120 }}>Vision version</label>
            <input
              type="number"
              step="0.1"
              min={1.0}
              value={visionVersion}
              onChange={(e) => onVisionVersionChange(Number(e.target.value) || 2.0)}
              style={{ ...SEL, width: 120, height: 32, borderRadius: 8, padding: '0 8px' }}
              title="Vision 模式的协议版本（2.0 为二段式）"
            />
          </div>
        )}
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
}

export type BottomPanelProps = {
  hint: string
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
  graphBlocks: Array<{ blockId: string; label: string; summary: string; updatedAt?: string }>
}

export function BottomPanel(props: BottomPanelProps) {
  const {
    hint,
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
    graphBlocks,
  } = props
  const panelMaxHeight = graphInspectorActive
    ? Math.max(320, Math.min(viewportHeight * 0.6, viewportHeight - 120))
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
      borderColor: '#4aa3ff',
      background: 'linear-gradient(135deg, rgba(74,163,255,0.15), rgba(74,163,255,0.05))',
      color: '#4aa3ff',
      boxShadow: 'none',
      textShadow: 'none',
    },
    full: {
      title: '常规补全：可多笔',
      label: 'FULL',
      borderColor: '#ffb84a',
      background: 'linear-gradient(135deg, rgba(255,184,74,0.15), rgba(255,184,74,0.05))',
      color: '#ffb84a',
      boxShadow: 'none',
      textShadow: 'none',
    },
    vision: {
      title: '视觉增强：AI 视觉理解与创意绘制',
      label: 'VISION',
      borderColor: '#9b5cff',
      background: 'linear-gradient(135deg, rgba(155,92,255,0.25), rgba(255,92,200,0.25))',
      color: '#c88bff',
      boxShadow: '0 0 12px rgba(155,92,255,0.6), 0 0 24px rgba(255,92,200,0.4)',
      textShadow: '0 0 6px rgba(255,255,255,0.8)',
    },
  } as const

  const styles = modeConfig[mode]

  return (
    <div
      style={{
        position: 'absolute',
        left: 16,
        right: 16,
        bottom: 16,
        zIndex: 1000,
        background: 'rgba(255,255,255,0.85)',
        backdropFilter: 'blur(8px)',
        border: '1px solid #e5e7eb',
        borderRadius: 12,
        padding: '10px 12px',
        boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
        maxHeight: panelMaxHeight,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <input
          style={{ ...SEL, width: '100%', borderRadius: 10, height: 40, padding: '0 12px' }}
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
              padding: '9px 22px',
              borderRadius: 999,
              border: '2px solid',
              borderColor: autoMaintainEnabled ? 'rgba(236,72,153,0.8)' : 'rgba(59,130,246,0.8)',
              background: autoMaintainEnabled
                ? 'linear-gradient(135deg, rgba(168,85,247,0.95), rgba(236,72,153,0.9))'
                : 'linear-gradient(135deg, rgba(37,99,235,0.85), rgba(59,130,246,0.75))',
              color: '#fff',
              fontWeight: 600,
              letterSpacing: '0.5px',
              boxShadow: autoMaintainEnabled
                ? '0 0 22px rgba(236,72,153,0.45), 0 0 42px rgba(168,85,247,0.35)'
                : '0 0 14px rgba(59,130,246,0.35)',
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
                  ? 'radial-gradient(circle at 20% 20%, rgba(255,255,255,0.35), transparent 55%)'
                  : 'radial-gradient(circle at 20% 20%, rgba(255,255,255,0.2), transparent 50%)',
                mixBlendMode: 'screen',
              }}
            />
          </button>
        )}
        <button
          title={styles.title}
          onClick={onModeCycle}
          style={{
            padding: '8px 18px',
            borderRadius: '10px',
            border: '2px solid',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'all 0.4s ease',
            borderColor: styles.borderColor,
            background: styles.background,
            color: styles.color,
            boxShadow: styles.boxShadow,
            textShadow: styles.textShadow,
          }}
        >
          {styles.label}
        </button>
        <Btn
          onClick={onSubmit}
          style={{ borderColor: '#4aa3ff', background: 'rgba(74,163,255,0.14)' }}
        >
          Send
        </Btn>
      </div>

      <div
        style={{
          flex: '1 1 auto',
          overflowY: 'auto',
          marginTop: 8,
          paddingRight: showDetailedBlocks ? 8 : 0,
        }}
      >
        <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>AI Feed (latest)</div>
        {aiFeed.length === 0 ? (
          <div style={{ fontSize: 12, color: '#999' }}>No AI packages yet.</div>
        ) : (
          aiFeed.map((entry) => (
            <div key={entry.payloadId} style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 12, color: '#444' }}>
                <b>{new Date(entry.time).toLocaleTimeString()}</b> · payload <code>{entry.payloadId}</code>
              </div>
              <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                {entry.items.map((item, idx) => (
                  <li
                    key={`${item.id}_${idx}`}
                    style={{ fontSize: 12, color: '#333', listStyle: 'disc' }}
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
            <div style={{ fontSize: 12, color: '#334155', marginBottom: 8 }}>
              Graph Blocks ({graphBlocksDetailed.length})
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                gap: 12,
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
                    boxShadow: `0 8px 18px ${toRgba(block.color, 0.18)}`,
                  }}
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
                          >
                            <div style={{ fontSize: 11, color: block.color, fontWeight: 600 }}>
                              #{frag.type || 'fragment'}
                            </div>
                            <div style={{ fontSize: 12, color: '#0f172a', lineHeight: 1.45 }}>
                              {frag.text}
                            </div>
                          </button>
                        </li>
                      ))
                    )}
                  </ul>
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
