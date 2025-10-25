export type PromptMode = 'full' | 'light' | 'vision'

export type ColorName =
  | 'black' | 'blue' | 'green' | 'grey'
  | 'light-blue' | 'light-green' | 'light-red' | 'light-violet'
  | 'orange' | 'red' | 'violet' | 'white' | 'yellow'

export type AIStrokeV11 = {
  id: string
  tool: 'pen' | 'line' | 'bezier' | 'rect' | 'ellipse' | 'poly' | 'eraser' | 'text' | 'edit' | string
  // points: [x, y, t?, pressure?]
  points: Array<[number, number, number?, number?]>
  style?: { size?: 's'|'m'|'l'|'xl'; color?: ColorName; opacity?: number }
  meta?: Record<string, any>
}

export type AIStrokePayload = {
  canvas?: { width?: number; height?: number; viewport?: [number, number, number, number] }
  strokes: AIStrokeV11[]
  intent?: 'complete' | 'hint' | 'alt'
  replace?: string[]
  version?: number
}
