import type {
  AnalysisResult,
  BenchmarkResult,
  DocSummary,
  GameResponse,
  IterationRow,
  ModelInfo,
  NeuralEvaluation,
  PresetInfo,
  SystemInfo,
  TrainingRun,
  TrainingStatus,
  WatchSettings,
  WatchSnapshot,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, response.status)
  }
  return (await response.json()) as T
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined })

// ------------------------------------------------------------------ system

export const getSystem = () => request<SystemInfo>('/system')
export const getModels = () =>
  request<{ models: ModelInfo[]; default: string }>('/models')
export const reloadModels = () => post<{ reloaded: boolean }>('/models/reload')

// ------------------------------------------------------------------- games

export interface NewGameOptions {
  human_color: 'white' | 'black' | 'none'
  assistance_enabled: boolean
  opponent: {
    kind: 'neural' | 'stockfish'
    model_id?: string | null
    simulations?: number
    temperature?: number
    stockfish_elo?: number | null
    stockfish_skill?: number | null
    stockfish_movetime_ms?: number
  }
}

export const createGame = (options: NewGameOptions) =>
  post<GameResponse>('/game', options)

export const getGame = (id: string) => request<{ game: GameResponse['game'] }>(`/game/${id}`)

export const playMove = (id: string, from: string, to: string, promotion?: string) =>
  post<GameResponse>(`/game/${id}/move`, { from, to, promotion: promotion ?? null })

export const requestEngineMove = (id: string) =>
  post<GameResponse>(`/game/${id}/engine-move`)

export const undoMoves = (id: string, plies = 2) =>
  post<GameResponse>(`/game/${id}/undo`, { plies })

export const resignGame = (id: string) => post<GameResponse>(`/game/${id}/resign`)

export const neuralEvaluation = (id: string, simulations = 64) =>
  post<NeuralEvaluation>(`/game/${id}/neural-evaluation`, {
    game_id: id,
    simulations,
  })

// ------------------------------------------- analysis (human assistance only)

export const analysePosition = (fen: string, depth = 14, multipv = 3) =>
  post<AnalysisResult>('/analysis/position', { fen, depth, multipv })

export const requestHint = (gameId: string, depth = 14, multipv = 3) =>
  post<AnalysisResult>('/analysis/hint', { game_id: gameId, depth, multipv })

// ---------------------------------------------------------------- training

export const getPresets = () =>
  request<{ presets: Record<string, PresetInfo>; default: string }>('/training/presets')

export const getTrainingStatus = (runId?: string) =>
  request<TrainingStatus>(`/training/status${runId ? `?run_id=${runId}` : ''}`)

export const getTrainingHistory = (runId: string) =>
  request<{ run_id: string; iterations: IterationRow[] }>(
    `/training/history?run_id=${runId}`,
  )

export const getTrainingEvents = (runId: string, limit = 60) =>
  request<{ events: Array<{ timestamp: number; level: string; message: string }> }>(
    `/training/events?run_id=${runId}&limit=${limit}`,
  )

export const getTrainingRuns = () =>
  request<{ runs: TrainingRun[] }>('/training/runs')

export const getSampleGames = (runId: string, limit = 6) =>
  request<{ games: Array<{ id: number; iteration: number; result: string; plies: number; termination: string; pgn: string }> }>(
    `/training/games?run_id=${runId}&limit=${limit}`,
  )

export const getWatch = (runId?: string) =>
  request<WatchSnapshot>(`/training/watch${runId ? `?run_id=${runId}` : ''}`)

export const setWatchSettings = (
  runId: string | null,
  settings: Partial<WatchSettings>,
) =>
  post<{ run_id: string; settings: WatchSettings }>('/training/watch/settings', {
    run_id: runId,
    ...settings,
  })

export interface StartTrainingOptions {
  preset: string
  name?: string
  iterations?: number
  games_per_iteration?: number
  simulations?: number
  workers?: number
  device?: 'auto' | 'cpu' | 'cuda'
  resume_run_id?: string
  overrides?: Record<string, unknown>
}

export const startTraining = (options: StartTrainingOptions) =>
  post<{ run_id: string; pid: number; log: string }>('/training/start', options)

export const stopTraining = (runId?: string, force = false) =>
  post<{ run_id: string; stopping: boolean }>('/training/stop', {
    run_id: runId ?? null,
    force,
  })

// --------------------------------------------------------------- benchmark

export const runBenchmark = (
  modelId: string,
  games: number,
  stockfishElo: number,
  movetimeMs = 50,
) =>
  post<BenchmarkResult>('/benchmark', {
    model_id: modelId,
    games,
    stockfish_elo: stockfishElo,
    movetime_ms: movetimeMs,
  })

// -------------------------------------------------------------------- docs

export const getDocs = () => request<{ documents: DocSummary[] }>('/docs')
export const getDoc = (slug: string) =>
  request<{ slug: string; title: string; markdown: string }>(`/docs/${slug}`)

// --------------------------------------------------------------- websocket

function openSocket<T>(path: string, onMessage: (payload: T) => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const socket = new WebSocket(`${protocol}://${window.location.host}${BASE}${path}`)
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data) as T)
    } catch {
      /* ignore malformed frames */
    }
  }
  return socket
}

export function openTrainingSocket(
  onMessage: (status: TrainingStatus) => void,
  runId?: string,
): WebSocket {
  return openSocket(`/training/ws${runId ? `?run_id=${runId}` : ''}`, onMessage)
}

/** Boards arrive on their own socket: faster cadence, and only while watched. */
export function openWatchSocket(
  onMessage: (snapshot: WatchSnapshot) => void,
  runId?: string,
  intervalMs = 500,
): WebSocket {
  const query = `?interval_ms=${intervalMs}${runId ? `&run_id=${runId}` : ''}`
  return openSocket(`/training/watch/ws${query}`, onMessage)
}
