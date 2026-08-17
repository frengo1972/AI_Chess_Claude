export type Colour = 'white' | 'black'

export interface EngineReport {
  source: 'neural' | 'stockfish'
  uci: string
  san: string
  evaluation?: number
  simulations?: number
  nodes?: number
  think_ms?: number
  top_moves?: TopMove[]
  principal_variation?: string[]
  model_id?: string
  elo?: number | null
  skill?: number | null
}

export interface TopMove {
  uci: string
  san: string
  probability: number
  visits: number
}

export interface GameState {
  id: string
  fen: string
  turn: Colour
  human_color: Colour | null
  human_to_move: boolean
  opponent: {
    kind: 'neural' | 'stockfish'
    model_id: string | null
    simulations: number
    stockfish_elo: number | null
    stockfish_skill: number | null
  }
  assistance_enabled: boolean
  legal_moves: Record<string, string[]>
  promotion_moves: string[]
  moves_uci: string[]
  moves_san: string[]
  last_move: string | null
  in_check: boolean
  checkers: string[]
  is_over: boolean
  result: string | null
  termination: string | null
  material: { white: number; black: number; difference: number }
  captured: { white: string[]; black: string[] }
  halfmove_clock: number
  fullmove_number: number
  engine_reports: EngineReport[]
  pgn: string
  updated_at: number
}

export interface GameResponse {
  game: GameState
  engine_move?: EngineReport | null
}

export interface ScoreView {
  centipawns: number | null
  mate_in: number | null
  white_perspective_cp: number | null
  win_probability: number
  text: string
}

export interface AnalysisLine {
  rank: number
  score: ScoreView
  moves_uci: string[]
  moves_san: string[]
  depth: number
  nodes: number
}

export interface AnalysisResult {
  fen: string
  depth: number
  time_ms: number
  engine: string
  lines: AnalysisLine[]
  human_to_move?: boolean
  for_color?: Colour
}

export interface NeuralEvaluation {
  value: number
  simulations: number
  top_moves: TopMove[]
  model_id: string
  terminal: string | null
}

export interface ModelInfo {
  id: string
  label: string
  run_id: string | null
  path: string | null
  iteration: number
  parameters: number
  size_mb: number
  history_length: number
  residual_blocks: number
  filters: number
  elo: number | null
  games_total: number | null
  trained: boolean
}

export interface SystemInfo {
  python: string
  platform: string
  torch: string
  cuda_available: boolean
  cuda_device: string | null
  device: string
  stockfish: {
    available: boolean
    path: string | null
    name: string | null
    min_elo: number
    max_elo: number
  }
  checkpoint_dir: string
}

export interface IterationRow {
  run_id: string
  iteration: number
  timestamp: number
  games_total: number | null
  positions_total: number | null
  buffer_size: number | null
  policy_loss: number | null
  value_loss: number | null
  total_loss: number | null
  learning_rate: number | null
  avg_plies: number | null
  draw_rate: number | null
  white_win_rate: number | null
  black_win_rate: number | null
  resign_rate: number | null
  policy_entropy: number | null
  avg_root_value: number | null
  selfplay_seconds: number | null
  train_seconds: number | null
  games_per_minute: number | null
  positions_per_second: number | null
  arena_score: number | null
  elo: number | null
  elo_delta: number | null
  promoted: number | null
  benchmark_elo: number | null
  benchmark_score: number | null
  value_gap: number | null
  wall_seconds: number | null
  extra: Record<string, unknown>
}

/** What a stopped run would pick up from, written by the trainer every iteration. */
export interface SavedRunState {
  run_id?: string
  iteration?: number
  games_total?: number
  positions_total?: number
  elo?: number
  wall_seconds?: number
  sessions?: number
  device?: string
  updated_at?: number
}

export interface TrainingRun {
  id: string
  name: string
  preset: string | null
  created_at: number
  updated_at: number
  status: 'running' | 'finished' | 'stopped' | 'failed'
  config: Record<string, any>
  network: Record<string, any>
  notes: string
  resumable?: boolean
  has_trainer_state?: boolean
  saved_state?: SavedRunState | null
}

export interface TrainingStatus {
  active: boolean
  process_alive?: boolean
  /** Marked running, but nothing has updated the run for a long time. */
  stale?: boolean
  status_age_seconds?: number | null
  run: TrainingRun | null
  phase: string
  live?: Record<string, any>
  last_iteration?: IterationRow | null
  stop_requested?: boolean
}

/* --------------------------------------------- watching self-play in progress */

export interface WatchFrame {
  ply: number
  fen: string
  last_move: string | null
  /** Search value of the move that produced this position, mover's POV. */
  value: number | null
}

export interface WatchBoard {
  slot: number
  run_id: string | null
  iteration: number
  game_index: number
  game_uid: string
  ply: number
  turn: Colour
  fen: string
  last_move: string | null
  value: number | null
  in_check: boolean
  material: { white: number; black: number; difference: number }
  top_moves: Array<{ uci: string; probability: number }>
  frames: WatchFrame[]
  finished: boolean
  result: string | null
  termination: string | null
  resigned: boolean
  updated_at: number
  age_seconds: number
}

export interface WatchSettings {
  enabled: boolean
  move_delay_ms: number
}

export interface WatchSnapshot {
  run_id: string | null
  phase: string
  active: boolean
  settings: WatchSettings
  boards: WatchBoard[]
}

export interface PresetSummary {
  blocks: number
  filters: number
  history: number
  simulations: number
  games_per_iteration: number
  parameters: number
  size_mb: number
}

export interface PresetInfo {
  config: Record<string, any>
  network: Record<string, any>
  summary: PresetSummary
}

export interface BenchmarkResult {
  games: number
  wins: number
  draws: number
  losses: number
  score: number
  opponent_elo: number
  estimated_elo: number
  elo_error_95: number
  duration_seconds: number
  model_id: string
  detail: Array<{
    index: number
    network_white: boolean
    result: string
    outcome: string
    termination: string
    plies: number
  }>
}

export interface DocSummary {
  slug: string
  title: string
  summary: string
  words: string
}
