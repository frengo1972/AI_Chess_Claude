import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Board, { type Arrow } from '../components/Board'
import EvalBar from '../components/EvalBar'
import {
  AnalysisLines,
  CapturedRow,
  MoveList,
  NeuralInsight,
  Panel,
  Stat,
} from '../components/panels'
import * as api from '../api/client'
import type {
  AnalysisResult,
  GameState,
  ModelInfo,
  NeuralEvaluation,
  SystemInfo,
} from '../api/types'

type HumanColour = 'white' | 'black' | 'none'
type OpponentKind = 'neural' | 'stockfish'

interface Setup {
  humanColour: HumanColour
  opponent: OpponentKind
  modelId: string
  simulations: number
  stockfishElo: number
  assistance: boolean
}

const RESULT_TEXT: Record<string, string> = {
  '1-0': 'Vince il Bianco',
  '0-1': 'Vince il Nero',
  '1/2-1/2': 'Patta',
}

const TERMINATION_TEXT: Record<string, string> = {
  checkmate: 'scacco matto',
  stalemate: 'stallo',
  insufficient_material: 'materiale insufficiente',
  fifty_move_rule: 'regola delle 50 mosse',
  threefold_repetition: 'tripla ripetizione',
  resignation: 'abbandono',
  max_plies: 'limite di mosse raggiunto',
}

export default function PlayPage() {
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [game, setGame] = useState<GameState | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [neural, setNeural] = useState<NeuralEvaluation | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoplay, setAutoplay] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [setup, setSetup] = useState<Setup>({
    humanColour: 'white',
    opponent: 'neural',
    modelId: '',
    simulations: 128,
    stockfishElo: 1500,
    assistance: true,
  })

  const analysisToken = useRef(0)

  /* --------------------------------------------------------------- boot */

  useEffect(() => {
    let cancelled = false
    Promise.all([api.getSystem(), api.getModels()])
      .then(([systemInfo, modelList]) => {
        if (cancelled) return
        setSystem(systemInfo)
        setModels(modelList.models)
        setSetup((current) => ({ ...current, modelId: modelList.default }))
      })
      .catch((cause) => setError(String(cause.message ?? cause)))
    return () => {
      cancelled = true
    }
  }, [])

  const startGame = useCallback(
    async (next: Setup) => {
      setBusy(true)
      setError(null)
      setAnalysis(null)
      setNeural(null)
      setAutoplay(false)
      try {
        const response = await api.createGame({
          human_color: next.humanColour,
          assistance_enabled: next.assistance,
          opponent: {
            kind: next.opponent,
            model_id: next.opponent === 'neural' ? next.modelId : null,
            simulations: next.simulations,
            stockfish_elo: next.opponent === 'stockfish' ? next.stockfishElo : null,
            stockfish_movetime_ms: 200,
          },
        })
        setGame(response.game)
      } catch (cause) {
        setError(String((cause as Error).message))
      } finally {
        setBusy(false)
      }
    },
    [],
  )

  // First game as soon as the model list is known.
  useEffect(() => {
    if (setup.modelId && !game) void startGame(setup)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setup.modelId])

  /* ---------------------------------------------------------- assistance */

  const assistanceActive = Boolean(
    game && game.assistance_enabled && game.human_color !== null && system?.stockfish.available,
  )

  useEffect(() => {
    if (!game || !assistanceActive) {
      setAnalysis(null)
      return
    }
    if (game.is_over) return
    const token = ++analysisToken.current
    const timer = window.setTimeout(() => {
      api
        .requestHint(game.id, 14, 3)
        .then((result) => {
          if (token === analysisToken.current) setAnalysis(result)
        })
        .catch(() => {
          if (token === analysisToken.current) setAnalysis(null)
        })
    }, 180)
    return () => window.clearTimeout(timer)
  }, [game, assistanceActive])

  /* ------------------------------------------------------------ autoplay */

  useEffect(() => {
    if (!autoplay || !game || game.is_over || game.human_color !== null) return
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const response = await api.requestEngineMove(game.id)
        if (!cancelled) setGame(response.game)
      } catch (cause) {
        if (!cancelled) {
          setError(String((cause as Error).message))
          setAutoplay(false)
        }
      }
    }, 260)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [autoplay, game])

  /* --------------------------------------------------------------- moves */

  const handleMove = useCallback(
    async (from: string, to: string, promotion?: string) => {
      if (!game || busy) return
      setBusy(true)
      try {
        const response = await api.playMove(game.id, from, to, promotion)
        setGame(response.game)
        setNeural(null)
      } catch (cause) {
        setError(String((cause as Error).message))
      } finally {
        setBusy(false)
      }
    },
    [game, busy],
  )

  const askNeural = useCallback(async () => {
    if (!game) return
    try {
      setNeural(await api.neuralEvaluation(game.id, Math.min(setup.simulations, 400)))
    } catch (cause) {
      setError(String((cause as Error).message))
    }
  }, [game, setup.simulations])

  /* ------------------------------------------------------------ derived */

  const arrows: Arrow[] = useMemo(() => {
    if (!assistanceActive || !analysis || !game || game.is_over) return []
    return analysis.lines.slice(0, 3).map((line, index) => ({
      from: line.moves_uci[0]?.slice(0, 2) ?? '',
      to: line.moves_uci[0]?.slice(2, 4) ?? '',
      kind: index === 0 ? ('primary' as const) : ('secondary' as const),
    }))
      .filter((arrow) => arrow.from && arrow.to)
  }, [analysis, assistanceActive, game])

  const orientation = game?.human_color ?? 'white'
  const lastReport = game?.engine_reports?.[game.engine_reports.length - 1]
  const whiteCp = analysis?.lines[0]?.score.white_perspective_cp ?? null
  const opponentLabel =
    game?.opponent.kind === 'stockfish'
      ? `Stockfish · ${game.opponent.stockfish_elo} Elo`
      : `Rete · ${game?.opponent.model_id ?? '—'}`
  const activeModel = models.find((model) => model.id === game?.opponent.model_id)

  return (
    <div className="play">
      {/* ------------------------------------------------------ left side */}
      <aside className="play__side">
        <Panel
          title="Assistenza · Stockfish"
          aside={
            <span className={`chip ${assistanceActive ? 'chip--live' : 'chip--off'}`}>
              <span className="dot" />
              {assistanceActive ? 'attiva' : 'spenta'}
            </span>
          }
          flush
        >
          {assistanceActive ? (
            <AnalysisLines analysis={analysis} turn={game?.turn ?? 'white'} />
          ) : (
            <p style={{ padding: 12, margin: 0, fontSize: 12 }} className="muted">
              {game?.human_color === null
                ? 'Nelle partite motore contro motore l’assistenza è disabilitata dal server: la rete non deve mai vedere Stockfish.'
                : !system?.stockfish.available
                  ? 'Binario di Stockfish non trovato. Esegui scripts/download_assets.py.'
                  : 'Assistenza disattivata per questa partita.'}
            </p>
          )}
        </Panel>

        <Panel title="Materiale" flush>
          <div style={{ padding: '10px 12px', display: 'grid', gap: 8 }}>
            <CapturedRow
              pieces={game?.captured.black ?? []}
              colour="b"
              delta={Math.max(0, game?.material.difference ?? 0)}
            />
            <CapturedRow
              pieces={game?.captured.white ?? []}
              colour="w"
              delta={Math.max(0, -(game?.material.difference ?? 0))}
            />
          </div>
          <div className="stat-grid">
            <Stat label="Bianco" value={game?.material.white ?? 0} />
            <Stat label="Nero" value={game?.material.black ?? 0} />
            <Stat label="Mossa" value={game?.fullmove_number ?? 1} />
            <Stat label="Regola 50" value={`${game?.halfmove_clock ?? 0}/100`} />
          </div>
        </Panel>

        {activeModel && (
          <Panel title="Rete avversaria" flush>
            <div className="stat-grid">
              <Stat label="Parametri" value={formatCount(activeModel.parameters)} />
              <Stat label="Dimensione" value={`${activeModel.size_mb.toFixed(1)} MB`} />
              <Stat label="Blocchi × filtri" value={`${activeModel.residual_blocks}×${activeModel.filters}`} />
              <Stat label="Storia (T)" value={activeModel.history_length} />
              <Stat label="Partite" value={formatCount(activeModel.games_total ?? 0)} />
              <Stat
                label="Elo relativo"
                value={activeModel.elo !== null ? Math.round(activeModel.elo) : '—'}
              />
            </div>
          </Panel>
        )}
      </aside>

      {/* -------------------------------------------------------- eval bar */}
      <div className="play__evalcol">
        <EvalBar
          whiteCentipawns={whiteCp}
          label={analysis?.lines[0]?.score.text ?? null}
          orientation={orientation}
          active={assistanceActive && !!analysis}
        />
      </div>

      {/* ----------------------------------------------------------- board */}
      <Board
        fen={game?.fen ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'}
        legalMoves={game && game.human_to_move && !game.is_over ? game.legal_moves : {}}
        orientation={orientation}
        lastMove={game?.last_move ?? null}
        checkers={game?.checkers ?? []}
        interactive={Boolean(game && game.human_to_move && !game.is_over && !busy)}
        arrows={arrows}
        onMove={handleMove}
      />

      {/* ----------------------------------------------------- right side */}
      <aside className="play__side">
        {error && (
          <div className="banner banner--error">
            {error}
            <button
              type="button"
              className="btn btn--small"
              style={{ marginLeft: 8 }}
              onClick={() => setError(null)}
            >
              chiudi
            </button>
          </div>
        )}

        {game?.is_over && (
          <div className="result">
            <div className="result__headline">
              {RESULT_TEXT[game.result ?? ''] ?? game.result}
            </div>
            <div className="result__detail">
              {TERMINATION_TEXT[game.termination ?? ''] ?? game.termination} ·{' '}
              {game.moves_san.length} semimosse
            </div>
          </div>
        )}

        <Panel
          title="Partita"
          aside={<span className="chip">{opponentLabel}</span>}
          flush
        >
          <MoveList moves={game?.moves_san ?? []} />
        </Panel>

        <Panel
          title="Cosa pensa la rete"
          aside={
            <button type="button" className="btn btn--small" onClick={askNeural}>
              analizza
            </button>
          }
        >
          <NeuralInsight evaluation={neural} lastReport={lastReport} />
        </Panel>

        <Panel title="Nuova partita">
          <div className="setup">
            <div className="field">
              <span className="field__label">Giochi con</span>
              <div className="setup__segment">
                {(['white', 'black', 'none'] as HumanColour[]).map((option) => (
                  <button
                    key={option}
                    type="button"
                    className="setup__option"
                    aria-pressed={setup.humanColour === option}
                    onClick={() => setSetup({ ...setup, humanColour: option })}
                  >
                    {option === 'white' ? 'Bianco' : option === 'black' ? 'Nero' : 'Guardo'}
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <span className="field__label">Avversario</span>
              <div className="setup__segment">
                {(['neural', 'stockfish'] as OpponentKind[]).map((option) => (
                  <button
                    key={option}
                    type="button"
                    className="setup__option"
                    aria-pressed={setup.opponent === option}
                    onClick={() => setSetup({ ...setup, opponent: option })}
                  >
                    {option === 'neural' ? 'Rete neurale' : 'Stockfish'}
                  </button>
                ))}
              </div>
            </div>

            {setup.opponent === 'neural' ? (
              <>
                <label className="field">
                  <span className="field__label">Modello</span>
                  <select
                    value={setup.modelId}
                    onChange={(event) => setSetup({ ...setup, modelId: event.target.value })}
                  >
                    {models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span className="field__label">Simulazioni MCTS</span>
                  <div className="setup__range">
                    <input
                      type="range"
                      min={1}
                      max={800}
                      step={1}
                      value={setup.simulations}
                      onChange={(event) =>
                        setSetup({ ...setup, simulations: Number(event.target.value) })
                      }
                    />
                    <output>{setup.simulations}</output>
                  </div>
                  <span className="stat__hint">
                    {setup.simulations === 1 ? 'policy pura, nessuna ricerca' : 'più alto = più forte e più lento'}
                  </span>
                </label>
              </>
            ) : (
              <label className="field">
                <span className="field__label">Forza Stockfish</span>
                <div className="setup__range">
                  <input
                    type="range"
                    min={system?.stockfish.min_elo ?? 1320}
                    max={Math.min(system?.stockfish.max_elo ?? 3190, 2800)}
                    step={10}
                    value={setup.stockfishElo}
                    onChange={(event) =>
                      setSetup({ ...setup, stockfishElo: Number(event.target.value) })
                    }
                  />
                  <output>{setup.stockfishElo}</output>
                </div>
              </label>
            )}

            <label className="switch">
              <input
                type="checkbox"
                checked={setup.assistance}
                onChange={(event) => setSetup({ ...setup, assistance: event.target.checked })}
              />
              <span>Assistenza Stockfish per me</span>
            </label>

            <button
              type="button"
              className="btn btn--primary"
              disabled={busy}
              onClick={() => void startGame(setup)}
            >
              Inizia partita
            </button>
          </div>
        </Panel>

        <Panel title="Comandi">
          <div className="row" style={{ flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn--small"
              disabled={!game || game.moves_uci.length === 0 || busy}
              onClick={async () => {
                if (!game) return
                setGame((await api.undoMoves(game.id, 2)).game)
              }}
            >
              Annulla mossa
            </button>
            <button
              type="button"
              className="btn btn--small btn--danger"
              disabled={!game || game.is_over || game.human_color === null}
              onClick={async () => {
                if (!game) return
                setGame((await api.resignGame(game.id)).game)
              }}
            >
              Abbandono
            </button>
            {game?.human_color === null && (
              <button
                type="button"
                className="btn btn--small"
                disabled={game.is_over}
                onClick={() => setAutoplay((value) => !value)}
              >
                {autoplay ? 'Pausa' : 'Gioca da solo'}
              </button>
            )}
            <button
              type="button"
              className="btn btn--small"
              disabled={!game}
              onClick={() => {
                if (game) void navigator.clipboard?.writeText(game.pgn)
              }}
            >
              Copia PGN
            </button>
          </div>
        </Panel>
      </aside>
    </div>
  )
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)} M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)} k`
  return String(value)
}
