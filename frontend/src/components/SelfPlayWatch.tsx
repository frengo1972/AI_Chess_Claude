import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Board from './Board'
import * as api from '../api/client'
import type { WatchBoard, WatchFrame, WatchSettings, WatchSnapshot } from '../api/types'
import './watch.css'

/**
 * Live view of the self-play games the trainer is generating.
 *
 * Two independent speeds are at play here, and keeping them separate is the
 * whole point of the component:
 *
 * * the *trainer's* speed -- self-play runs as fast as the machine allows, and
 *   asking it to pause between moves (`move_delay_ms`) really does slow the run
 *   down;
 * * the *viewer's* speed -- each worker publishes a rolling window of recent
 *   positions, so we can step through them at a human pace while the trainer
 *   keeps going. A board that falls too far behind jumps to the oldest position
 *   still available rather than desynchronising for good.
 */

const NO_MOVES: Record<string, string[]> = {}
const NO_CHECKERS: string[] = []
const noop = () => undefined

const SPEEDS: Array<{ label: string; movesPerSecond: number }> = [
  { label: 'istantaneo', movesPerSecond: 0 },
  { label: '½ al secondo', movesPerSecond: 0.5 },
  { label: '1 al secondo', movesPerSecond: 1 },
  { label: '2 al secondo', movesPerSecond: 2 },
  { label: '4 al secondo', movesPerSecond: 4 },
]

const DELAYS: Array<{ label: string; ms: number }> = [
  { label: 'nessuna', ms: 0 },
  { label: '200 ms', ms: 200 },
  { label: '500 ms', ms: 500 },
  { label: '1 s', ms: 1000 },
  { label: '2 s', ms: 2000 },
]

const COUNTS = [1, 2, 4, 8]

const RESULT_LABEL: Record<string, string> = {
  '1-0': 'vince il Bianco',
  '0-1': 'vince il Nero',
  '1/2-1/2': 'patta',
}

const TERMINATION_LABEL: Record<string, string> = {
  checkmate: 'scacco matto',
  stalemate: 'stallo',
  insufficient_material: 'materiale insufficiente',
  fifty_moves: 'regola delle 50 mosse',
  repetition: 'ripetizione',
  resignation: 'abbandono',
  max_plies: 'limite di semimosse',
  no_legal_moves: 'nessuna mossa legale',
}

interface SelfPlayWatchProps {
  runId: string | null
  /** Reported upwards so a newly started run can be launched already watching. */
  onSettings: (settings: WatchSettings) => void
}

export default function SelfPlayWatch({ runId, onSettings }: SelfPlayWatchProps) {
  const [open, setOpen] = useState(false)
  const [snapshot, setSnapshot] = useState<WatchSnapshot | null>(null)
  const [pending, setPending] = useState<WatchSettings | null>(null)
  const [speed, setSpeed] = useState(1)
  const [count, setCount] = useState(4)
  const [error, setError] = useState<string | null>(null)

  const settings: WatchSettings =
    pending ?? snapshot?.settings ?? { enabled: false, move_delay_ms: 0 }

  /* ------------------------------------------------------------- transport */

  useEffect(() => {
    if (!open) {
      setSnapshot(null)
      return
    }
    const socket = api.openWatchSocket(setSnapshot, runId ?? undefined)
    return () => socket.close()
  }, [open, runId])

  // Drop the optimistic value once the trainer's own settings agree with it.
  useEffect(() => {
    const live = snapshot?.settings
    if (!live || !pending) return
    if (live.enabled === pending.enabled && live.move_delay_ms === pending.move_delay_ms) {
      setPending(null)
    }
  }, [snapshot, pending])

  const push = useCallback(
    async (next: WatchSettings) => {
      setPending(next)
      onSettings(next)
      setError(null)
      try {
        await api.setWatchSettings(runId ?? null, next)
      } catch (cause) {
        setPending(null)
        setError(String((cause as Error).message))
      }
    },
    [runId, onSettings],
  )

  /* --------------------------------------------------------------- pacing */

  const boards = useMemo(() => snapshot?.boards ?? [], [snapshot])
  const visible = useMemo(() => boards.slice(0, count), [boards, count])
  const visibleRef = useRef<WatchBoard[]>(visible)
  visibleRef.current = visible

  const [cursors, setCursors] = useState<Record<string, number>>({})

  useEffect(() => {
    if (!open || speed === 0) return
    const timer = window.setInterval(() => {
      setCursors((current) => {
        const next: Record<string, number> = {}
        let changed = false
        for (const board of visibleRef.current) {
          const oldest = board.frames[0]?.ply ?? board.ply
          const shown = current[board.game_uid]
          if (shown === undefined) {
            next[board.game_uid] = oldest
            changed = true
          } else if (shown >= board.ply) {
            next[board.game_uid] = shown
          } else {
            // One move on, unless the window has already moved past us.
            next[board.game_uid] = Math.max(shown + 1, oldest)
            changed = true
          }
        }
        // Boards that disappeared take their cursor with them.
        if (!changed && Object.keys(next).length === Object.keys(current).length) {
          return current
        }
        return next
      })
    }, Math.max(80, 1000 / speed))
    return () => window.clearInterval(timer)
  }, [open, speed])

  /* --------------------------------------------------------------- render */

  const phase = snapshot?.phase ?? 'idle'
  const publishing = Boolean(snapshot?.settings.enabled)
  const waiting = publishing && phase !== 'selfplay' && boards.length === 0

  return (
    <section className="panel watch">
      <header className="panel__head">
        <h2 className="panel__title">Self-play in diretta</h2>
        <div className="watch__head">
          {open && (
            <span className={`chip ${publishing ? 'chip--live' : 'chip--off'}`}>
              <span className="dot" />
              {publishing ? `fase: ${phase}` : 'non pubblica'}
            </span>
          )}
          <button type="button" className="btn btn--small" onClick={() => setOpen(!open)}>
            {open ? 'Chiudi' : 'Apri'}
          </button>
        </div>
      </header>

      {open && (
        <div className="panel__body">
          <div className="watch__controls">
            <label className="watch__switch">
              <input
                type="checkbox"
                checked={settings.enabled}
                onChange={(event) =>
                  void push({ ...settings, enabled: event.target.checked })
                }
              />
              <span>Pubblica le partite</span>
            </label>

            <label className="field">
              <span className="field__label">Riproduzione</span>
              <select
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value))}
              >
                {SPEEDS.map((option) => (
                  <option key={option.label} value={option.movesPerSecond}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="field__label">Pausa del trainer</span>
              <select
                value={settings.move_delay_ms}
                onChange={(event) =>
                  void push({ ...settings, move_delay_ms: Number(event.target.value) })
                }
              >
                {DELAYS.map((option) => (
                  <option key={option.ms} value={option.ms}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="field__label">Scacchiere</span>
              <select value={count} onChange={(event) => setCount(Number(event.target.value))}>
                {COUNTS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <p className="muted watch__note">
            La riproduzione rallenta solo ciò che vedi. La <em>pausa del trainer</em> ferma
            davvero i worker fra una mossa e l’altra: utile per seguire una partita, ma
            rallenta l’addestramento in proporzione.
          </p>

          {error && <div className="banner banner--error">{error}</div>}

          {!settings.enabled && (
            <p className="muted watch__empty">
              Attiva <strong>Pubblica le partite</strong> per vedere le partite di self-play
              in corso. Con l’opzione spenta il trainer non scrive nulla.
            </p>
          )}

          {waiting && (
            <p className="muted watch__empty">
              Nessuna partita pubblicata al momento: il run è nella fase{' '}
              <strong>{phase}</strong>. Le scacchiere compaiono quando riprende il
              self-play.
            </p>
          )}

          {visible.length > 0 && (
            <div className="watch__grid" data-count={visible.length}>
              {visible.map((board) => (
                <WatchTile
                  key={board.slot}
                  board={board}
                  frame={frameToShow(board, cursors[board.game_uid], speed)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

/* --------------------------------------------------------------------- tile */

function WatchTile({ board, frame }: { board: WatchBoard; frame: WatchFrame }) {
  const atLatest = frame.ply >= board.ply
  const stale = board.age_seconds > 8
  const behind = board.ply - frame.ply
  const turn = (frame.fen.split(' ')[1] ?? 'w') === 'w' ? 'white' : 'black'
  // The published value belongs to the side that just moved; show it as White's.
  const whiteValue = frame.value === null ? null : turn === 'white' ? -frame.value : frame.value

  return (
    <article className={`watch__tile${stale ? ' watch__tile--stale' : ''}`}>
      <header className="watch__tilehead">
        <span className="watch__slot">worker {board.slot + 1}</span>
        <span className="muted">
          iter {board.iteration} · partita {board.game_index + 1}
        </span>
      </header>

      <div className="watch__board">
        <Board
          fen={frame.fen}
          legalMoves={NO_MOVES}
          orientation="white"
          lastMove={frame.last_move}
          checkers={NO_CHECKERS}
          interactive={false}
          onMove={noop}
        />
      </div>

      <ValueMeter value={whiteValue} />

      <footer className="watch__tilefoot">
        <span className="numeric">
          mossa {Math.floor(frame.ply / 2) + 1}
          {behind > 0 && <span className="muted"> · −{behind}</span>}
        </span>
        <span className="numeric muted">
          {board.material.difference === 0
            ? 'materiale pari'
            : `materiale ${board.material.difference > 0 ? '+' : ''}${board.material.difference}`}
        </span>
        {board.finished && atLatest ? (
          <span className="watch__result">
            {RESULT_LABEL[board.result ?? ''] ?? board.result} ·{' '}
            {TERMINATION_LABEL[board.termination ?? ''] ?? board.termination}
          </span>
        ) : (
          <span className="muted">{turn === 'white' ? 'muove il Bianco' : 'muove il Nero'}</span>
        )}
      </footer>
    </article>
  )
}

/**
 * The network's own value for the position, White at the right. Deliberately not
 * an eval bar in centipawns: this number is the net's opinion, not an engine's.
 */
function ValueMeter({ value }: { value: number | null }) {
  const clamped = Math.max(-1, Math.min(1, value ?? 0))
  const percent = ((clamped + 1) / 2) * 100
  return (
    <div className="watch__meter" title="Valutazione della rete, dal punto di vista del Bianco">
      <span className="watch__metertrack">
        <span className="watch__meterfill" style={{ width: `${percent}%` }} />
        <span className="watch__metermid" />
      </span>
      <span className="watch__metervalue numeric">
        {value === null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(2)}`}
      </span>
    </div>
  )
}

/** The position to render: the paced cursor if we have one, else the newest. */
function frameToShow(
  board: WatchBoard,
  cursor: number | undefined,
  speed: number,
): WatchFrame {
  const latest: WatchFrame = {
    ply: board.ply,
    fen: board.fen,
    last_move: board.last_move,
    value: board.value,
  }
  if (speed === 0 || cursor === undefined) return latest
  return board.frames.find((frame) => frame.ply === cursor) ?? board.frames[0] ?? latest
}
