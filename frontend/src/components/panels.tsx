import './panels.css'
import type { AnalysisResult, GameState, NeuralEvaluation, TopMove } from '../api/types'

/* --------------------------------------------------------------- generic */

export function Panel({
  title,
  aside,
  flush,
  children,
}: {
  title: string
  aside?: React.ReactNode
  flush?: boolean
  children: React.ReactNode
}) {
  return (
    <section className="panel">
      <header className="panel__head">
        <h2 className="panel__title">{title}</h2>
        {aside}
      </header>
      <div className={`panel__body${flush ? ' panel__body--flush' : ''}`}>{children}</div>
    </section>
  )
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: React.ReactNode
  hint?: string
}) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      {hint && <span className="stat__hint">{hint}</span>}
    </div>
  )
}

/* -------------------------------------------------------- Stockfish lines */

export function AnalysisLines({
  analysis,
  turn,
}: {
  analysis: AnalysisResult | null
  turn: 'white' | 'black'
}) {
  if (!analysis || !analysis.lines.length) {
    return <p className="muted" style={{ padding: '12px', margin: 0, fontSize: 12 }}>
      Nessuna analisi disponibile.
    </p>
  }
  return (
    <div className="lines">
      {analysis.lines.map((line) => (
        <div key={line.rank} className={`line${line.rank === 1 ? ' line--best' : ''}`}>
          <span className={`line__score line__score--${turn}`}>{line.score.text}</span>
          <span className="line__moves">
            <strong>{line.moves_san[0] ?? '—'}</strong>{' '}
            {line.moves_san.slice(1, 7).join(' ')}
          </span>
        </div>
      ))}
    </div>
  )
}

/* --------------------------------------------------------- neural network */

export function NeuralInsight({
  evaluation,
  lastReport,
}: {
  evaluation: NeuralEvaluation | null
  lastReport: GameState['engine_reports'][number] | undefined
}) {
  const value = evaluation?.value ?? lastReport?.evaluation ?? null
  const moves: TopMove[] = evaluation?.top_moves ?? lastReport?.top_moves ?? []
  const usedSearch = (evaluation?.simulations ?? lastReport?.simulations ?? 0) > 1

  return (
    <div className="stack">
      <div className="stat-grid" style={{ borderRadius: 'var(--radius)', overflow: 'hidden' }}>
        <Stat
          label="Valutazione rete"
          value={value === null ? '—' : value.toFixed(3)}
          hint="+1 vince chi muove"
        />
        <Stat
          label={usedSearch ? 'Simulazioni' : 'Modalità'}
          value={usedSearch ? (evaluation?.simulations ?? lastReport?.simulations ?? 0) : 'policy'}
          hint={usedSearch ? 'MCTS per mossa' : 'nessuna ricerca'}
        />
        <Stat label="Nodi creati" value={lastReport?.nodes?.toLocaleString('it-IT') ?? '—'} />
        <Stat
          label="Tempo"
          value={lastReport?.think_ms !== undefined ? `${lastReport.think_ms} ms` : '—'}
        />
      </div>

      {moves.length > 0 && (
        <div className="topmoves">
          {moves.map((move) => (
            <div key={move.uci} className="topmove">
              <span className="topmove__san">{move.san}</span>
              <span className="topmove__track">
                <span
                  className="topmove__fill"
                  style={{ width: `${Math.max(2, move.probability * 100)}%` }}
                />
              </span>
              <span className="topmove__value">{(move.probability * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {lastReport?.principal_variation?.length ? (
        <p className="muted" style={{ margin: 0, fontSize: 12 }}>
          Variante: {lastReport.principal_variation.join(' ')}
        </p>
      ) : null}
    </div>
  )
}

/* ------------------------------------------------------------- move list */

export function MoveList({ moves }: { moves: string[] }) {
  if (!moves.length) {
    return (
      <div className="movelist">
        <p className="movelist__empty">La partita non è ancora iniziata.</p>
      </div>
    )
  }
  const rows: Array<[number, string, string | undefined]> = []
  for (let index = 0; index < moves.length; index += 2) {
    rows.push([index / 2 + 1, moves[index], moves[index + 1]])
  }
  return (
    <div className="movelist">
      {rows.map(([number, white, black], rowIndex) => (
        <MoveRow
          key={number}
          number={number}
          white={white}
          black={black}
          currentWhite={rowIndex * 2 === moves.length - 1}
          currentBlack={rowIndex * 2 + 1 === moves.length - 1}
        />
      ))}
    </div>
  )
}

function MoveRow({
  number,
  white,
  black,
  currentWhite,
  currentBlack,
}: {
  number: number
  white: string
  black?: string
  currentWhite: boolean
  currentBlack: boolean
}) {
  return (
    <>
      <span className="movelist__number">{number}.</span>
      <span className={`movelist__move${currentWhite ? ' movelist__move--current' : ''}`}>
        {white}
      </span>
      <span className={`movelist__move${currentBlack ? ' movelist__move--current' : ''}`}>
        {black ?? ''}
      </span>
    </>
  )
}

/* -------------------------------------------------------------- captured */

const PIECE_ORDER = ['Q', 'R', 'B', 'N', 'P']

export function CapturedRow({
  pieces,
  colour,
  delta,
}: {
  pieces: string[]
  colour: 'w' | 'b'
  delta: number
}) {
  const sorted = [...pieces].sort(
    (a, b) => PIECE_ORDER.indexOf(a) - PIECE_ORDER.indexOf(b),
  )
  return (
    <div className="captured">
      {sorted.map((piece, index) => (
        <img key={`${piece}${index}`} src={`/pieces/cburnett/${colour}${piece}.svg`} alt={piece} />
      ))}
      {delta > 0 && <span className="captured__delta">+{delta}</span>}
    </div>
  )
}
