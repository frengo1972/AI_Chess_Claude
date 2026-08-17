import type { Colour } from '../api/types'

interface EvalBarProps {
  /** Centipawns from White's point of view; `null` while unknown. */
  whiteCentipawns: number | null
  /** Text as the engine reports it, from the side-to-move's point of view. */
  label: string | null
  orientation: Colour
  active: boolean
}

/**
 * The classic vertical advantage bar. The mapping from centipawns to bar height
 * is a logistic curve, not linear: the difference between +1 and +2 pawns
 * matters far more than between +8 and +9.
 */
export default function EvalBar({
  whiteCentipawns,
  label,
  orientation,
  active,
}: EvalBarProps) {
  const centipawns = Math.max(-2000, Math.min(2000, whiteCentipawns ?? 0))
  const whiteShare = active ? 1 / (1 + Math.pow(10, -centipawns / 400)) : 0.5
  const percent = Math.max(3, Math.min(97, whiteShare * 100))
  const whiteAtBottom = orientation === 'white'

  return (
    <div
      className="evalbar"
      title={active ? `Valutazione Stockfish: ${label ?? '—'}` : 'Assistenza disattivata'}
      aria-label="Barra di valutazione"
    >
      <div
        className="evalbar__white"
        style={
          whiteAtBottom
            ? { height: `${percent}%`, bottom: 0, top: 'auto' }
            : { height: `${100 - percent}%`, top: 0, bottom: 'auto' }
        }
      />
      <div className="evalbar__mid" />
      {active && label && (
        <span
          className={`evalbar__label evalbar__label--${
            (whiteAtBottom ? centipawns >= 0 : centipawns < 0) ? 'bottom' : 'top'
          }`}
        >
          {label}
        </span>
      )}
    </div>
  )
}
