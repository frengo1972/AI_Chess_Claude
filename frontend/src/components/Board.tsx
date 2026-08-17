import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import './Board.css'
import type { Colour } from '../api/types'

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const
const RANKS = ['1', '2', '3', '4', '5', '6', '7', '8'] as const

export interface Arrow {
  from: string
  to: string
  kind: 'primary' | 'secondary'
  label?: string
}

interface BoardProps {
  fen: string
  legalMoves: Record<string, string[]>
  orientation: Colour
  lastMove: string | null
  checkers: string[]
  interactive: boolean
  arrows?: Arrow[]
  onMove: (from: string, to: string, promotion?: string) => void
}

/** Piece letters as they appear in a FEN, mapped to the cburnett file names. */
function pieceAsset(symbol: string): string {
  const colour = symbol === symbol.toUpperCase() ? 'w' : 'b'
  return `/pieces/cburnett/${colour}${symbol.toUpperCase()}.svg`
}

/** Expand the board part of a FEN into a `{ square: pieceLetter }` map. */
export function parseFen(fen: string): Record<string, string> {
  const placement = fen.split(' ')[0] ?? ''
  const pieces: Record<string, string> = {}
  let rank = 7
  let file = 0
  for (const character of placement) {
    if (character === '/') {
      rank -= 1
      file = 0
    } else if (character >= '1' && character <= '8') {
      file += Number(character)
    } else {
      if (rank >= 0 && file < 8) {
        pieces[`${FILES[file]}${RANKS[rank]}`] = character
      }
      file += 1
    }
  }
  return pieces
}

function squareOrder(orientation: Colour): string[] {
  const files = orientation === 'white' ? FILES : [...FILES].reverse()
  const ranks = orientation === 'white' ? [...RANKS].reverse() : RANKS
  const order: string[] = []
  for (const rank of ranks) {
    for (const file of files) {
      order.push(`${file}${rank}`)
    }
  }
  return order
}

/** Centre of a square in 0-8 board units, honouring the current orientation. */
function squareCentre(square: string, orientation: Colour): { x: number; y: number } {
  const fileIndex = FILES.indexOf(square[0] as (typeof FILES)[number])
  const rankIndex = RANKS.indexOf(square[1] as (typeof RANKS)[number])
  const column = orientation === 'white' ? fileIndex : 7 - fileIndex
  const row = orientation === 'white' ? 7 - rankIndex : rankIndex
  return { x: column + 0.5, y: row + 0.5 }
}

function isLightSquare(square: string): boolean {
  const fileIndex = FILES.indexOf(square[0] as (typeof FILES)[number])
  const rankIndex = RANKS.indexOf(square[1] as (typeof RANKS)[number])
  return (fileIndex + rankIndex) % 2 === 1
}

export default function Board({
  fen,
  legalMoves,
  orientation,
  lastMove,
  checkers,
  interactive,
  arrows = [],
  onMove,
}: BoardProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const boardRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState(480)
  const [selected, setSelected] = useState<string | null>(null)
  const [drag, setDrag] = useState<{ square: string; x: number; y: number } | null>(null)
  const [pending, setPending] = useState<{ from: string; to: string } | null>(null)
  const pointerOrigin = useRef<{ x: number; y: number } | null>(null)

  const pieces = useMemo(() => parseFen(fen), [fen])
  const order = useMemo(() => squareOrder(orientation), [orientation])
  const turn = (fen.split(' ')[1] ?? 'w') === 'w' ? 'white' : 'black'

  // The board is always a square that fits the space it is given.
  useLayoutEffect(() => {
    const element = wrapRef.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const box = entries[0].contentRect
      setSize(Math.max(200, Math.floor(Math.min(box.width, box.height))))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const destinations = selected ? (legalMoves[selected] ?? []) : []
  const dragDestinations = drag ? (legalMoves[drag.square] ?? []) : []
  const activeDestinations = drag ? dragDestinations : destinations

  const lastFrom = lastMove ? lastMove.slice(0, 2) : null
  const lastTo = lastMove ? lastMove.slice(2, 4) : null

  const needsPromotion = useCallback(
    (from: string, to: string) => {
      const piece = pieces[from]
      if (!piece || piece.toUpperCase() !== 'P') return false
      const targetRank = to[1]
      return (piece === 'P' && targetRank === '8') || (piece === 'p' && targetRank === '1')
    },
    [pieces],
  )

  const commit = useCallback(
    (from: string, to: string) => {
      if (!legalMoves[from]?.includes(to)) return
      if (needsPromotion(from, to)) {
        setPending({ from, to })
        setSelected(null)
        return
      }
      setSelected(null)
      onMove(from, to)
    },
    [legalMoves, needsPromotion, onMove],
  )

  const squareFromPoint = useCallback(
    (clientX: number, clientY: number): string | null => {
      const board = boardRef.current
      if (!board) return null
      const box = board.getBoundingClientRect()
      const column = Math.floor(((clientX - box.left) / box.width) * 8)
      const row = Math.floor(((clientY - box.top) / box.height) * 8)
      if (column < 0 || column > 7 || row < 0 || row > 7) return null
      return order[row * 8 + column]
    },
    [order],
  )

  const handlePointerDown = (event: React.PointerEvent, square: string) => {
    if (!interactive || pending) return
    const piece = pieces[square]
    const ownPiece =
      piece && (piece === piece.toUpperCase() ? 'white' : 'black') === turn

    if (selected && selected !== square && legalMoves[selected]?.includes(square)) {
      commit(selected, square)
      return
    }
    if (!ownPiece || !legalMoves[square]?.length) {
      setSelected(null)
      return
    }

    event.currentTarget.setPointerCapture?.(event.pointerId)
    pointerOrigin.current = { x: event.clientX, y: event.clientY }
    setSelected(square)
    setDrag({ square, x: event.clientX, y: event.clientY })
  }

  // Pointer move/up are bound to the window so a fast drag that leaves the
  // board still lands correctly.
  useEffect(() => {
    if (!drag) return
    const move = (event: PointerEvent) =>
      setDrag((current) => (current ? { ...current, x: event.clientX, y: event.clientY } : null))
    const up = (event: PointerEvent) => {
      const origin = pointerOrigin.current
      const travelled =
        origin
          ? Math.hypot(event.clientX - origin.x, event.clientY - origin.y)
          : Number.POSITIVE_INFINITY
      const target = squareFromPoint(event.clientX, event.clientY)
      const from = drag.square
      setDrag(null)
      pointerOrigin.current = null

      // A short press without travel is a click: keep the square selected so
      // the player can complete the move with a second click.
      if (travelled < 6) return
      if (target && target !== from) commit(from, target)
      else setSelected(null)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
    }
  }, [drag, commit, squareFromPoint])

  useEffect(() => {
    setSelected(null)
    setPending(null)
  }, [fen])

  const promotionColour = pending ? (pending.to[1] === '8' ? 'w' : 'b') : 'w'

  return (
    <div className="board-wrap" ref={wrapRef}>
      <div
        className="board"
        ref={boardRef}
        style={{ width: size, height: size, containerType: 'inline-size' }}
      >
        {order.map((square, index) => {
          const light = isLightSquare(square)
          const piece = pieces[square]
          const isDestination = activeDestinations.includes(square)
          const row = Math.floor(index / 8)
          const column = index % 8
          const showFile = row === 7
          const showRank = column === 0

          const classes = ['board__square', light ? 'board__square--light' : 'board__square--dark']
          if (square === lastFrom || square === lastTo) classes.push('board__square--last')
          if (square === selected) classes.push('board__square--selected')
          if (checkers.length && piece && piece.toUpperCase() === 'K') {
            const kingColour = piece === 'K' ? 'white' : 'black'
            if (kingColour === turn) classes.push('board__square--check')
          }
          if (interactive && legalMoves[square]?.length) classes.push('board__square--origin')

          return (
            <div
              key={square}
              className={classes.join(' ')}
              data-square={square}
              onPointerDown={(event) => handlePointerDown(event, square)}
            >
              {showRank && (
                <span
                  className={`board__coord board__coord--rank board__coord--${light ? 'on-light' : 'on-dark'}`}
                >
                  {square[1]}
                </span>
              )}
              {showFile && (
                <span
                  className={`board__coord board__coord--file board__coord--${light ? 'on-light' : 'on-dark'}`}
                >
                  {square[0]}
                </span>
              )}
              {isDestination && (
                <span className={`board__hint${piece ? ' board__hint--capture' : ''}`} />
              )}
              {piece && (
                <img
                  className={`board__piece${drag?.square === square ? ' board__piece--dragging' : ''}`}
                  src={pieceAsset(piece)}
                  alt={piece}
                  draggable={false}
                />
              )}
            </div>
          )
        })}

        <ArrowLayer arrows={arrows} orientation={orientation} />

        {pending && (
          <div className="promotion" onClick={() => setPending(null)}>
            <div className="promotion__card" onClick={(event) => event.stopPropagation()}>
              {['q', 'r', 'b', 'n'].map((choice) => (
                <button
                  key={choice}
                  type="button"
                  className="promotion__option"
                  onClick={() => {
                    onMove(pending.from, pending.to, choice)
                    setPending(null)
                  }}
                >
                  <img
                    src={`/pieces/cburnett/${promotionColour}${choice.toUpperCase()}.svg`}
                    alt={choice}
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {drag && pieces[drag.square] && (
        <img
          className="board__floating"
          src={pieceAsset(pieces[drag.square])}
          alt=""
          style={{ left: drag.x, top: drag.y, width: size / 8, height: size / 8 }}
        />
      )}
    </div>
  )
}

function ArrowLayer({ arrows, orientation }: { arrows: Arrow[]; orientation: Colour }) {
  if (!arrows.length) return null
  return (
    <svg className="board__overlay" viewBox="0 0 8 8" preserveAspectRatio="none">
      <defs>
        <marker
          id="arrow-primary"
          viewBox="0 0 10 10"
          refX="7"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 9 5 L 0 9 z" fill="var(--arrow-primary)" />
        </marker>
        <marker
          id="arrow-secondary"
          viewBox="0 0 10 10"
          refX="7"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 9 5 L 0 9 z" fill="var(--arrow-secondary)" />
        </marker>
      </defs>
      {arrows.map((arrow, index) => {
        const start = squareCentre(arrow.from, orientation)
        const end = squareCentre(arrow.to, orientation)
        const dx = end.x - start.x
        const dy = end.y - start.y
        const length = Math.hypot(dx, dy) || 1
        // Stop short of the target square so the head sits inside it.
        const trim = 0.32
        const tipX = end.x - (dx / length) * trim
        const tipY = end.y - (dy / length) * trim
        const colour = arrow.kind === 'primary' ? 'var(--arrow-primary)' : 'var(--arrow-secondary)'
        return (
          <line
            key={`${arrow.from}${arrow.to}${index}`}
            x1={start.x}
            y1={start.y}
            x2={tipX}
            y2={tipY}
            stroke={colour}
            strokeWidth={arrow.kind === 'primary' ? 0.14 : 0.1}
            strokeLinecap="round"
            markerEnd={`url(#arrow-${arrow.kind})`}
          />
        )
      })}
    </svg>
  )
}
