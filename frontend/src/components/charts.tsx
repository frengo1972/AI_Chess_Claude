import { useMemo, useRef, useState } from 'react'
import './charts.css'

export interface Series {
  name: string
  colour: string
  /** `null` marks a gap (an iteration that skipped training, for example). */
  values: Array<number | null>
}

interface LineChartProps {
  title: string
  subtitle?: string
  /** Shared x values — every series must have one entry per label. */
  labels: number[]
  series: Series[]
  format?: (value: number) => string
  /** Force the y domain, e.g. `[0, 1]` for rates. */
  domain?: [number, number]
  height?: number
  /** Reference line, e.g. 0.5 for a match score. */
  reference?: { value: number; label: string }
}

const PADDING = { top: 10, right: 46, bottom: 20, left: 40 }

/**
 * A small multiple line chart drawn as inline SVG.
 *
 * One y-axis by construction: series that do not share a unit belong in
 * separate charts. Every chart carries a crosshair + tooltip, a legend when
 * there is more than one series, and a direct label on the last point.
 */
export default function LineChart({
  title,
  subtitle,
  labels,
  series,
  format = (value) => value.toFixed(2),
  domain,
  height = 168,
  reference,
}: LineChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<number | null>(null)
  const width = 420

  const { min, max } = useMemo(() => {
    if (domain) return { min: domain[0], max: domain[1] }
    const numbers = series.flatMap((item) =>
      item.values.filter((value): value is number => value !== null && Number.isFinite(value)),
    )
    if (reference) numbers.push(reference.value)
    if (!numbers.length) return { min: 0, max: 1 }
    let low = Math.min(...numbers)
    let high = Math.max(...numbers)
    if (low === high) {
      low -= 0.5
      high += 0.5
    }
    const margin = (high - low) * 0.12
    return { min: low - margin, max: high + margin }
  }, [series, domain, reference])

  const plotWidth = width - PADDING.left - PADDING.right
  const plotHeight = height - PADDING.top - PADDING.bottom

  const xAt = (index: number) =>
    PADDING.left + (labels.length <= 1 ? plotWidth / 2 : (index / (labels.length - 1)) * plotWidth)
  const yAt = (value: number) =>
    PADDING.top + plotHeight - ((value - min) / (max - min)) * plotHeight

  const ticks = useMemo(() => {
    const count = 4
    return Array.from({ length: count + 1 }, (_, index) => min + ((max - min) * index) / count)
  }, [min, max])

  const hasData = labels.length > 0 && series.some((item) => item.values.some((v) => v !== null))

  const handleMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (!svg || labels.length === 0) return
    const box = svg.getBoundingClientRect()
    const relative = ((event.clientX - box.left) / box.width) * width
    const ratio = (relative - PADDING.left) / plotWidth
    const index = Math.round(ratio * (labels.length - 1))
    setHover(Math.max(0, Math.min(labels.length - 1, index)))
  }

  return (
    <figure className="chart" style={{ margin: 0 }}>
      <div className="chart__head">
        <figcaption className="chart__title">{title}</figcaption>
        {subtitle && <span className="chart__subtitle">{subtitle}</span>}
      </div>

      {series.length > 1 && (
        <div className="chart__legend">
          {series.map((item) => (
            <span key={item.name} className="chart__legend-item">
              <span className="chart__swatch" style={{ background: item.colour }} />
              {item.name}
            </span>
          ))}
        </div>
      )}

      {!hasData ? (
        <div className="chart__empty">nessun dato ancora</div>
      ) : (
        <>
          <svg
            ref={svgRef}
            className="chart__plot"
            viewBox={`0 0 ${width} ${height}`}
            height={height}
            preserveAspectRatio="none"
            onPointerMove={handleMove}
            onPointerLeave={() => setHover(null)}
          >
            {ticks.map((tick) => (
              <g key={tick}>
                <line
                  className="chart__grid"
                  x1={PADDING.left}
                  x2={width - PADDING.right}
                  y1={yAt(tick)}
                  y2={yAt(tick)}
                />
                <text className="chart__tick" x={PADDING.left - 6} y={yAt(tick) + 3} textAnchor="end">
                  {format(tick)}
                </text>
              </g>
            ))}

            {reference && (
              <line
                className="chart__axis"
                strokeDasharray="4 4"
                x1={PADDING.left}
                x2={width - PADDING.right}
                y1={yAt(reference.value)}
                y2={yAt(reference.value)}
              />
            )}

            <line
              className="chart__axis"
              x1={PADDING.left}
              x2={width - PADDING.right}
              y1={PADDING.top + plotHeight}
              y2={PADDING.top + plotHeight}
            />

            {labels.length > 1 &&
              [0, labels.length - 1].map((index) => (
                <text
                  key={index}
                  className="chart__tick"
                  x={xAt(index)}
                  y={height - 6}
                  textAnchor={index === 0 ? 'start' : 'end'}
                >
                  {labels[index]}
                </text>
              ))}

            {series.map((item) => {
              const segments: string[] = []
              let open = false
              item.values.forEach((value, index) => {
                if (value === null || !Number.isFinite(value)) {
                  open = false
                  return
                }
                segments.push(`${open ? 'L' : 'M'}${xAt(index)},${yAt(value)}`)
                open = true
              })
              const lastIndex = lastDefined(item.values)
              return (
                <g key={item.name}>
                  <path className="chart__line" d={segments.join(' ')} stroke={item.colour} />
                  {lastIndex !== -1 && (
                    <>
                      <circle
                        cx={xAt(lastIndex)}
                        cy={yAt(item.values[lastIndex] as number)}
                        r={3.5}
                        fill={item.colour}
                        className="chart__marker"
                      />
                      <text
                        className="chart__endlabel"
                        x={xAt(lastIndex) + 8}
                        y={yAt(item.values[lastIndex] as number) + 3.5}
                      >
                        {format(item.values[lastIndex] as number)}
                      </text>
                    </>
                  )}
                </g>
              )
            })}

            {hover !== null && (
              <>
                <line
                  className="chart__crosshair"
                  x1={xAt(hover)}
                  x2={xAt(hover)}
                  y1={PADDING.top}
                  y2={PADDING.top + plotHeight}
                />
                {series.map((item) => {
                  const value = item.values[hover]
                  if (value === null || value === undefined || !Number.isFinite(value)) return null
                  return (
                    <circle
                      key={item.name}
                      cx={xAt(hover)}
                      cy={yAt(value)}
                      r={4}
                      fill={item.colour}
                      className="chart__marker"
                    />
                  )
                })}
              </>
            )}
          </svg>

          {hover !== null && (
            <div
              className="chart__tooltip"
              style={{
                left: `${(xAt(hover) / width) * 100}%`,
                top: `${PADDING.top + 14}px`,
              }}
            >
              <div className="chart__tooltip-title">Iterazione {labels[hover]}</div>
              {series.map((item) => {
                const value = item.values[hover]
                return (
                  <div key={item.name} className="chart__tooltip-row">
                    <span className="chart__swatch" style={{ background: item.colour }} />
                    <span>{item.name}</span>
                    <span>
                      {value === null || value === undefined || !Number.isFinite(value)
                        ? '—'
                        : format(value)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}
    </figure>
  )
}

function lastDefined(values: Array<number | null>): number {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index]
    if (value !== null && Number.isFinite(value)) return index
  }
  return -1
}
