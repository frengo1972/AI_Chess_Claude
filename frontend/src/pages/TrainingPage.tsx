import { useCallback, useEffect, useMemo, useState } from 'react'
import LineChart, { type Series } from '../components/charts'
import { Panel, Stat } from '../components/panels'
import * as api from '../api/client'
import type {
  IterationRow,
  PresetInfo,
  TrainingRun,
  TrainingStatus,
} from '../api/types'
import './training.css'

const PHASE_LABEL: Record<string, string> = {
  selfplay: 'self-play',
  train: 'addestramento',
  arena: 'arena',
  benchmark: 'benchmark',
  idle: 'in attesa',
}

const STATUS_LABEL: Record<string, string> = {
  running: 'in corso',
  finished: 'completato',
  stopped: 'interrotto',
  failed: 'fallito',
}

export default function TrainingPage() {
  const [presets, setPresets] = useState<Record<string, PresetInfo>>({})
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [history, setHistory] = useState<IterationRow[]>([])
  const [events, setEvents] = useState<Array<{ timestamp: number; level: string; message: string }>>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [form, setForm] = useState({
    preset: 'small',
    name: '',
    iterations: 40,
    games: 0,
    simulations: 0,
    workers: 0,
  })

  /* ----------------------------------------------------------- bootstrap */

  const refreshRuns = useCallback(async () => {
    const payload = await api.getTrainingRuns()
    setRuns(payload.runs)
    setSelectedRun((current) => current ?? payload.runs[0]?.id ?? null)
  }, [])

  useEffect(() => {
    api.getPresets().then((payload) => setPresets(payload.presets)).catch(() => undefined)
    refreshRuns().catch((cause) => setError(String(cause.message ?? cause)))
  }, [refreshRuns])

  /* ------------------------------------------------------- live updates */

  useEffect(() => {
    const socket = api.openTrainingSocket((payload) => {
      setStatus(payload)
      if (payload.run && !selectedRun) setSelectedRun(payload.run.id)
    }, selectedRun ?? undefined)
    return () => socket.close()
  }, [selectedRun])

  useEffect(() => {
    if (!selectedRun) return
    let cancelled = false
    const load = () => {
      api
        .getTrainingHistory(selectedRun)
        .then((payload) => !cancelled && setHistory(payload.iterations))
        .catch(() => undefined)
      api
        .getTrainingEvents(selectedRun, 40)
        .then((payload) => !cancelled && setEvents(payload.events))
        .catch(() => undefined)
    }
    load()
    const timer = window.setInterval(load, 5000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [selectedRun, status?.last_iteration?.iteration])

  /* ---------------------------------------------------------- controls */

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const response = await api.startTraining({
        preset: form.preset,
        name: form.name || form.preset,
        iterations: form.iterations || undefined,
        games_per_iteration: form.games || undefined,
        simulations: form.simulations || undefined,
        workers: form.workers || undefined,
      })
      setSelectedRun(response.run_id)
      await refreshRuns()
    } catch (cause) {
      setError(String((cause as Error).message))
    } finally {
      setBusy(false)
    }
  }

  const stop = async (force = false) => {
    setBusy(true)
    try {
      await api.stopTraining(status?.run?.id ?? selectedRun ?? undefined, force)
    } catch (cause) {
      setError(String((cause as Error).message))
    } finally {
      setBusy(false)
    }
  }

  /* ----------------------------------------------------------- derived */

  const labels = useMemo(() => history.map((row) => row.iteration), [history])
  const pick = useCallback(
    (key: keyof IterationRow): Array<number | null> =>
      history.map((row) => {
        const value = row[key]
        return typeof value === 'number' ? value : null
      }),
    [history],
  )

  const last = history[history.length - 1]
  const run = status?.run ?? runs.find((item) => item.id === selectedRun) ?? null
  const network = (run?.network ?? {}) as Record<string, number>
  const isActive = status?.active && status.run?.id === selectedRun

  const series = (name: string, key: keyof IterationRow, colour: string): Series => ({
    name,
    colour,
    values: pick(key),
  })

  return (
    <div className="training">
      {/* --------------------------------------------------------- header */}
      <header className="training__header">
        <div className="training__identity">
          <h1 className="training__title">Apprendimento</h1>
          <p className="training__lead">
            La rete gioca contro sé stessa, impara dalle proprie partite e viene promossa
            solo se batte la versione precedente.
          </p>
        </div>
        <div className="training__runpicker">
          <label className="field">
            <span className="field__label">Run</span>
            <select
              value={selectedRun ?? ''}
              onChange={(event) => setSelectedRun(event.target.value || null)}
            >
              {runs.length === 0 && <option value="">nessun run</option>}
              {runs.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {item.id.slice(-8)} · {STATUS_LABEL[item.status] ?? item.status}
                </option>
              ))}
            </select>
          </label>
          <span className={`chip ${isActive ? 'chip--live' : 'chip--off'}`}>
            <span className="dot" />
            {isActive ? PHASE_LABEL[status?.phase ?? 'idle'] ?? status?.phase : 'fermo'}
          </span>
        </div>
      </header>

      {error && <div className="banner banner--error">{error}</div>}

      {/* ------------------------------------------------------------ KPI */}
      <section className="training__kpis">
        <Stat label="Iterazione" value={last?.iteration ?? 0} hint={`di ${run?.config?.train?.iterations ?? '—'}`} />
        <Stat label="Partite giocate" value={formatCount(last?.games_total ?? 0)} hint="cumulative" />
        <Stat label="Posizioni" value={formatCount(last?.positions_total ?? 0)} hint="esempi di training" />
        <Stat
          label="Elo relativo"
          value={last?.elo !== null && last?.elo !== undefined ? Math.round(last.elo) : 0}
          hint="rispetto alla rete iniziale"
        />
        <Stat
          label="Elo vs Stockfish"
          value={
            last?.benchmark_elo !== null && last?.benchmark_elo !== undefined
              ? Math.round(last.benchmark_elo)
              : '—'
          }
          hint="stima assoluta"
        />
        <Stat
          label="Parametri rete"
          value={formatCount(network.parameters ?? 0)}
          hint={`${network.residual_blocks ?? '—'}×${network.filters ?? '—'} · ${(network.size_mb ?? 0).toFixed(1)} MB`}
        />
        <Stat label="Replay buffer" value={formatCount(last?.buffer_size ?? 0)} hint="posizioni in memoria" />
        <Stat
          label="Throughput"
          value={last?.games_per_minute ? `${last.games_per_minute.toFixed(1)}/min` : '—'}
          hint="partite di self-play"
        />
      </section>

      {/* --------------------------------------------------------- charts */}
      <section className="training__charts">
        <LineChart
          title="Forza nel tempo"
          subtitle="Elo"
          labels={labels}
          series={[
            series('Elo relativo', 'elo', 'var(--series-1)'),
            series('Elo vs Stockfish', 'benchmark_elo', 'var(--series-2)'),
          ]}
          format={(value) => value.toFixed(0)}
        />
        <LineChart
          title="Score in arena"
          subtitle="candidato contro campione"
          labels={labels}
          series={[series('Score', 'arena_score', 'var(--series-1)')]}
          domain={[0, 1]}
          reference={{ value: 0.5, label: 'parità' }}
          format={(value) => value.toFixed(2)}
        />
        <LineChart
          title="Policy loss"
          subtitle="cross-entropy vs visite MCTS"
          labels={labels}
          series={[series('policy', 'policy_loss', 'var(--series-1)')]}
          format={(value) => value.toFixed(2)}
        />
        <LineChart
          title="Value loss"
          subtitle="MSE vs risultato finale"
          labels={labels}
          series={[series('value', 'value_loss', 'var(--series-2)')]}
          format={(value) => value.toFixed(3)}
        />
        <LineChart
          title="Esiti delle partite"
          subtitle="quota sul totale"
          labels={labels}
          series={[
            series('Patte', 'draw_rate', 'var(--series-1)'),
            series('Vince Bianco', 'white_win_rate', 'var(--series-2)'),
            series('Vince Nero', 'black_win_rate', 'var(--series-3)'),
          ]}
          domain={[0, 1]}
          format={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <LineChart
          title="Lunghezza delle partite"
          subtitle="semimosse medie"
          labels={labels}
          series={[series('semimosse', 'avg_plies', 'var(--series-1)')]}
          format={(value) => value.toFixed(0)}
        />
        <LineChart
          title="Entropia della policy"
          subtitle="quanto esplora la ricerca"
          labels={labels}
          series={[series('entropia', 'policy_entropy', 'var(--series-3)')]}
          format={(value) => value.toFixed(2)}
        />
        <LineChart
          title="Velocità"
          subtitle="partite di self-play al minuto"
          labels={labels}
          series={[series('partite/min', 'games_per_minute', 'var(--series-2)')]}
          format={(value) => value.toFixed(1)}
        />
      </section>

      {/* -------------------------------------------------------- bottom */}
      <section className="training__bottom">
        <Panel title="Avvia un addestramento">
          <div className="setup">
            <label className="field">
              <span className="field__label">Preset</span>
              <select
                value={form.preset}
                onChange={(event) => setForm({ ...form, preset: event.target.value })}
              >
                {Object.keys(presets).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            {presets[form.preset] && (
              <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                {presets[form.preset].summary.blocks}×{presets[form.preset].summary.filters},
                T={presets[form.preset].summary.history},{' '}
                {formatCount(presets[form.preset].summary.parameters)} parametri ·{' '}
                {presets[form.preset].summary.simulations} simulazioni ·{' '}
                {presets[form.preset].summary.games_per_iteration} partite/iterazione
              </p>
            )}

            <label className="field">
              <span className="field__label">Nome</span>
              <input
                type="text"
                value={form.name}
                placeholder={form.preset}
                onChange={(event) => setForm({ ...form, name: event.target.value })}
              />
            </label>

            <div className="training__formgrid">
              <label className="field">
                <span className="field__label">Iterazioni</span>
                <input
                  type="number"
                  min={1}
                  value={form.iterations}
                  onChange={(event) => setForm({ ...form, iterations: Number(event.target.value) })}
                />
              </label>
              <label className="field">
                <span className="field__label">Partite/iter</span>
                <input
                  type="number"
                  min={0}
                  value={form.games}
                  onChange={(event) => setForm({ ...form, games: Number(event.target.value) })}
                />
              </label>
              <label className="field">
                <span className="field__label">Simulazioni</span>
                <input
                  type="number"
                  min={0}
                  value={form.simulations}
                  onChange={(event) => setForm({ ...form, simulations: Number(event.target.value) })}
                />
              </label>
              <label className="field">
                <span className="field__label">Worker</span>
                <input
                  type="number"
                  min={0}
                  value={form.workers}
                  onChange={(event) => setForm({ ...form, workers: Number(event.target.value) })}
                />
              </label>
            </div>
            <p className="muted" style={{ margin: 0, fontSize: 11 }}>
              0 = usa il valore del preset.
            </p>

            <div className="row">
              <button
                type="button"
                className="btn btn--primary"
                disabled={busy || Boolean(status?.active)}
                onClick={() => void start()}
              >
                Avvia
              </button>
              <button
                type="button"
                className="btn"
                disabled={busy || !status?.active}
                onClick={() => void stop(false)}
              >
                Ferma
              </button>
              <button
                type="button"
                className="btn btn--danger btn--small"
                disabled={busy || !status?.active}
                onClick={() => void stop(true)}
              >
                Forza stop
              </button>
            </div>
            {status?.stop_requested && (
              <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                Stop richiesto: il trainer esce alla fine dell’iterazione corrente.
              </p>
            )}
          </div>
        </Panel>

        <Panel title="Registro" flush>
          <div className="training__events scroll">
            {events.length === 0 && (
              <p className="muted" style={{ padding: 12, margin: 0, fontSize: 12 }}>
                Nessun evento.
              </p>
            )}
            {events.map((event, index) => (
              <div key={index} className={`training__event training__event--${event.level}`}>
                <span className="numeric muted">
                  {new Date(event.timestamp * 1000).toLocaleTimeString('it-IT')}
                </span>
                <span>{event.message}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Configurazione del run" flush>
          <div className="training__events scroll">
            {run ? (
              <table className="training__config">
                <tbody>
                  {flattenConfig(run.config).map(([key, value]) => (
                    <tr key={key}>
                      <th>{key}</th>
                      <td className="numeric">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted" style={{ padding: 12, margin: 0, fontSize: 12 }}>
                Seleziona un run.
              </p>
            )}
          </div>
        </Panel>
      </section>
    </div>
  )
}

function formatCount(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)} k`
  return String(Math.round(value))
}

const CONFIG_KEYS: Array<[string, string]> = [
  ['network.history_length', 'storia T'],
  ['network.residual_blocks', 'blocchi'],
  ['network.filters', 'filtri'],
  ['search.simulations', 'simulazioni'],
  ['search.c_puct', 'c_puct'],
  ['search.dirichlet_epsilon', 'rumore radice'],
  ['selfplay.games_per_iteration', 'partite/iterazione'],
  ['selfplay.workers', 'worker'],
  ['selfplay.max_game_plies', 'semimosse max'],
  ['train.batch_size', 'batch'],
  ['train.steps_per_iteration', 'step/iterazione'],
  ['train.learning_rate', 'learning rate'],
  ['train.optimizer', 'ottimizzatore'],
  ['train.replay_buffer_size', 'replay buffer'],
  ['arena.games', 'partite arena'],
  ['arena.win_threshold', 'soglia promozione'],
]

function flattenConfig(config: Record<string, any>): Array<[string, string]> {
  return CONFIG_KEYS.map(([path, label]) => {
    const value = path.split('.').reduce<any>((node, key) => node?.[key], config)
    return [label, value === undefined || value === null ? '—' : String(value)] as [string, string]
  })
}
