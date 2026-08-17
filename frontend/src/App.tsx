import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import PlayPage from './pages/PlayPage'
import TrainingPage from './pages/TrainingPage'
import DocsPage from './pages/DocsPage'
import * as api from './api/client'
import type { SystemInfo } from './api/types'

type Theme = 'light' | 'dark' | 'system'

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('aichess-theme') as Theme) ?? 'system',
  )

  useEffect(() => {
    api.getSystem().then(setSystem).catch(() => undefined)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    localStorage.setItem('aichess-theme', theme)
  }, [theme])

  return (
    <div className="shell">
      <header className="topbar">
        <span className="topbar__brand">
          <img src="/pieces/cburnett/wN.svg" alt="" />
          AI Chess
          <span className="topbar__mark">rete neurale · self-play</span>
        </span>

        <nav className="topbar__nav">
          <NavLink to="/play" className="topbar__link">
            Gioca
          </NavLink>
          <NavLink to="/training" className="topbar__link">
            Training
          </NavLink>
          <NavLink to="/docs" className="topbar__link">
            Documentazione
          </NavLink>
        </nav>

        <span className="topbar__spacer" />

        <div className="topbar__status">
          {system && (
            <>
              <span className={`chip ${system.cuda_available ? 'chip--live' : 'chip--off'}`}>
                <span className="dot" />
                {system.cuda_available ? system.cuda_device ?? 'CUDA' : 'CPU'}
              </span>
              <span
                className={`chip ${system.stockfish.available ? '' : 'chip--off'}`}
                title={system.stockfish.path ?? 'non trovato'}
              >
                {system.stockfish.available ? 'Stockfish pronto' : 'Stockfish assente'}
              </span>
            </>
          )}
          <button
            type="button"
            className="btn btn--small"
            onClick={() =>
              setTheme((current) =>
                current === 'dark' ? 'light' : current === 'light' ? 'system' : 'dark',
              )
            }
            title="Tema chiaro / scuro / di sistema"
          >
            {theme === 'dark' ? 'Scuro' : theme === 'light' ? 'Chiaro' : 'Auto'}
          </button>
        </div>
      </header>

      <main className="page">
        <Routes>
          <Route path="/" element={<Navigate to="/play" replace />} />
          <Route path="/play" element={<PlayPage />} />
          <Route
            path="/training"
            element={
              <div className="page page--scroll" style={{ height: '100%' }}>
                <TrainingPage />
              </div>
            }
          />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="*" element={<Navigate to="/play" replace />} />
        </Routes>
      </main>
    </div>
  )
}
