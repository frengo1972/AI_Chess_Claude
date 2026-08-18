import { useEffect, useId, useRef, useState } from 'react'

/**
 * Renders a fenced ```mermaid code block as an inline SVG diagram.
 *
 * `mermaid` is imported dynamically, not at module scope: it pulls in a
 * renderer per diagram type (flowchart, sequence, gantt, ...) and is only
 * ever needed on the Docs page. A static import would ship that weight in
 * every route's bundle, since this app has no route-level code-splitting.
 *
 * Mermaid is (re-)initialised on every render call rather than once
 * globally: re-running `mermaid.initialize` is how its own docs recommend
 * switching themes at runtime, and this app's theme (light/dark/system) can
 * change without a page reload.
 */

function currentIsDark(): boolean {
  const explicit = document.documentElement.getAttribute('data-theme')
  if (explicit === 'dark') return true
  if (explicit === 'light') return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export default function MermaidDiagram({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '-')
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDark, setIsDark] = useState(currentIsDark)

  useEffect(() => {
    const root = document.documentElement
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const recompute = () => setIsDark(currentIsDark())
    const observer = new MutationObserver(recompute)
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] })
    media.addEventListener('change', recompute)
    return () => {
      observer.disconnect()
      media.removeEventListener('change', recompute)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    import('mermaid').then(({ default: mermaid }) => {
      if (cancelled) return
      mermaid.initialize({
        startOnLoad: false,
        theme: isDark ? 'dark' : 'default',
        securityLevel: 'strict',
        fontFamily: 'inherit',
      })
      return mermaid.render(`mermaid-${id}`, chart)
    })
      .then((result) => {
        if (!cancelled && result && containerRef.current) {
          containerRef.current.innerHTML = result.svg
          setError(null)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [chart, id, isDark])

  if (error) {
    return (
      <div className="docs__mermaid docs__mermaid--error">
        <p>Diagramma non renderizzabile: {error}</p>
        <pre>{chart}</pre>
      </div>
    )
  }

  return <div className="docs__mermaid" ref={containerRef} role="img" />
}
