import { useEffect, useState } from 'react'
import type { ReactElement } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import * as api from '../api/client'
import type { DocSummary } from '../api/types'
import MermaidDiagram from '../components/MermaidDiagram'
import './docs.css'

export default function DocsPage() {
  const [documents, setDocuments] = useState<DocSummary[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [markdown, setMarkdown] = useState<string>('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api
      .getDocs()
      .then((payload) => {
        setDocuments(payload.documents)
        setActive((current) => current ?? payload.documents[0]?.slug ?? null)
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!active) return
    setLoading(true)
    api
      .getDoc(active)
      .then((payload) => setMarkdown(payload.markdown))
      .catch(() => setMarkdown('# Documento non disponibile'))
      .finally(() => setLoading(false))
  }, [active])

  return (
    <div className="docs">
      <nav className="docs__nav scroll">
        <p className="docs__navtitle">Documentazione</p>
        {documents.map((document) => (
          <button
            key={document.slug}
            type="button"
            className="docs__navitem"
            aria-current={document.slug === active ? 'page' : undefined}
            onClick={() => setActive(document.slug)}
          >
            <span className="docs__navlabel">{document.title}</span>
            <span className="docs__navsummary">{document.summary}</span>
          </button>
        ))}
      </nav>

      <article className="docs__body scroll">
        {loading ? (
          <p className="muted">Caricamento…</p>
        ) : (
          <div className="prose">
            <Markdown
              remarkPlugins={[remarkGfm]}
              components={{
                pre: ({ children }) => {
                  // Fenced ```mermaid blocks become live diagrams; everything
                  // else keeps the normal <pre><code> rendering.
                  const child = children as ReactElement<{
                    className?: string
                    children?: unknown
                  }>
                  const className = child?.props?.className ?? ''
                  const language = /language-(\w+)/.exec(className)?.[1]
                  if (language === 'mermaid') {
                    const code = String(child.props.children ?? '').replace(/\n$/, '')
                    return <MermaidDiagram chart={code} />
                  }
                  return <pre>{children}</pre>
                },
                a: ({ href, children }) => {
                  // Cross-links between documents stay inside the app.
                  const internal = href?.endsWith('.md')
                  if (internal) {
                    const slug = href!.replace(/\.md$/, '').replace(/^\.\//, '')
                    return (
                      <button
                        type="button"
                        className="prose__link"
                        onClick={() => setActive(slug)}
                      >
                        {children}
                      </button>
                    )
                  }
                  return (
                    <a href={href} target="_blank" rel="noreferrer">
                      {children}
                    </a>
                  )
                },
              }}
            >
              {markdown}
            </Markdown>
          </div>
        )}
      </article>
    </div>
  )
}
