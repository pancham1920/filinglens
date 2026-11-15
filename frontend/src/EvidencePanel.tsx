import type { Source } from './api'

interface EvidencePanelProps {
  sources: Source[]
}

export default function EvidencePanel({
  sources,
}: EvidencePanelProps) {
  return (
    <aside
      className="evidence-panel"
      aria-labelledby="evidence-title"
    >
      <div className="panel-heading">
        <h2 id="evidence-title">Supporting evidence</h2>
        <span className="muted">
          {sources.length} {sources.length === 1 ? 'source' : 'sources'}
        </span>
      </div>

      <div className="evidence-list">
        {sources.map((source) => (
          <article
            key={source.chunk_id}
            id={`evidence-${source.citation_id}`}
            className="evidence-card"
            aria-labelledby={`source-title-${source.citation_id}`}
            tabIndex={-1}
          >
            <div className="evidence-heading">
              <span className="source-number">
                [{source.citation_id}]
              </span>

              <div>
                <h3 id={`source-title-${source.citation_id}`}>
                  {source.section}
                </h3>
                <p className="muted">
                  {source.ticker} · Filing year {source.fiscal_year}
                  {' · '}{source.content_type}
                </p>
              </div>
            </div>

            {/* Show the complete chunk, preserving table line breaks. */}
            <pre
              className="evidence-text"
              tabIndex={0}
              aria-label={`Evidence text for source ${source.citation_id}`}
            >
              {source.text}
            </pre>

            <a
              className="source-link"
              href={source.citation_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open original SEC filing ↗
            </a>
          </article>
        ))}
      </div>
    </aside>
  )
}