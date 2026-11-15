import type { Source } from './api'

interface AnswerTextProps {
  text: string
  sources: Source[]
}

export default function AnswerText({
  text,
  sources,
}: AnswerTextProps) {
  // Capturing parentheses keep citation markers in the resulting array.
  const parts = text.split(/(\[\d+\])/)

  return (
    <p className="answer-text">
      {parts.map((part, index) => {
        const match = part.match(/^\[(\d+)\]$/)

        if (!match) {
          return part
        }

        const citationId = Number(match[1])
        const source = sources.find(
          (source) => source.citation_id === citationId,
        )

        // Only create links for references included in the API response.
        if (!source) {
          return part
        }

        return (
          <a
            key={index}
            className="citation"
            href={`#evidence-${citationId}`}
            aria-label={
                `Read source ${citationId}: ${source.section}, ` +
                `${source.company}, ${source.fiscal_year}`
            }
            >
            {part}
          </a>
        )
      })}
    </p>
  )
}