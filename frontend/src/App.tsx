import { useState } from 'react'
import type { FormEvent } from 'react'

import AnswerText from './AnswerText'
import EvidencePanel from './EvidencePanel'

import { askQuestion } from './api'
import type { AskResponse } from './api'

import './App.css'

// Keep this list aligned with the filings actually indexed.
const companies = [
  {
    ticker: 'AAPL',
    name: 'Apple Inc.',
    years: [2024],
  },
]

const sections = [
  'Business',
  'Risk Factors',
  'MD&A',
  'Financial Statements',
]

function App() {
  const [question, setQuestion] = useState('')
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [status, setStatus] = useState<
    'idle' | 'loading' | 'success' | 'error'
  >('idle')
  const [result, setResult] = useState<AskResponse | null>(null)
  const [error, setError] = useState('')
  const [company, setCompany] = useState(companies[0])
  const [year, setYear] = useState(companies[0].years[0])
  const [section, setSection] = useState('')

  // Preserve the filters used for the displayed answer.
  const [submittedFilters, setSubmittedFilters] = useState({
    company: companies[0].name,
    year: companies[0].years[0],
    section: '',
  })

  const isLoading = status === 'loading'

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || isLoading) {
      return
    }

    // Keep the displayed result attached to the submitted question.
    setSubmittedQuestion(trimmedQuestion)
    setSubmittedFilters({
      company: company.name,
      year,
      section,
    })
    setResult(null)
    setError('')
    setStatus('loading')

    try {
      const response = await askQuestion({
        question: trimmedQuestion,
        ticker: company.ticker,
        year,
        ...(section ? { section } : {}),
      })

      setResult(response)
      setStatus('success')
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : 'Something went wrong. Please try again.',
      )
      setStatus('error')
    }
  }

  return (
    <div className="workspace">
      <header className="site-header">
        <a className="brand" href="./" aria-label="FilingLens home">
          <span className="brand-mark" aria-hidden="true">
            FL
          </span>
          FilingLens
        </a>

        <span className="header-description">
          Research backed by filings
        </span>
      </header>

      <main>
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">COMPANY RESEARCH</p>
          <h1 id="page-title">Ask the filing. Check the evidence.</h1>
          <p className="muted">
            Explore company disclosures with answers linked to their sources.
          </p>
        </section>

        <form
          className="question-panel"
          aria-labelledby="question-title"
          onSubmit={handleSubmit}
        >
          <div className="panel-heading">
            <h2 id="question-title">Your question</h2>
            <span className="badge">
              {company.ticker} {year} Form 10-K
            </span>
          </div>

          {/* Filters remain fixed until section 4.5. */}
          <div className="filters">
            <label>
              Company
              <select
                value={company.ticker}
                disabled={isLoading}
                onChange={(event) => {
                  const selected = companies.find(
                    (item) => item.ticker === event.target.value,
                  )

                  if (selected) {
                    setCompany(selected)
                    setYear(selected.years[0])
                    setSection('')
                  }
                }}
              >
                {companies.map((item) => (
                  <option key={item.ticker} value={item.ticker}>
                    {item.name} · {item.ticker}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Filing year
              <select
                value={year}
                disabled={isLoading}
                onChange={(event) => {
                  setYear(Number(event.target.value))
                  setSection('')
                }}
              >
                {company.years.map((availableYear) => (
                  <option key={availableYear} value={availableYear}>
                    {availableYear}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Section
              <select
                value={section}
                disabled={isLoading}
                onChange={(event) => setSection(event.target.value)}
              >
                <option value="">All sections</option>

                {sections.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="question-label" htmlFor="question">
            Question about the selected filing
          </label>

          <textarea
            id="question"
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What was Apple's net income in fiscal 2024?"
            maxLength={1000}
            required
            disabled={isLoading}
          />

          <div className="question-footer">
            <p className="muted">
              {company.ticker} {year} Form 10-K
              {' · '}Section: {section || 'All sections'}
            </p>

            <button
              className="primary-button"
              type="submit"
              disabled={isLoading || !question.trim()}
            >
              {isLoading ? 'Preparing answer…' : 'Ask filing'}
            </button>
          </div>
        </form>

        <div className="results-layout">
          <section
            className="answer-panel"
            aria-labelledby="answer-title"
            aria-busy={isLoading}
          >
            <div className="panel-heading">
              <h2 id="answer-title">Answer</h2>

              {result && (
                <span className="badge">
                  {result.status === 'answered'
                    ? 'Answer ready'
                    : 'Insufficient evidence'}
                </span>
              )}
            </div>

            {status === 'idle' && (
              <p className="muted">
                Ask a question to explore the selected filing.
              </p>
            )}

            {submittedQuestion && (
              <div className="submitted-question">
                <p>{submittedQuestion}</p>
                <p>
                  {submittedFilters.company} · Filing year {submittedFilters.year}
                  {' · '}{submittedFilters.section || 'All sections'}
                </p>
              </div>
            )}

            {isLoading && (
              <p className="request-message" role="status">
                Searching the filing and preparing an answer…
              </p>
            )}

            {status === 'error' && (
              <p className="error-message" role="alert">
                {error}
              </p>
            )}

            {result && (
              <>
                <AnswerText
                  text={result.answer}
                  sources={result.sources}
              />

                {result.status === 'answered' ? (
                  <>
                    <div className="answer-footer">
                      <span>{submittedFilters.company}</span>
                      <span>{submittedFilters.year} Form 10-K</span>
                    </div>

                    <p className="muted verification-note">
                      Review the supporting evidence before relying on an answer.
                    </p>
                  </>
                ) : (
                  <p className="muted verification-note">
                    Try a different question about the selected filing.
                  </p>
                )}
              </>
            )}
          </section>

          {result?.status === 'answered' &&
            result.sources.length > 0 && (
              <EvidencePanel sources={result.sources} />
            )}
        </div>

        
      </main>

      <footer className="site-footer">
        FilingLens · Answers grounded in company disclosures
      </footer>
    </div>
  )
}

export default App