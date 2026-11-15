export interface Source {
  citation_id: number
  chunk_id: string
  ticker: string
  company: string
  fiscal_year: number
  section: string
  content_type: 'prose' | 'table'
  text: string
  citation_url: string
}

export interface AskRequest {
  question: string
  ticker: string
  year: number
  section?: string
}

export interface AskResponse {
  question: string
  status: 'answered' | 'insufficient_evidence'
  answer: string
  sources: Source[]
}

export async function askQuestion(
  request: AskRequest,
): Promise<AskResponse> {
  let response: Response

  try {
    response = await fetch('http://127.0.0.1:8001/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })
  } catch {
    throw new Error(
      'Cannot reach the backend. Check that it is running and try again.',
    )
  }

  if (!response.ok) {
    const error = await response.json().catch(() => null)

    throw new Error(
      typeof error?.detail === 'string'
        ? error.detail
        : `The request failed (HTTP ${response.status}). Please try again.`,
    )
  }

  return response.json()
}