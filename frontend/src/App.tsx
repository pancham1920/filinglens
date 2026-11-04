import { useEffect, useState } from 'react'

function App() {
  const [status, setStatus] = useState('Checking backend...')

  useEffect(() => {
    const controller = new AbortController()

    async function checkBackend() {
      try {
        const response = await fetch('http://127.0.0.1:8001/health', {
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = await response.json()

        setStatus(
          data.status === 'ok'
            ? 'Backend connected'
            : 'Unexpected backend response',
        )
      } catch {
        if (!controller.signal.aborted) {
          setStatus('Cannot reach backend')
        }
      }
    }

    void checkBackend()

    return () => controller.abort()
  }, [])

  return (
    <main>
      <h1>FilingLens</h1>
      <p>Cited answers from SEC annual filings.</p>
      <p role="status">{status}</p>
    </main>
  )
}

export default App