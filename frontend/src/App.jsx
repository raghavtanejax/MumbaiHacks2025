import { useState } from 'react'
import './index.css'

function App() {
  const [claim, setClaim] = useState('')
  const [imageBase64, setImageBase64] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null) // Added error state

  const analyzeClaim = async () => {
    if (!claim) return

    setLoading(true)
    setResult(null)
    setError(null) // Clear previous errors

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000) // 120 second timeout

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: claim,
          image_base64: null
        }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error analyzing claim:', error)
      if (error.name === 'AbortError') {
        setError("Request timed out. The server took too long to respond.")
      } else {
        setError("Failed to connect to the server. Please ensure the backend is running.")
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header style={{ textAlign: 'center', marginBottom: '4rem', paddingTop: '2rem' }}>
        <h1>VERITAS HEALTH AGENT</h1>
        <p style={{ fontSize: '1.25rem', color: 'var(--color-text-muted)', maxWidth: '600px', margin: '0 auto' }}>
          Neutralizing health misinformation with AI-powered analysis.
        </p>
      </header>

      <main style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* How it works section */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '3rem', textAlign: 'center' }}>
          <div className="card glass" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📝</div>
            <h3 style={{ fontSize: '1.1rem' }}>1. Submit Claim</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>Paste text to analyze.</p>
          </div>
          <div className="card glass" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🤖</div>
            <h3 style={{ fontSize: '1.1rem' }}>2. AI Analysis</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>Multi-modal agent checks facts.</p>
          </div>
          <div className="card glass" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</div>
            <h3 style={{ fontSize: '1.1rem' }}>3. Verification</h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>Get verdict & corrective info.</p>
          </div>
        </div>

        <div className="card glass">
          <h2 style={{ marginBottom: '1.5rem' }}>Analyze a Health Claim</h2>
          <textarea
            rows="4"
            placeholder="Paste a health claim here (e.g., 'Drinking lemon water cures cancer')..."
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
          />

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button
              className="btn btn-primary"
              onClick={analyzeClaim}
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Verify Claim'}
            </button>
          </div>
        </div>

        {error && (
          <div className="card" style={{ marginTop: '2rem', borderLeft: '4px solid var(--color-danger)', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
            <h3 style={{ color: 'var(--color-danger)', margin: 0 }}>Error</h3>
            <p style={{ margin: '0.5rem 0 0 0' }}>{error}</p>
          </div>
        )}

        {result && (
          <div className="card" style={{
            marginTop: '2rem', borderTop: `4px solid ${result.verdict.includes('True') ? 'var(--color-success)' :
              result.verdict.includes('False') || result.verdict.includes('Misleading') ? 'var(--color-danger)' : 'var(--color-warning)'
              }`
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.5rem' }}>
                  {result.verdict}
                </h3>
                <span style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)' }}>
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ fontSize: '3rem' }}>
                {result.verdict.includes('True') ? '✅' :
                  result.verdict.includes('False') || result.verdict.includes('Misleading') ? '⚠️' : '❓'}
              </div>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: 'var(--color-primary)', marginBottom: '0.5rem' }}>AI Analysis</h4>
              <p style={{ lineHeight: '1.6' }}>{result.explanation}</p>
            </div>

            {result.corrective_information && (
              <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                <h4 style={{ color: 'var(--color-success)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>💡</span> Corrective Information
                </h4>
                <p style={{ margin: 0 }}>{result.corrective_information}</p>
              </div>
            )}

            {result.sources && result.sources.length > 0 && (
              <div style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '1rem' }}>
                <h4 style={{ fontSize: '0.9rem', color: 'var(--color-text-muted)', marginBottom: '0.5rem' }}>Sources</h4>
                <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
                  {result.sources.map((source, index) => (
                    <li key={index} style={{ color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>{source}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
