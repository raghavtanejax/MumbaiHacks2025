import { useState } from 'react'
import './index.css'

function App() {
  const [claim, setClaim] = useState('')
  const [imageBase64, setImageBase64] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const analyzeClaim = async () => {
    if (!claim && !imageBase64) return
    setLoading(true)
    setResult(null)
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: claim,
          image_base64: imageBase64
        }),
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error analyzing claim:', error)
      alert('Failed to analyze claim. Make sure the backend is running.')
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
        <div className="card glass">
          <h2 style={{ marginBottom: '1.5rem' }}>Analyze a Health Claim</h2>
          <textarea
            rows="4"
            placeholder="Paste a health claim here (e.g., 'Drinking lemon water cures cancer')..."
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
          />

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--color-text-muted)' }}>
              Or upload an image (e.g., screenshot of a post)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) {
                  const reader = new FileReader();
                  reader.onloadend = () => {
                    setImageBase64(reader.result);
                    if (!claim) setClaim("Image uploaded");
                  };
                  reader.readAsDataURL(file);
                }
              }}
            />
          </div>

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

        {result && (
          <div className="card" style={{ marginTop: '2rem', borderLeft: '4px solid var(--color-primary)' }}>
            <h3>Analysis Result</h3>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Verdict: </strong>
              <span style={{
                color: result.verdict.includes('True') ? 'var(--color-success)' :
                  result.verdict.includes('False') ? 'var(--color-danger)' : 'var(--color-warning)',
                fontWeight: 'bold'
              }}>
                {result.verdict}
              </span>
            </div>
            <p>{result.explanation}</p>

            {result.sources && result.sources.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <h4>Sources</h4>
                <ul>
                  {result.sources.map((source, index) => (
                    <li key={index} style={{ color: 'var(--color-text-muted)' }}>{source}</li>
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
