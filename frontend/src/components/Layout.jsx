import { useEffect, useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import * as apiCalls from '../api/endpoints.js'
import { useAuth } from '../context/AuthContext.jsx'
import { Banner } from './common.jsx'

const DISCLAIMER =
  'Research prototype. Exploratory results on the HDSD dataset. Not for diagnosis or clinical decisions.'

const LINKS = [
  { to: '/transcribe', label: 'Transcribe' },
  { to: '/history', label: 'History' },
  { to: '/benchmark', label: 'Benchmark' },
]

export default function Layout() {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()
  const [device, setDevice] = useState(null)
  const [backendDown, setBackendDown] = useState(false)

  useEffect(() => {
    let cancelled = false
    apiCalls
      .deviceInfo()
      .then((info) => {
        if (!cancelled) {
          setDevice(info)
          setBackendDown(false)
        }
      })
      .catch(() => {
        if (!cancelled) setBackendDown(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function handleSignOut() {
    await signOut()
    navigate('/login', { replace: true })
  }

  const cpuOnly = device && device.resolved_device !== 'cuda'

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <strong>HDSD Hindi Transcriber</strong>
            <span>Dysarthric speech · Whisper-small · three models</span>
          </div>
          <nav className="nav">
            {LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) => (isActive ? 'active' : undefined)}
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <div className="whoami">
            <span>{user?.full_name || user?.email}</span>
            <button type="button" className="small" onClick={handleSignOut}>
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="banner-stack">
          <div className="disclaimer">{DISCLAIMER}</div>

          {backendDown ? (
            <Banner kind="error">
              Cannot reach the backend. Start it with{' '}
              <code>uvicorn app.main:app --reload</code> in <code>backend/</code>.
            </Banner>
          ) : null}

          {cpuOnly ? (
            <Banner kind="warn">
              The backend reports <strong>CPU-only</strong> inference
              {device?.resolved_device ? ` (device: ${device.resolved_device})` : ''}.
              Transcription still works but will be noticeably slower — expect
              several seconds per 30-second window per model.
            </Banner>
          ) : null}
        </div>

        <Outlet context={{ device }} />
      </main>
    </div>
  )
}
