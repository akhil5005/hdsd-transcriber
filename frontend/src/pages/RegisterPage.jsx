import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'
import { Banner } from '../components/common.jsx'
import { useAuth } from '../context/AuthContext.jsx'

const MIN_PASSWORD = 8

export default function RegisterPage() {
  const { signIn } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', full_name: '', password: '' })
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  function update(field) {
    return (event) => setForm((prev) => ({ ...prev, [field]: event.target.value }))
  }

  async function submit(event) {
    event.preventDefault()
    if (form.password.length < MIN_PASSWORD) {
      setError(`Password must be at least ${MIN_PASSWORD} characters.`)
      return
    }
    setBusy(true)
    setError(null)
    try {
      await apiCalls.register(form)
      // Registering does not issue tokens, so sign in straight away.
      await signIn(form.email, form.password)
      navigate('/transcribe', { replace: true })
    } catch (err) {
      setError(errorMessage(err, 'Could not register.'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-shell">
      <form className="auth-card" onSubmit={submit}>
        <h1>Create an account</h1>
        <p className="sub">
          You will be able to upload recordings and see your own transcripts only.
        </p>

        {error ? <Banner kind="error">{error}</Banner> : null}

        <div className="field" style={{ marginTop: error ? '0.85rem' : 0 }}>
          <label htmlFor="full_name">Full name</label>
          <input id="full_name" required value={form.full_name} onChange={update('full_name')} />
        </div>

        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            autoComplete="username"
            required
            value={form.email}
            onChange={update('email')}
          />
        </div>

        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            required
            minLength={MIN_PASSWORD}
            value={form.password}
            onChange={update('password')}
          />
          <span className="hint">At least {MIN_PASSWORD} characters.</span>
        </div>

        <button type="submit" className="primary" disabled={busy} style={{ width: '100%' }}>
          {busy ? 'Creating…' : 'Create account'}
        </button>

        <p className="auth-foot">
          Already registered? <Link to="/login">Sign in</Link>
        </p>
      </form>
    </div>
  )
}
