import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'
import { Spinner } from './common.jsx'

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth()
  const location = useLocation()

  // Wait for the initial session check; redirecting first would bounce a
  // signed-in user to the login screen on every refresh.
  if (loading) {
    return (
      <div className="auth-shell">
        <div className="progress">
          <Spinner />
          Checking your session…
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return children
}
