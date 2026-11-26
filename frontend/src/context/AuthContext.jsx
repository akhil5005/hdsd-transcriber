import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { setAuthLostHandler, tokens } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  // `loading` covers the initial "do we already have a valid session?" check,
  // so protected routes can wait instead of flashing the login screen.
  const [loading, setLoading] = useState(true)

  const clearSession = useCallback(() => {
    tokens.clear()
    setUser(null)
  }, [])

  useEffect(() => {
    setAuthLostHandler(clearSession)
  }, [clearSession])

  useEffect(() => {
    let cancelled = false

    async function restore() {
      if (!tokens.access && !tokens.refresh) {
        setLoading(false)
        return
      }
      try {
        const profile = await apiCalls.me()
        if (!cancelled) setUser(profile)
      } catch {
        // The axios interceptor already tried to refresh; if we are here the
        // session is genuinely gone.
        if (!cancelled) clearSession()
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    restore()
    return () => {
      cancelled = true
    }
  }, [clearSession])

  const signIn = useCallback(async (email, password) => {
    const data = await apiCalls.login(email, password)
    setUser(data.user)
    return data.user
  }, [])

  const signOut = useCallback(async () => {
    try {
      await apiCalls.logout()
    } finally {
      setUser(null)
    }
  }, [])

  const value = useMemo(
    () => ({ user, loading, signIn, signOut, isAuthenticated: Boolean(user) }),
    [user, loading, signIn, signOut],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside <AuthProvider>')
  return context
}
