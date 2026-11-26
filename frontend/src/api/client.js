import axios from 'axios'

// In dev VITE_API_URL is blank and requests go to /api/v1/..., which the Vite
// dev server proxies to the backend (same origin, so no CORS). In a build it
// is the backend's public URL, baked in at build time.
const baseURL = `${(import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')}/api/v1`

const ACCESS_KEY = 'hdsd.access'
const REFRESH_KEY = 'hdsd.refresh'

export const tokens = {
  get access() {
    return localStorage.getItem(ACCESS_KEY)
  },
  get refresh() {
    return localStorage.getItem(REFRESH_KEY)
  },
  save({ access_token, refresh_token }) {
    if (access_token) localStorage.setItem(ACCESS_KEY, access_token)
    if (refresh_token) localStorage.setItem(REFRESH_KEY, refresh_token)
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY)
    localStorage.removeItem(REFRESH_KEY)
  },
}

export const api = axios.create({ baseURL, timeout: 120_000 })

api.interceptors.request.use((config) => {
  const access = tokens.access
  if (access) config.headers.Authorization = `Bearer ${access}`
  return config
})

// Called when refreshing fails, so the app can drop to the login screen.
let onAuthLost = () => {}
export function setAuthLostHandler(handler) {
  onAuthLost = handler
}

// A single in-flight refresh shared by every 401 that arrives while it runs.
// Without this, three parallel requests would each burn a refresh token, and
// the backend rotates them — the 2nd and 3rd would fail and log the user out.
let refreshing = null

function refreshOnce() {
  if (!refreshing) {
    refreshing = axios
      .post(`${baseURL}/auth/refresh`, { refresh_token: tokens.refresh })
      .then(({ data }) => {
        tokens.save(data)
        return data.access_token
      })
      .finally(() => {
        refreshing = null
      })
  }
  return refreshing
}

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const { response, config } = error

    // Only the token-issuing calls must skip the refresh-and-retry dance:
    // refreshing in response to their own 401 would recurse. Every other
    // /auth/ route — /auth/me in particular — is an ordinary protected
    // resource, and a plain `includes('/auth/')` test wrongly swept it up,
    // so restoring a session with an expired access token logged the user
    // out instead of refreshing.
    const isTokenCall = /\/auth\/(login|register|refresh|logout)/.test(config?.url || '')

    if (response?.status !== 401 || config?._retried || isTokenCall || !tokens.refresh) {
      return Promise.reject(error)
    }

    try {
      const access = await refreshOnce()
      config._retried = true
      config.headers = { ...config.headers, Authorization: `Bearer ${access}` }
      return api.request(config)
    } catch (refreshError) {
      tokens.clear()
      onAuthLost()
      return Promise.reject(refreshError)
    }
  },
)

/** Pull a readable message out of a FastAPI error response. */
export function errorMessage(error, fallback = 'Something went wrong.') {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length) {
    return detail.map((item) => item.msg || String(item)).join('; ')
  }
  if (error?.code === 'ECONNABORTED') return 'The request timed out.'
  if (error?.message === 'Network Error') {
    return 'Cannot reach the backend. Is it running?'
  }
  return error?.message || fallback
}

export default api
