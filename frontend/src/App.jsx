import { Suspense, lazy } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import ProtectedRoute from './components/ProtectedRoute.jsx'
import { AuthProvider } from './context/AuthContext.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import RegisterPage from './pages/RegisterPage.jsx'
import TranscribePage from './pages/TranscribePage.jsx'
import { Spinner } from './components/common.jsx'

// Recharts is ~450 kB of the bundle and only the benchmark page needs it.
const BenchmarkPage = lazy(() => import('./pages/BenchmarkPage.jsx'))

export default function App() {
  return (
    <BrowserRouter
      // Opt in early so the console stays clean on react-router 6.
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route path="/transcribe" element={<TranscribePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route
              path="/benchmark"
              element={
                <Suspense
                  fallback={
                    <div className="card">
                      <div className="progress">
                        <Spinner /> Loading charts…
                      </div>
                    </div>
                  }
                >
                  <BenchmarkPage />
                </Suspense>
              }
            />
          </Route>

          <Route path="/" element={<Navigate to="/transcribe" replace />} />
          <Route path="*" element={<Navigate to="/transcribe" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
