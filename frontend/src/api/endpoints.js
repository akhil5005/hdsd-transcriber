import api, { tokens } from './client.js'

// --- auth -------------------------------------------------------------------
export async function register(payload) {
  const { data } = await api.post('/auth/register', payload)
  return data
}

export async function login(email, password) {
  const { data } = await api.post('/auth/login', { email, password })
  tokens.save(data)
  return data
}

export async function logout() {
  const refresh = tokens.refresh
  try {
    if (refresh) await api.post('/auth/logout', { refresh_token: refresh })
  } finally {
    tokens.clear()
  }
}

export async function me() {
  const { data } = await api.get('/auth/me')
  return data
}

// --- models / health --------------------------------------------------------
export async function listModels() {
  const { data } = await api.get('/models')
  return data
}

export async function deviceInfo() {
  const { data } = await api.get('/models/device')
  return data
}

export async function health() {
  const { data } = await api.get('/health')
  return data
}

// --- transcriptions ---------------------------------------------------------
export async function createJob({ file, modelKey, referenceText, onProgress }) {
  const form = new FormData()
  form.append('file', file)
  form.append('model_key', modelKey)
  if (referenceText && referenceText.trim()) {
    form.append('reference_text', referenceText)
  }

  const { data } = await api.post('/transcriptions', form, {
    // Let the browser set the multipart boundary.
    headers: { 'Content-Type': undefined },
    onUploadProgress: (event) => {
      if (onProgress && event.total) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    },
  })
  return data
}

export async function getJob(jobId) {
  const { data } = await api.get(`/transcriptions/${jobId}`)
  return data
}

export async function listJobs(params = {}) {
  const { data } = await api.get('/transcriptions', { params })
  return data
}

export async function deleteJob(jobId) {
  await api.delete(`/transcriptions/${jobId}`)
}

export async function retryJob(jobId) {
  const { data } = await api.post(`/transcriptions/${jobId}/retry`)
  return data
}

export async function aggregateMetrics() {
  const { data } = await api.get('/transcriptions/aggregate')
  return data
}

/** Download an export without leaving the page. */
export async function downloadExport(jobId, format) {
  const response = await api.get(`/transcriptions/${jobId}/export`, {
    params: { format },
    responseType: 'blob',
  })
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url
  link.download = `transcript-${jobId}.${format}`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

// --- benchmark --------------------------------------------------------------
export async function getBenchmark(params = {}) {
  const { data } = await api.get('/benchmark', { params })
  return data
}

export async function getBenchmarkSummary() {
  const { data } = await api.get('/benchmark/summary')
  return data
}
