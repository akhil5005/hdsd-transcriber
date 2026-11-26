import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'
import Dropzone from '../components/Dropzone.jsx'
import JobResults from '../components/JobResults.jsx'
import ModelSelector from '../components/ModelSelector.jsx'
import { Banner, Spinner } from '../components/common.jsx'

const POLL_MS = 1200

export default function TranscribePage() {
  const [file, setFile] = useState(null)
  const [modelKey, setModelKey] = useState('all')
  const [referenceText, setReferenceText] = useState('')
  const [uploadPercent, setUploadPercent] = useState(0)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [loadedModels, setLoadedModels] = useState([])

  const pollTimer = useRef(null)

  useEffect(() => {
    apiCalls
      .listModels()
      .then((rows) => setLoadedModels(rows.filter((row) => row.loaded).map((r) => r.key)))
      .catch(() => setLoadedModels([]))
  }, [job?.status])

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current)
      pollTimer.current = null
    }
  }, [])

  useEffect(() => stopPolling, [stopPolling])

  // A function declaration, not useCallback: poll re-arms itself through the
  // timer, and it is never a hook dependency, so a stable identity buys
  // nothing and only creates a self-reference the linter rightly flags.
  async function poll(jobId) {
    try {
      const next = await apiCalls.getJob(jobId)
      setJob(next)
      if (next.status === 'queued' || next.status === 'running') {
        pollTimer.current = setTimeout(() => poll(jobId), POLL_MS)
      } else {
        setBusy(false)
      }
    } catch (err) {
      setError(errorMessage(err, 'Lost track of the job.'))
      setBusy(false)
    }
  }

  async function submit(event) {
    event.preventDefault()
    if (!file) {
      setError('Choose a recording first.')
      return
    }

    stopPolling()
    setError(null)
    setJob(null)
    setBusy(true)
    setUploadPercent(0)

    try {
      const created = await apiCalls.createJob({
        file,
        modelKey,
        referenceText,
        onProgress: setUploadPercent,
      })
      setJob({ ...created, id: created.job_id, status: created.status })
      poll(created.job_id)
    } catch (err) {
      setError(errorMessage(err, 'Upload failed.'))
      setBusy(false)
    }
  }

  function reset() {
    stopPolling()
    setFile(null)
    setReferenceText('')
    setJob(null)
    setError(null)
    setBusy(false)
    setUploadPercent(0)
  }

  const running = job && (job.status === 'queued' || job.status === 'running')
  const finished = job && job.status === 'done'
  const failed = job && job.status === 'failed'

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Transcribe</h1>
        <p>
          Upload an MP4 or audio recording of a dysarthric speaker and get a Hindi
          transcript in Devanagari. Audio is converted to 16&nbsp;kHz mono and
          decoded greedily under the frozen research settings.
        </p>
      </div>

      {error ? <Banner kind="error">{error}</Banner> : null}

      <form className="card stack" onSubmit={submit}>
        <div>
          <h2 style={{ marginBottom: '0.5rem' }}>1 · Recording</h2>
          <Dropzone file={file} onFile={setFile} disabled={busy} />
        </div>

        <div>
          <h2 style={{ marginBottom: '0.5rem' }}>2 · Model</h2>
          <ModelSelector
            value={modelKey}
            onChange={setModelKey}
            disabled={busy}
            loaded={loadedModels}
          />
        </div>

        <div>
          <h2 style={{ marginBottom: '0.5rem' }}>3 · Reference transcript (optional)</h2>
          <div className="field" style={{ marginBottom: 0 }}>
            <label htmlFor="reference">
              Ground truth in Devanagari — enables WER, CER and the word-level diff
            </label>
            <textarea
              id="reference"
              className="devanagari"
              lang="hi"
              placeholder="यहाँ संदर्भ प्रतिलेख लिखें…"
              value={referenceText}
              onChange={(event) => setReferenceText(event.target.value)}
              disabled={busy}
            />
            <span className="hint">
              Leave this blank to see transcripts, timings and pairwise agreement
              between models instead.
            </span>
          </div>
        </div>

        <div className="button-row">
          <button type="submit" className="primary" disabled={busy || !file}>
            {busy ? 'Working…' : modelKey === 'all' ? 'Transcribe with all 3' : 'Transcribe'}
          </button>
          <button type="button" onClick={reset} disabled={busy && !failed}>
            Reset
          </button>
        </div>
      </form>

      {busy && uploadPercent > 0 && uploadPercent < 100 ? (
        <div className="card">
          <div className="progress">
            <Spinner />
            <span>Uploading… {uploadPercent}%</span>
            <span className="track">
              <i style={{ width: `${uploadPercent}%`, animation: 'none', marginLeft: 0 }} />
            </span>
          </div>
        </div>
      ) : null}

      {running ? (
        <div className="card">
          <div className="progress">
            <Spinner />
            <span>
              {job.status === 'queued' ? 'Queued' : 'Transcribing'}
              {modelKey === 'all' ? ' with three models' : ''}…
            </span>
            <span className="track">
              <i />
            </span>
          </div>
          <p className="card-note">
            Inference runs one model at a time so the GPU is never asked to hold
            two decodes at once. A short clip through all three models usually
            takes a few seconds.
          </p>
        </div>
      ) : null}

      {failed ? (
        <Banner kind="error">
          {job.error || 'The job failed.'}{' '}
          <button
            type="button"
            className="link"
            onClick={async () => {
              try {
                setBusy(true)
                await apiCalls.retryJob(job.id)
                poll(job.id)
              } catch (err) {
                setError(errorMessage(err))
                setBusy(false)
              }
            }}
          >
            Retry
          </button>
        </Banner>
      ) : null}

      {finished ? <JobResults job={job} /> : null}
    </div>
  )
}
