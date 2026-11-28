import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'
import Dropzone from '../components/Dropzone.jsx'
import Recorder from '../components/Recorder.jsx'
import JobResults from '../components/JobResults.jsx'
import ModelSelector from '../components/ModelSelector.jsx'
import { Banner, Spinner } from '../components/common.jsx'

const POLL_MS = 1200

export default function TranscribePage() {
  const [file, setFile] = useState(null)
  const [inputMode, setInputMode] = useState('upload')
  const [datasetPrompt, setDatasetPrompt] = useState(null)
  //: Where `file` came from. The two input modes share one file, so without
  //: this the recorder describes an uploaded file as something it captured.
  const [fileSource, setFileSource] = useState(null)
  //: Sentence id seen in the filename, even when it is not one we hold.
  //: Lets us say "H99 is not seeded" rather than "not a corpus file".
  const [unknownPromptId, setUnknownPromptId] = useState(null)
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

  // A corpus filename tells us the sentence before anything is uploaded, so
  // the reference box can be replaced with the actual ground truth instead of
  // asking for something we already know.
  useEffect(() => {
    const promptId = apiCalls.promptIdFromFilename(file?.name)
    if (!promptId) {
      setDatasetPrompt(null)
      setUnknownPromptId(null)
      return
    }
    let cancelled = false
    apiCalls
      .getPrompt(promptId)
      .then((prompt) => {
        if (cancelled) return
        setDatasetPrompt(prompt)
        setUnknownPromptId(null)
      })
      .catch(() => {
        // The filename names a sentence we do not hold -- a partially seeded
        // table, or a file from outside this corpus. Fall back to asking,
        // but say which id went unmatched rather than claiming the file is
        // not a corpus file at all.
        if (cancelled) return
        setDatasetPrompt(null)
        setUnknownPromptId(promptId)
      })
    return () => {
      cancelled = true
    }
  }, [file])

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
        // Only send what the user can actually see. When a corpus sentence is
        // showing, the textarea is not rendered, so anything still in
        // referenceText is left over from an earlier file -- and the backend
        // treats a typed reference as authoritative, which would score the
        // job against text the user had no way to read or correct while the
        // form promised them the dataset sentence.
        referenceText: datasetPrompt ? '' : referenceText,
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

          <div className="input-tabs" role="group" aria-label="How to supply audio">
            <button
              type="button"
              aria-pressed={inputMode === 'upload'}
              onClick={() => setInputMode('upload')}
              disabled={busy}
            >
              Upload a file
            </button>
            <button
              type="button"
              aria-pressed={inputMode === 'record'}
              onClick={() => setInputMode('record')}
              disabled={busy}
            >
              Record with microphone
            </button>
          </div>

          {inputMode === 'upload' ? (
            <>
              <Dropzone
                file={file}
                onFile={(chosen) => {
                  setFile(chosen)
                  setFileSource(chosen ? 'upload' : null)
                }}
                disabled={busy}
              />
              <span className="hint">
                A file straight from the HDSD corpus (for example
                M22_S1_H15_M1.wav) is recognised by its name, and its sentence is
                used as ground truth automatically.
              </span>
            </>
          ) : (
            <>
              <Recorder
                onRecorded={(captured) => {
                  setFile(captured)
                  setFileSource('record')
                }}
                disabled={busy}
              />
              {file ? (
                <p className="hint">
                  {fileSource === 'record'
                    ? `Captured ${file.name} (${Math.round(file.size / 1024)} KB). Record again to replace it.`
                    : `Still using the uploaded file ${file.name} (${Math.round(file.size / 1024)} KB). Recording replaces it.`}
                </p>
              ) : null}
            </>
          )}
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

        {datasetPrompt ? (
          // Ground truth is already known, so showing an empty box to type it
          // into would be busywork. Show the sentence instead.
          <div>
            <h2 style={{ marginBottom: '0.5rem' }}>3 · Ground truth</h2>
            <div className="card inset">
              <div className="card-head">
                <span className="gt-chip">from dataset · {datasetPrompt.prompt_id}</span>
                {datasetPrompt.verified ? null : (
                  <span className="muted">not yet verified by a person</span>
                )}
              </div>
              <p className="devanagari gt-sentence" lang="hi">
                {datasetPrompt.devanagari}
              </p>
              <p className="hint">
                Recognised from the filename, so WER, CER and the word-level diff
                are scored automatically — nothing to type.
                {datasetPrompt.verified
                  ? ''
                  : ' This sentence was transcribed from the corpus’s romanised labels and has not been checked yet, so treat the scores as provisional.'}
              </p>
            </div>
          </div>
        ) : (
          <div>
            <h2 style={{ marginBottom: '0.5rem' }}>
              3 · Reference transcript {inputMode === 'record' ? '(optional)' : ''}
            </h2>
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
                {inputMode === 'record'
                  ? 'A recording is new speech with no sentence id, so type what was said if you want WER and CER. Leave it blank to just get the transcript.'
                  : unknownPromptId
                    ? `The filename names sentence ${unknownPromptId}, but that sentence is not in the ground-truth table. Type it to get WER and CER, or leave blank for transcripts only.`
                    : file
                      ? 'This file is not from the corpus, so there is no ground truth to look up. Type it to get WER and CER, or leave blank for transcripts only.'
                      : 'Choose a corpus file and its sentence is filled in for you. For anything else, type the ground truth here.'}
              </span>
            </div>
          </div>
        )}

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
