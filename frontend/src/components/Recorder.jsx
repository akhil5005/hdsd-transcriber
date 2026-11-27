import { useCallback, useEffect, useRef, useState } from 'react'
import { Banner } from './common.jsx'

/** Prefer Opus in WebM: small, and the backend already accepts .webm. */
function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
  ]
  if (typeof MediaRecorder === 'undefined') return null
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || ''
}

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
}

function clock(ms) {
  const total = Math.floor(ms / 1000)
  const mm = String(Math.floor(total / 60)).padStart(2, '0')
  const ss = String(total % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

/**
 * Record from the microphone, with a live bar spectrum so the speaker can see
 * their voice registering.
 *
 * The visualiser earns its place here: a dysarthric speaker needs to know the
 * mic is picking them up without waiting for a transcript to come back. Quiet
 * input shows as flat bars straight away.
 *
 * Produces a File, so it feeds the same upload path as a chosen file.
 */
export default function Recorder({ onRecorded, disabled = false }) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState(null)
  const [level, setLevel] = useState(0)

  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const audioCtxRef = useRef(null)
  const rafRef = useRef(null)
  const startedAtRef = useRef(0)
  const chunksRef = useRef([])

  /** Release the mic, the audio graph and the animation frame. */
  const teardown = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (streamRef.current) {
      // Without this the browser keeps showing its recording indicator.
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
    recorderRef.current = null
  }, [])

  useEffect(() => teardown, [teardown])

  function draw(analyser) {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const bins = new Uint8Array(analyser.frequencyBinCount)
    const styles = getComputedStyle(canvas)
    const accent = styles.getPropertyValue('--accent').trim() || '#1f5f8b'
    const faint = styles.getPropertyValue('--border').trim() || '#dcdbd6'

    const render = () => {
      rafRef.current = requestAnimationFrame(render)
      analyser.getByteFrequencyData(bins)

      // Canvas is laid out in CSS pixels; match the backing store to the DPR
      // so the bars are not blurry on a scaled display.
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr
        canvas.height = h * dpr
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      // Baseline, so silence reads as "listening" rather than "broken".
      ctx.fillStyle = faint
      ctx.fillRect(0, h / 2 - 0.5, w, 1)

      const bars = 48
      const step = Math.floor(bins.length / bars) || 1
      const barWidth = w / bars
      let sum = 0
      ctx.fillStyle = accent
      for (let i = 0; i < bars; i += 1) {
        let peak = 0
        for (let j = 0; j < step; j += 1) {
          peak = Math.max(peak, bins[i * step + j] || 0)
        }
        sum += peak
        // Mirrored around the centre line: the familiar voice-bar look.
        const barHeight = Math.max(2, (peak / 255) * (h * 0.9))
        const x = i * barWidth + barWidth * 0.2
        ctx.fillRect(x, h / 2 - barHeight / 2, barWidth * 0.6, barHeight)
      }
      setLevel(sum / bars / 255)
      setElapsed(Date.now() - startedAtRef.current)
    }
    render()
  }

  async function start() {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser cannot record audio.')
      return
    }
    const mimeType = pickMimeType()
    if (mimeType === null) {
      setError('This browser does not support MediaRecorder.')
      return
    }

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Left off on purpose: these are clinical recordings, and browser
          // clean-up would alter the very speech characteristics under study.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
    } catch {
      // Denied, dismissed, or no input device all surface the same way.
      setError(
        'Could not use the microphone. Allow microphone access for this site, then try again.',
      )
      return
    }

    streamRef.current = stream
    chunksRef.current = []

    const AudioCtx = window.AudioContext || window.webkitAudioContext
    const audioCtx = new AudioCtx()
    audioCtxRef.current = audioCtx
    const source = audioCtx.createMediaStreamSource(stream)
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 256
    analyser.smoothingTimeConstant = 0.75
    source.connect(analyser)

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    recorderRef.current = recorder
    recorder.ondataavailable = (event) => {
      if (event.data?.size) chunksRef.current.push(event.data)
    }
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || 'audio/webm'
      const blob = new Blob(chunksRef.current, { type })
      const extension = type.includes('ogg') ? 'ogg' : 'webm'
      // A plain name carrying no HDSD sentence id, so this can never be
      // mistaken for a corpus file by the ground-truth lookup.
      const file = new File([blob], `recording-${stamp()}.${extension}`, { type })
      teardown()
      setRecording(false)
      setLevel(0)
      if (blob.size > 0) onRecorded(file)
      else setError('Nothing was recorded. Check the microphone and try again.')
    }

    startedAtRef.current = Date.now()
    setElapsed(0)
    recorder.start()
    setRecording(true)
    draw(analyser)
  }

  function stop() {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    } else {
      teardown()
      setRecording(false)
    }
  }

  return (
    <div className="recorder">
      <div className="recorder-head">
        <button
          type="button"
          className={recording ? 'danger' : ''}
          onClick={recording ? stop : start}
          disabled={disabled}
        >
          {recording ? 'Stop recording' : 'Start recording'}
        </button>
        <span className="recorder-time">{clock(elapsed)}</span>
        {recording ? (
          <span className="recorder-live">
            <span className="recorder-dot" aria-hidden="true" />
            {level < 0.02 ? 'very quiet — speak up' : 'listening'}
          </span>
        ) : (
          <span className="muted">Speak the sentence, then stop.</span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className={`waveform${recording ? ' live' : ''}`}
        role="img"
        aria-label={
          recording ? 'Live microphone level' : 'Microphone level, not recording'
        }
      />

      <Banner kind="error">{error}</Banner>
    </div>
  )
}
