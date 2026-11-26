import { useRef, useState } from 'react'

const ACCEPTED = ['.mp4', '.mov', '.mkv', '.webm', '.wav', '.mp3', '.m4a', '.flac']

function prettySize(bytes) {
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`
}

export default function Dropzone({ file, onFile, disabled = false, maxMb = 200 }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [localError, setLocalError] = useState(null)

  function accept(candidate) {
    if (!candidate) return
    const suffix = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPTED.includes(suffix)) {
      setLocalError(`${suffix || 'That file type'} is not supported.`)
      return
    }
    if (candidate.size > maxMb * 1024 * 1024) {
      setLocalError(`That file is larger than the ${maxMb} MB limit.`)
      return
    }
    setLocalError(null)
    onFile(candidate)
  }

  function onDrop(event) {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    accept(event.dataTransfer.files?.[0])
  }

  if (file) {
    return (
      <div>
        <div className="filechip">
          <span className="name">{file.name}</span>
          <span className="meta">{prettySize(file.size)}</span>
          <button
            type="button"
            className="small"
            onClick={() => {
              onFile(null)
              if (inputRef.current) inputRef.current.value = ''
            }}
            disabled={disabled}
          >
            Remove
          </button>
        </div>
        {localError ? <p className="hint">{localError}</p> : null}
      </div>
    )
  }

  return (
    <div>
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            if (!disabled) inputRef.current?.click()
          }
        }}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <strong>Drop a recording here, or click to choose</strong>
        <div className="types">
          {ACCEPTED.join(' · ')} — up to {maxMb} MB
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        className="sr-only"
        accept={ACCEPTED.join(',')}
        disabled={disabled}
        onChange={(event) => accept(event.target.files?.[0])}
      />
      {localError ? (
        <p className="hint" style={{ color: 'var(--status-bad)' }}>
          {localError}
        </p>
      ) : null}
    </div>
  )
}
