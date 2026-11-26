import { useEffect, useState } from 'react'

/** The series palette, mirroring the validated CSS custom properties. */
export const MODEL_COLOR = {
  pretrained: 'var(--series-pretrained)',
  full_ft: 'var(--series-full_ft)',
  lora: 'var(--series-lora)',
}

export const MODEL_SHORT = {
  pretrained: 'Pretrained',
  full_ft: 'Full FT',
  lora: 'LoRA',
}

export const MODEL_ORDER = ['pretrained', 'full_ft', 'lora']

export function Swatch({ modelKey, color }) {
  return (
    <span
      className="swatch"
      style={{ background: color || MODEL_COLOR[modelKey] || 'var(--border-strong)' }}
      aria-hidden="true"
    />
  )
}

export function Banner({ kind = 'info', children }) {
  if (!children) return null
  return <div className={`banner ${kind}`}>{children}</div>
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />
}

export function StatusBadge({ status }) {
  return <span className={`badge ${status}`}>{status}</span>
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>
}

/** Formats a number to fixed decimals, or an em dash when absent. */
export function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
}

export function fmtInt(value) {
  if (value === null || value === undefined) return '—'
  return Number(value).toLocaleString()
}

export function fmtBytes(bytes) {
  if (!bytes) return '—'
  const mb = bytes / (1024 * 1024)
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return '—'
  const total = Math.round(seconds)
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return mins ? `${mins}m ${secs}s` : `${seconds.toFixed(1)}s`
}

export function fmtDateTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

/** A labelled metric tile. */
export function Metric({ label, value, unit, highlight = false, title }) {
  return (
    <div className={`metric${highlight ? ' good' : ''}`} title={title}>
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit ? <span className="unit"> {unit}</span> : null}
      </div>
    </div>
  )
}

/** Copy-to-clipboard button with a short confirmation. */
export function CopyButton({ text, label = 'Copy' }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return undefined
    const timer = setTimeout(() => setCopied(false), 1600)
    return () => clearTimeout(timer)
  }, [copied])

  async function copy() {
    try {
      await navigator.clipboard.writeText(text || '')
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return (
    <button type="button" className="small" onClick={copy} disabled={!text}>
      {copied ? 'Copied' : label}
    </button>
  )
}

/**
 * Chart wrapper that also exposes the underlying numbers as a table.
 *
 * Not decoration: two slots in the light palette fall below 3:1 against the
 * surface, and the palette's relief rule requires visible labels or a table
 * view wherever that is true.
 */
export function ChartCard({ title, note, children, table, defaultTable = false }) {
  const [showTable, setShowTable] = useState(defaultTable)

  return (
    <section className="card">
      <div className="card-head">
        <h3>{title}</h3>
        {table ? (
          <button type="button" className="small" onClick={() => setShowTable((v) => !v)}>
            {showTable ? 'Show chart' : 'Show table'}
          </button>
        ) : null}
      </div>
      {showTable && table ? <div className="table-wrap">{table}</div> : (
        <div className="chart-box">{children}</div>
      )}
      {note ? <p className="card-note">{note}</p> : null}
    </section>
  )
}

/** Explicit legend; identity is never carried by colour alone. */
export function SeriesLegend({ items }) {
  return (
    <div className="chart-legend">
      {items.map((item) => (
        <span className="item" key={item.label}>
          <Swatch color={item.color} />
          {item.label}
        </span>
      ))}
    </div>
  )
}
