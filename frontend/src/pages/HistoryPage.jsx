import { useCallback, useEffect, useState } from 'react'
import { errorMessage } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'
import JobResults from '../components/JobResults.jsx'
import {
  Banner,
  Empty,
  MODEL_ORDER,
  MODEL_SHORT,
  Metric,
  Spinner,
  StatusBadge,
  Swatch,
  fmt,
  fmtDateTime,
  fmtInt,
} from '../components/common.jsx'

const PAGE_SIZE = 20

export default function HistoryPage() {
  const [filters, setFilters] = useState({
    model_key: '',
    status: '',
    created_after: '',
    created_before: '',
  })
  const [page, setPage] = useState({ items: [], total: 0, offset: 0 })
  const [aggregate, setAggregate] = useState([])
  const [expanded, setExpanded] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(
    async (offset = 0) => {
      setLoading(true)
      setError(null)
      try {
        const params = { limit: PAGE_SIZE, offset }
        if (filters.model_key) params.model_key = filters.model_key
        if (filters.status) params.status = filters.status
        if (filters.created_after) {
          params.created_after = new Date(filters.created_after).toISOString()
        }
        if (filters.created_before) {
          // Include the whole selected day.
          const end = new Date(filters.created_before)
          end.setHours(23, 59, 59, 999)
          params.created_before = end.toISOString()
        }
        const [jobs, totals] = await Promise.all([
          apiCalls.listJobs(params),
          apiCalls.aggregateMetrics(),
        ])
        setPage({ ...jobs, offset })
        setAggregate(totals)
      } catch (err) {
        setError(errorMessage(err, 'Could not load history.'))
      } finally {
        setLoading(false)
      }
    },
    [filters],
  )

  useEffect(() => {
    load(0)
  }, [load])

  async function toggle(jobId) {
    if (expanded === jobId) {
      setExpanded(null)
      setDetail(null)
      return
    }
    setExpanded(jobId)
    setDetail(null)
    try {
      setDetail(await apiCalls.getJob(jobId))
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  async function remove(jobId) {
    if (!window.confirm('Delete this job, its transcripts and the uploaded file?')) {
      return
    }
    try {
      await apiCalls.deleteJob(jobId)
      if (expanded === jobId) {
        setExpanded(null)
        setDetail(null)
      }
      await load(page.offset)
    } catch (err) {
      setError(errorMessage(err, 'Could not delete the job.'))
    }
  }

  function update(field) {
    return (event) => setFilters((prev) => ({ ...prev, [field]: event.target.value }))
  }

  const hasNext = page.offset + PAGE_SIZE < page.total
  const hasPrev = page.offset > 0

  return (
    <div className="stack">
      <div className="page-head">
        <h1>History</h1>
        <p>Your previous jobs. Only you can see them; deleting one also removes the uploaded file.</p>
      </div>

      {error ? <Banner kind="error">{error}</Banner> : null}

      {aggregate.length ? (
        <section className="card">
          <div className="card-head">
            <h2>Live metrics across your scored jobs</h2>
          </div>
          <div className="chart-grid">
            {[...aggregate]
              .sort(
                (a, b) =>
                  MODEL_ORDER.indexOf(a.model_key) - MODEL_ORDER.indexOf(b.model_key),
              )
              .map((row) => (
                <div key={row.model_key}>
                  <h3 style={{ marginBottom: '0.4rem' }}>
                    <Swatch modelKey={row.model_key} />{' '}
                    {MODEL_SHORT[row.model_key] || row.model_key}
                  </h3>
                  <div className="metric-row">
                    <Metric label="Jobs scored" value={fmtInt(row.scored_jobs)} />
                    <Metric label="Mean WER" value={fmt(row.mean_wer)} unit="%" />
                    <Metric label="Mean CER" value={fmt(row.mean_cer)} unit="%" />
                    <Metric label="Mean RTF" value={fmt(row.mean_rtf)} />
                  </div>
                </div>
              ))}
          </div>
          <p className="card-note">
            Averaged over jobs where you supplied a reference transcript. Jobs
            without one contribute no WER or CER, because there is nothing to
            score against.
          </p>
        </section>
      ) : null}

      <section className="card">
        <div className="filters">
          <div className="field">
            <label htmlFor="f-model">Model</label>
            <select id="f-model" value={filters.model_key} onChange={update('model_key')}>
              <option value="">Any</option>
              {MODEL_ORDER.map((key) => (
                <option key={key} value={key}>
                  {MODEL_SHORT[key]}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="f-status">Status</label>
            <select id="f-status" value={filters.status} onChange={update('status')}>
              <option value="">Any</option>
              {['queued', 'running', 'done', 'failed'].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="f-after">From</label>
            <input
              id="f-after"
              type="date"
              value={filters.created_after}
              onChange={update('created_after')}
            />
          </div>
          <div className="field">
            <label htmlFor="f-before">To</label>
            <input
              id="f-before"
              type="date"
              value={filters.created_before}
              onChange={update('created_before')}
            />
          </div>
          <button
            type="button"
            onClick={() =>
              setFilters({ model_key: '', status: '', created_after: '', created_before: '' })
            }
          >
            Clear
          </button>
        </div>

        {loading ? (
          <div className="progress">
            <Spinner /> Loading…
          </div>
        ) : page.items.length === 0 ? (
          <Empty>No jobs match these filters.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Created</th>
                  <th>File</th>
                  <th>Mode</th>
                  <th>Models</th>
                  <th>Status</th>
                  <th>Reference</th>
                  <th className="num">Audio</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {page.items.map((job) => (
                  <tr key={job.id}>
                    <td>{fmtDateTime(job.created_at)}</td>
                    <td title={job.media?.original_filename}>
                      {job.media?.original_filename || '—'}
                    </td>
                    <td>{job.mode}</td>
                    <td>
                      {job.model_keys.length
                        ? job.model_keys.map((key) => (
                            <span key={key} style={{ marginRight: 6 }}>
                              <Swatch modelKey={key} /> {MODEL_SHORT[key] || key}
                            </span>
                          ))
                        : job.requested_model_key
                          ? MODEL_SHORT[job.requested_model_key]
                          : '—'}
                    </td>
                    <td>
                      <StatusBadge status={job.status} />
                    </td>
                    <td>{job.has_reference ? 'yes' : 'no'}</td>
                    <td className="num">{fmt(job.media?.duration_sec, 1)}s</td>
                    <td>
                      <div className="button-row">
                        <button
                          type="button"
                          className="small"
                          onClick={() => toggle(job.id)}
                          disabled={job.status !== 'done'}
                        >
                          {expanded === job.id ? 'Hide' : 'View'}
                        </button>
                        <button
                          type="button"
                          className="small danger"
                          onClick={() => remove(job.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="row-between" style={{ marginTop: '0.75rem' }}>
          <span className="muted">
            {page.total} job{page.total === 1 ? '' : 's'}
          </span>
          <div className="button-row">
            <button
              type="button"
              className="small"
              disabled={!hasPrev}
              onClick={() => load(Math.max(0, page.offset - PAGE_SIZE))}
            >
              Previous
            </button>
            <button
              type="button"
              className="small"
              disabled={!hasNext}
              onClick={() => load(page.offset + PAGE_SIZE)}
            >
              Next
            </button>
          </div>
        </div>
      </section>

      {expanded ? (
        detail ? (
          <JobResults job={detail} />
        ) : (
          <div className="card">
            <div className="progress">
              <Spinner /> Loading result…
            </div>
          </div>
        )
      ) : null}
    </div>
  )
}
