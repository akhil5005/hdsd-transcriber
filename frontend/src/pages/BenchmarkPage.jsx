import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { errorMessage } from '../api/client.js'
import * as apiCalls from '../api/endpoints.js'
import {
  Banner,
  ChartCard,
  MODEL_COLOR,
  MODEL_ORDER,
  MODEL_SHORT,
  SeriesLegend,
  Spinner,
  Swatch,
  fmt,
  fmtBytes,
  fmtInt,
} from '../components/common.jsx'

/** Short axis labels; the full case names live in the table and the note. */
const SHORT_CASE = {
  test: 'Unfamiliar spk\n+ unfamiliar sent',
  test_speaker_train_prompt: 'Unfamiliar spk\n+ familiar sent',
  train_speaker_test_prompt: 'Familiar spk\n+ unfamiliar sent',
}

const AXIS = { stroke: 'var(--border-strong)', fontSize: 11 }
const GRID = 'var(--border)'

function TooltipBox({ active, payload, label, unit = '%' }) {
  if (!active || !payload?.length) return null
  return (
    <div
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-strong)',
        borderRadius: 6,
        padding: '0.5rem 0.65rem',
        fontSize: '0.8rem',
        boxShadow: 'var(--shadow)',
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.map((item) => (
        <div key={item.name} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span
            className="swatch"
            style={{ background: item.color || item.fill }}
            aria-hidden="true"
          />
          <span style={{ color: 'var(--text-secondary)' }}>{item.name}</span>
          <strong className="mono" style={{ marginLeft: 'auto' }}>
            {fmt(item.value)}
            {unit}
          </strong>
        </div>
      ))}
    </div>
  )
}

/** Grouped bars: one group per evaluation case, one bar per model. */
function GroupedByCase({ cases, series, unit = '%' }) {
  const data = cases.map((kase, index) => {
    const row = { case: SHORT_CASE[kase.split] || kase.label }
    series.forEach((entry) => {
      row[MODEL_SHORT[entry.model_key] || entry.model_key] = entry.values[index]
    })
    return row
  })

  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 18, right: 8, bottom: 26, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="case" {...AXIS} interval={0} tickLine={false} />
          <YAxis {...AXIS} tickLine={false} width={38} />
          <Tooltip content={<TooltipBox unit={unit} />} cursor={{ fill: 'var(--surface-2)' }} />
          {series.map((entry) => (
            <Bar
              key={entry.model_key}
              dataKey={MODEL_SHORT[entry.model_key] || entry.model_key}
              fill={MODEL_COLOR[entry.model_key]}
              radius={[4, 4, 0, 0]}
              maxBarSize={34}
            >
              {/* Direct labels: the relief rule for the low-contrast slots. */}
              <LabelList
                dataKey={MODEL_SHORT[entry.model_key] || entry.model_key}
                position="top"
                formatter={(value) => fmt(value, 1)}
                style={{ fill: 'var(--text-secondary)', fontSize: 10 }}
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <SeriesLegend
        items={MODEL_ORDER.map((key) => ({
          label: MODEL_SHORT[key],
          color: MODEL_COLOR[key],
        }))}
      />
    </>
  )
}

function CaseTable({ cases, series, unit = '%' }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Evaluation case</th>
          <th className="num">n</th>
          {MODEL_ORDER.map((key) => (
            <th className="num" key={key}>
              {MODEL_SHORT[key]} {unit}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {cases.map((kase, index) => (
          <tr key={kase.split}>
            <td>
              {kase.label}
              {kase.is_primary ? <span className="badge" style={{ marginLeft: 6 }}>primary</span> : null}
            </td>
            <td className="num">{kase.utterances}</td>
            {MODEL_ORDER.map((key) => {
              const entry = series.find((s) => s.model_key === key)
              return (
                <td className="num" key={key}>
                  {fmt(entry?.values[index])}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Mean WER across training seeds, with sample-SD error bars. */
function SeedSpread({ rows, cases }) {
  const primary = cases.find((kase) => kase.is_primary)?.split || 'test'
  const data = MODEL_ORDER.map((key) => {
    const row = rows.find((r) => r.model_key === key && r.split === primary)
    return {
      model: MODEL_SHORT[key] || key,
      modelKey: key,
      mean: row?.mean_wer ?? null,
      sd: row?.sd_wer ?? 0,
      seeds: row?.seed_count ?? 0,
    }
  })

  return (
    <>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 20, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="model" {...AXIS} tickLine={false} />
          <YAxis {...AXIS} tickLine={false} width={38} />
          <Tooltip content={<TooltipBox />} cursor={{ fill: 'var(--surface-2)' }} />
          <Bar dataKey="mean" radius={[4, 4, 0, 0]} maxBarSize={52} name="Mean WER">
            {data.map((row) => (
              <Cell key={row.modelKey} fill={MODEL_COLOR[row.modelKey]} />
            ))}
            {/* No direct label here: the error-bar cap sits exactly where a
                "top" label would go and the two collide. The legend below
                carries the numbers instead, and the table view has them all. */}
            <ErrorBar
              dataKey="sd"
              width={6}
              strokeWidth={2}
              stroke="var(--text-secondary)"
              direction="y"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="chart-legend">
        {data.map((row) => (
          <span className="item" key={row.modelKey}>
            <Swatch modelKey={row.modelKey} />
            {row.model} — <strong className="mono">{fmt(row.mean)}</strong>
            {row.sd ? ` ±${fmt(row.sd, 2)} pp` : ''} over {row.seeds} seed
            {row.seeds === 1 ? '' : 's'}
            {row.sd ? '' : ' (no SD)'}
          </span>
        ))}
      </div>
    </>
  )
}

function SeedTable({ rows, cases }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Evaluation case</th>
          <th className="num">seeds</th>
          <th className="num">mean WER %</th>
          <th className="num">SD pp</th>
          <th className="num">mean CER %</th>
          <th className="num">SD pp</th>
        </tr>
      </thead>
      <tbody>
        {cases.flatMap((kase) =>
          MODEL_ORDER.map((key) => {
            const row = rows.find((r) => r.model_key === key && r.split === kase.split)
            if (!row) return null
            return (
              <tr key={`${key}-${kase.split}`}>
                <td>
                  <Swatch modelKey={key} /> {MODEL_SHORT[key]}
                </td>
                <td>{kase.label}</td>
                <td className="num">{fmtInt(row.seed_count)}</td>
                <td className="num">{fmt(row.mean_wer)}</td>
                <td className="num">{row.sd_wer === null ? '—' : fmt(row.sd_wer)}</td>
                <td className="num">{fmt(row.mean_cer)}</td>
                <td className="num">{row.sd_cer === null ? '—' : fmt(row.sd_cer)}</td>
              </tr>
            )
          }),
        )}
      </tbody>
    </table>
  )
}

/** Stacked S/D/I counts, grouped per model, for one evaluation case. */
function ErrorBreakdown({ rows, cases }) {
  const [split, setSplit] = useState(cases.find((k) => k.is_primary)?.split || 'test')
  const forCase = rows.filter((row) => row.split === split)

  const data = MODEL_ORDER.map((key) => {
    const row = forCase.find((r) => r.model_key === key)
    return {
      model: MODEL_SHORT[key] || key,
      Substitutions: row?.substitutions ?? 0,
      Deletions: row?.deletions ?? 0,
      Insertions: row?.insertions ?? 0,
    }
  })

  const segments = [
    { key: 'Substitutions', color: MODEL_COLOR.pretrained },
    { key: 'Deletions', color: MODEL_COLOR.full_ft },
    { key: 'Insertions', color: MODEL_COLOR.lora },
  ]

  return (
    <>
      <div className="field" style={{ maxWidth: '22rem', marginBottom: '0.6rem' }}>
        <label htmlFor="sdi-case">Evaluation case</label>
        <select id="sdi-case" value={split} onChange={(event) => setSplit(event.target.value)}>
          {cases.map((kase) => (
            <option key={kase.split} value={kase.split}>
              {kase.label} (n={kase.utterances})
            </option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 12, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="model" {...AXIS} tickLine={false} />
          <YAxis {...AXIS} tickLine={false} width={46} />
          <Tooltip content={<TooltipBox unit="" />} cursor={{ fill: 'var(--surface-2)' }} />
          <Legend wrapperStyle={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }} />
          {segments.map((segment, index) => (
            <Bar
              key={segment.key}
              dataKey={segment.key}
              stackId="sdi"
              fill={segment.color}
              maxBarSize={52}
              // 2px surface gap between stacked segments.
              stroke="var(--surface-1)"
              strokeWidth={2}
              radius={index === segments.length - 1 ? [4, 4, 0, 0] : undefined}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="card-note">
        Counts, not rates. Colour here means <em>error type</em>, not model — the
        model is on the horizontal axis.
      </p>
    </>
  )
}

function SdiTable({ rows, cases }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Evaluation case</th>
          <th className="num">S</th>
          <th className="num">D</th>
          <th className="num">I</th>
          <th className="num">reference units</th>
        </tr>
      </thead>
      <tbody>
        {cases.flatMap((kase) =>
          MODEL_ORDER.map((key) => {
            const row = rows.find((r) => r.model_key === key && r.split === kase.split)
            if (!row) return null
            return (
              <tr key={`${key}-${kase.split}`}>
                <td>
                  <Swatch modelKey={key} /> {MODEL_SHORT[key]}
                </td>
                <td>{kase.label}</td>
                <td className="num">{fmtInt(row.substitutions)}</td>
                <td className="num">{fmtInt(row.deletions)}</td>
                <td className="num">{fmtInt(row.insertions)}</td>
                <td className="num">{fmtInt(row.reference_units)}</td>
              </tr>
            )
          }),
        )}
      </tbody>
    </table>
  )
}

function EfficiencyTable({ rows, selectedRuns }) {
  const selected = new Set(Object.values(selectedRuns || {}))
  const ordered = [...rows].sort(
    (a, b) =>
      MODEL_ORDER.indexOf(a.model_key) - MODEL_ORDER.indexOf(b.model_key) ||
      a.run.localeCompare(b.run),
  )

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Run</th>
            <th className="num">seed</th>
            <th className="num">trainable params</th>
            <th className="num">checkpoint</th>
            <th className="num">train peak</th>
            <th>Scope</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((row) => (
            <tr key={row.run} className={selected.has(row.run) ? 'selected-run' : undefined}>
              <td>
                <Swatch modelKey={row.model_key} /> {MODEL_SHORT[row.model_key]}
              </td>
              <td className="mono">
                {row.run}
                {selected.has(row.run) ? (
                  <span className="badge" style={{ marginLeft: 6 }}>selected</span>
                ) : null}
              </td>
              <td className="num">{row.seed ?? '—'}</td>
              <td className="num">{fmtInt(row.trainable_parameters)}</td>
              <td className="num">{fmtBytes(row.checkpoint_bytes)}</td>
              <td className="num">
                {row.adaptation_peak_gib ? `${fmt(row.adaptation_peak_gib, 2)} GiB` : '—'}
              </td>
              <td style={{ whiteSpace: 'normal' }}>{row.checkpoint_scope}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function BenchmarkPage() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    apiCalls
      .getBenchmarkSummary()
      .then(setSummary)
      .catch((err) => setError(errorMessage(err, 'Could not load the benchmark.')))
  }, [])

  if (error) return <Banner kind="error">{error}</Banner>
  if (!summary) {
    return (
      <div className="card">
        <div className="progress">
          <Spinner /> Loading benchmark…
        </div>
      </div>
    )
  }

  const cases = summary.evaluation_cases

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Research benchmark</h1>
        <p>
          Saved results from the HDSD study for the development-selected checkpoint
          of each model. These are not live measurements — they come from the frozen
          evaluation tables.
        </p>
      </div>

      <div className="chart-grid">
        <ChartCard
          title="Word error rate by evaluation case"
          note="Lower is better. Percentages."
          table={<CaseTable cases={cases} series={summary.wer_by_case} />}
        >
          <GroupedByCase cases={cases} series={summary.wer_by_case} />
        </ChartCard>

        <ChartCard
          title="Character error rate by evaluation case"
          note="Code points including spaces, so word-boundary errors are visible."
          table={<CaseTable cases={cases} series={summary.cer_by_case} />}
        >
          <GroupedByCase cases={cases} series={summary.cer_by_case} />
        </ChartCard>

        <ChartCard
          title="Seed spread on the primary case (mean ± SD)"
          note="Error bars are the sample standard deviation across training seeds. The pretrained baseline is a single run, so it has no SD."
          table={<SeedTable rows={summary.seed_spread} cases={cases} />}
        >
          <SeedSpread rows={summary.seed_spread} cases={cases} />
        </ChartCard>

        <ChartCard
          title="Error breakdown (substitutions / deletions / insertions)"
          table={<SdiTable rows={summary.error_breakdown} cases={cases} />}
        >
          <ErrorBreakdown rows={summary.error_breakdown} cases={cases} />
        </ChartCard>
      </div>

      <section className="card">
        <div className="card-head">
          <h2>Speaker-macro WER by evaluation case</h2>
        </div>
        <GroupedByCase cases={cases} series={summary.macro_speaker_wer_by_case} />
        <p className="card-note">
          Averaged per speaker and then across speakers, so a talkative speaker
          does not dominate the figure.
        </p>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Adaptation efficiency</h2>
        </div>
        <EfficiencyTable rows={summary.efficiency} selectedRuns={summary.selected_runs} />
        <p className="card-note">
          The LoRA adapter is about 12.7&nbsp;MB against roughly 973&nbsp;MB for the
          full fine-tune, but it still needs the pinned base model at inference
          time — the adapter is not a standalone checkpoint.
        </p>
      </section>

      <div className="disclaimer">{summary.disclaimer}</div>

      {summary.notes?.length ? (
        <ul className="muted" style={{ margin: 0, paddingLeft: '1.1rem' }}>
          {summary.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
