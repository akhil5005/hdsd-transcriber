import * as apiCalls from '../api/endpoints.js'
import {
  Banner,
  CopyButton,
  MODEL_COLOR,
  MODEL_ORDER,
  MODEL_SHORT,
  Metric,
  fmt,
  fmtInt,
  Swatch,
} from './common.jsx'

const KIND_CLASS = {
  equal: 'equal',
  substitution: 'substitution',
  deletion: 'deletion',
  insertion: 'insertion',
}

/**
 * Word-level diff against the reference.
 *
 * A deletion has no hypothesis token, so the reference word is shown struck
 * through — otherwise a dropped word would simply be invisible, which is the
 * one error type clinicians most want to see.
 */
function Diff({ tokens }) {
  if (!tokens?.length) return <span className="muted">No diff available.</span>

  return (
    <p className="diff">
      {tokens.map((token, index) => {
        const shown = token.kind === 'deletion' ? token.reference : token.hypothesis
        if (!shown) return null
        const title =
          token.kind === 'substitution'
            ? `expected: ${token.reference}`
            : token.kind === 'deletion'
              ? 'missing from the transcript'
              : token.kind === 'insertion'
                ? 'not in the reference'
                : undefined
        return (
          <span key={index} className={`tok ${KIND_CLASS[token.kind]}`} title={title}>
            {shown}{' '}
          </span>
        )
      })}
    </p>
  )
}

function DiffLegend() {
  return (
    <div className="diff-legend">
      <span>
        <span className="key substitution">substitution</span> wrong word
      </span>
      <span>
        <span className="key deletion">deletion</span> word missing
      </span>
      <span>
        <span className="key insertion">insertion</span> extra word
      </span>
    </div>
  )
}

function ModelColumn({ row, scores, isBest, hasReference }) {
  const diff = scores?.diff

  return (
    <article className="model-column" style={{ '--series-color': MODEL_COLOR[row.model_key] }}>
      <header>
        <Swatch modelKey={row.model_key} />
        <span className="title">{MODEL_SHORT[row.model_key] || row.model_key}</span>
        {isBest ? <span className="badge best">lowest WER</span> : null}
        <span className="badge" style={{ marginLeft: 'auto' }}>
          {row.device || '—'}
        </span>
      </header>

      <div className="body stack">
        <div className="metric-row">
          {hasReference ? (
            <>
              <Metric label="WER" value={fmt(scores?.wer)} unit="%" highlight={isBest} />
              <Metric label="CER" value={fmt(scores?.cer)} unit="%" />
              <Metric
                label="S / D / I"
                value={`${fmtInt(scores?.substitutions)}/${fmtInt(
                  scores?.deletions,
                )}/${fmtInt(scores?.insertions)}`}
              />
            </>
          ) : null}
          <Metric label="Inference" value={fmtInt(row.inference_ms)} unit="ms" />
          <Metric
            label="RTF"
            value={fmt(row.rtf)}
            title="Real-time factor: inference time ÷ audio duration. Lower is faster."
          />
        </div>

        <div>
          <div className="row-between" style={{ marginBottom: '0.35rem' }}>
            <h3>Transcript</h3>
            <CopyButton text={row.text_raw} />
          </div>
          <p className="devanagari" style={{ margin: 0 }}>
            {row.text_raw || <span className="muted">(empty)</span>}
          </p>
        </div>

        {hasReference && diff ? (
          <div>
            <h3 style={{ marginBottom: '0.35rem' }}>Against the reference</h3>
            <Diff tokens={diff} />
          </div>
        ) : null}

        {row.chunks && row.chunks.length > 1 ? (
          <details>
            <summary className="muted">{row.chunks.length} windows</summary>
            <div className="table-wrap" style={{ marginTop: '0.5rem' }}>
              <table>
                <thead>
                  <tr>
                    <th className="num">from</th>
                    <th className="num">to</th>
                    <th>text</th>
                  </tr>
                </thead>
                <tbody>
                  {row.chunks.map((chunk) => (
                    <tr key={chunk.index}>
                      <td className="num">{fmt(chunk.start_sec, 1)}s</td>
                      <td className="num">{fmt(chunk.end_sec, 1)}s</td>
                      <td className="devanagari" style={{ whiteSpace: 'normal' }}>
                        {chunk.text}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ) : null}

        <p className="muted" style={{ margin: 0 }}>
          decode config <code>{(row.gen_config_hash || '').slice(0, 12) || '—'}</code>
        </p>
      </div>
    </article>
  )
}

export default function JobResults({ job }) {
  const comparison = job.comparison || {}
  const hasReference = Boolean(comparison.has_reference)
  const scores = comparison.scores || {}
  const best = comparison.lowest_wer_model

  const rows = [...(job.transcriptions || [])].sort(
    (a, b) => MODEL_ORDER.indexOf(a.model_key) - MODEL_ORDER.indexOf(b.model_key),
  )

  return (
    <div className="stack">
      <section className="card">
        <div className="card-head">
          <h2>
            {rows.length > 1 ? 'Comparison' : 'Result'}
            {job.media?.original_filename ? (
              <span className="muted"> · {job.media.original_filename}</span>
            ) : null}
          </h2>
          <div className="button-row">
            {['txt', 'json', 'csv'].map((format) => (
              <button
                key={format}
                type="button"
                className="small"
                onClick={() => apiCalls.downloadExport(job.id, format)}
              >
                Export {format.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="metric-row">
          <Metric
            label="Audio"
            value={fmt(job.media?.duration_sec, 1)}
            unit="s"
          />
          <Metric label="Models" value={rows.length} />
          <Metric
            label="Reference"
            value={
              job.reference_source === 'dataset'
                ? `dataset · ${job.reference_prompt_id}`
                : hasReference
                  ? 'typed'
                  : 'none'
            }
          />
          {comparison.unanimous !== undefined ? (
            <Metric
              label="Outputs"
              value={comparison.unanimous ? 'identical' : 'differ'}
            />
          ) : null}
        </div>

        {hasReference ? <DiffLegend /> : null}
      </section>

      {job.reference_text ? (
        <section className="card">
          <div className="card-head">
            <h3>Ground truth</h3>
            {job.reference_source === 'dataset' ? (
              <span className="gt-chip">from dataset · {job.reference_prompt_id}</span>
            ) : (
              <span className="badge">typed by you</span>
            )}
          </div>
          <p className="devanagari gt-sentence" lang="hi">
            {job.reference_text}
          </p>
          <p className="card-note">
            Every score below is measured against this sentence.
          </p>
        </section>
      ) : null}

      {job.reference_source === 'dataset' && job.reference_verified === false ? (
        <Banner kind="warn">
          Ground truth for {job.reference_prompt_id} came from the corpus's own
          romanised word labels and has not been checked by a person yet. The
          transcripts below are real; treat the WER and CER as provisional until
          the sentence is verified.
        </Banner>
      ) : null}

      <div className="compare-grid">
        {rows.map((row) => (
          <ModelColumn
            key={row.model_key}
            row={row}
            scores={scores[row.model_key]}
            isBest={hasReference && row.model_key === best && rows.length > 1}
            hasReference={hasReference}
          />
        ))}
      </div>
    </div>
  )
}
