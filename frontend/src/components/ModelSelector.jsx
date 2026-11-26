import { MODEL_COLOR, Swatch } from './common.jsx'

/**
 * The four choices from PROMPT.md section 9. Descriptions name the selected
 * seed, because which checkpoint ran is part of what the result means.
 */
const CHOICES = [
  {
    key: 'pretrained',
    title: 'Pretrained',
    desc: 'Whisper-small, OpenAI weights at the pinned revision.',
  },
  {
    key: 'full_ft',
    title: 'Full fine-tune',
    desc: 'All 240.6M parameters adapted. lr 5e-6, seed 123.',
  },
  {
    key: 'lora',
    title: 'LoRA adapter',
    desc: '1.77M trainable. lr 5e-5, seed 2026, r=16, α=32, q/v.',
  },
  {
    key: 'all',
    title: 'Compare all 3',
    desc: 'Runs every model on the same audio, side by side.',
  },
]

export default function ModelSelector({ value, onChange, disabled = false, loaded = [] }) {
  return (
    <div className="model-grid" role="radiogroup" aria-label="Model">
      {CHOICES.map((choice) => {
        const selected = value === choice.key
        const isLoaded =
          choice.key === 'all'
            ? loaded.length === 3
            : loaded.includes(choice.key)

        return (
          <button
            type="button"
            key={choice.key}
            role="radio"
            aria-checked={selected}
            className={`model-option${selected ? ' selected' : ''}`}
            disabled={disabled}
            onClick={() => onChange(choice.key)}
          >
            <span className="title">
              {choice.key === 'all' ? (
                <span aria-hidden="true" style={{ display: 'inline-flex', gap: 2 }}>
                  <Swatch color={MODEL_COLOR.pretrained} />
                  <Swatch color={MODEL_COLOR.full_ft} />
                  <Swatch color={MODEL_COLOR.lora} />
                </span>
              ) : (
                <Swatch modelKey={choice.key} />
              )}
              {choice.title}
            </span>
            <span className="desc">{choice.desc}</span>
            <span className="desc" style={{ color: 'var(--text-muted)' }}>
              {isLoaded ? 'weights resident' : 'loads on first use'}
            </span>
          </button>
        )
      })}
    </div>
  )
}
