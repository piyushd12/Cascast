import clsx from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export function Panel({ title, right, children, className }: {
  title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={clsx('panel flex min-h-0 flex-col', className)}>
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {right}
        </header>
      )}
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: string
}) {
  return (
    <div className="panel px-4 py-3">
      <div className="label">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-2xl font-semibold tracking-tight" style={tone ? { color: tone } : undefined}>{value}</span>
      </div>
      {sub && <div className="mt-0.5 text-xs text-ink-2">{sub}</div>}
    </div>
  )
}

export function Button({ variant = 'ghost', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'ghost' | 'danger'
}) {
  return (
    <button {...props}
      className={clsx('inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary' && 'bg-accent text-white hover:brightness-110',
        variant === 'danger' && 'bg-critical/90 text-white hover:bg-critical',
        variant === 'ghost' && 'border border-line bg-panel-2 text-ink-2 hover:text-ink',
        className)} />
  )
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex rounded-md border border-line bg-panel-2 p-0.5 text-xs" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} onClick={() => onChange(o.value)}
          className={clsx('rounded px-2.5 py-1 font-medium transition',
            o.value === value ? 'bg-accent text-white' : 'text-ink-2 hover:text-ink')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

const KIND_STYLE: Record<string, string> = {
  cascade: 'bg-critical/15 text-critical',
  fault: 'bg-serious/15 text-serious',
  restored: 'bg-good/15 text-good',
  weather: 'bg-accent/15 text-accent',
}

export function Kind({ kind }: { kind: string }) {
  return <span className={clsx('rounded px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide', KIND_STYLE[kind])}>{kind}</span>
}
