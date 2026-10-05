import clsx from 'clsx'
import { Send, Sparkles, Wrench, X } from 'lucide-react'
import { Fragment, useEffect, useRef, useState, type ReactNode } from 'react'

interface Turn { role: 'user' | 'assistant'; content: string; tools?: { name: string; ok?: boolean }[]; error?: string }

const SUGGESTIONS = [
  'What is the riskiest contingency right now, and why?',
  'If lines 7 and 8 both fail, what happens?',
  'Would 5% demand response make the grid N-1 secure?',
]

const TOOL_LABEL: Record<string, string> = {
  get_grid_status: 'Reading grid status',
  simulate_contingency: 'Simulating contingency',
  inspect_line: 'Inspecting line',
  evaluate_load_relief: 'Evaluating demand response',
}

export function Copilot({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns])

  const patchLast = (fn: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), fn(ts[ts.length - 1])])

  const ask = async (text: string) => {
    if (!text.trim() || busy) return
    const history: Turn[] = [...turns, { role: 'user', content: text.trim() }]
    setTurns([...history, { role: 'assistant', content: '', tools: [] }])
    setInput('')
    setBusy(true)
    try {
      const res = await fetch('/api/copilot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: history.filter((t) => t.content).map(({ role, content }) => ({ role, content })),
        }),
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += value
        const frames = buf.split('\n\n')
        buf = frames.pop() ?? ''
        for (const f of frames) {
          if (!f.startsWith('data: ')) continue
          const ev = JSON.parse(f.slice(6))
          if (ev.type === 'text') patchLast((t) => ({ ...t, content: t.content + ev.text }))
          else if (ev.type === 'tool') patchLast((t) => ({ ...t, tools: [...(t.tools ?? []), { name: ev.name }] }))
          else if (ev.type === 'tool_result') patchLast((t) => {
            const tools = [...(t.tools ?? [])]
            const i = tools.findLastIndex((x) => x.name === ev.name && x.ok === undefined)
            if (i >= 0) tools[i] = { ...tools[i], ok: ev.ok }
            return { ...t, tools }
          })
          else if (ev.type === 'error') patchLast((t) => ({ ...t, error: ev.message }))
        }
      }
    } catch (e) {
      patchLast((t) => ({ ...t, error: `Request failed: ${String(e)}` }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside aria-hidden={!open}
      className={clsx('fixed inset-y-0 right-0 z-30 flex w-full max-w-[440px] flex-col border-l border-line bg-panel shadow-2xl transition-transform duration-300',
        open ? 'translate-x-0' : 'translate-x-full')}>
      <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-4">
        <Sparkles size={16} className="text-accent" />
        <div className="font-semibold">Operator Copilot</div>
        <span className="text-xs text-muted">Claude + live grid tools</span>
        <button onClick={onClose} className="ml-auto rounded p-1 text-ink-2 hover:text-ink" aria-label="Close copilot"><X size={18} /></button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {turns.length === 0 && (
          <div className="space-y-3">
            <p className="text-sm text-ink-2">
              Ask about the live grid. The copilot calls the same GNN and physics simulator you see on the dashboard,
              so every answer is grounded in real numbers.
            </p>
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => ask(s)}
                className="block w-full rounded-lg border border-line bg-panel-2 px-3 py-2 text-left text-sm text-ink-2 hover:text-ink">
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={clsx('text-sm', t.role === 'user' && 'ml-10 rounded-lg bg-accent/15 px-3 py-2 text-ink')}>
            {t.tools && t.tools.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {t.tools.map((tool, j) => (
                  <span key={j} className={clsx('flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]',
                    tool.ok === false ? 'border-critical/40 text-critical' : 'border-line text-ink-2')}>
                    <Wrench size={11} className={tool.ok === undefined ? 'pulse' : undefined} />
                    {TOOL_LABEL[tool.name] ?? tool.name}
                  </span>
                ))}
              </div>
            )}
            {t.role === 'assistant' ? <Markdown text={t.content} /> : t.content}
            {t.role === 'assistant' && busy && i === turns.length - 1 && !t.content && !t.error &&
              <span className="pulse text-muted">Thinking…</span>}
            {t.error && <div className="mt-2 rounded-md border border-critical/40 bg-critical/10 px-3 py-2 text-xs text-critical">{t.error}</div>}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form className="flex gap-2 border-t border-line p-3" onSubmit={(e) => { e.preventDefault(); ask(input) }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask the copilot…" aria-label="Message"
          className="min-w-0 flex-1 rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-ink outline-none placeholder:text-muted focus:border-accent" />
        <button type="submit" disabled={busy || !input.trim()} aria-label="Send"
          className="rounded-md bg-accent px-3 text-white disabled:opacity-40"><Send size={16} /></button>
      </form>
    </aside>
  )
}

/** Tiny Markdown subset: paragraphs, bullet lists, **bold**, `code`. */
function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = []
  let list: string[] = []
  const flush = () => {
    if (list.length) blocks.push(<ul key={blocks.length} className="my-1.5 list-disc space-y-1 pl-5">{list.map((l, i) => <li key={i}>{inline(l)}</li>)}</ul>)
    list = []
  }
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/)
    if (m) { list.push(m[1]); continue }
    flush()
    if (line.trim()) {
      const h = line.match(/^#{1,4}\s+(.*)/)
      blocks.push(<p key={blocks.length} className={clsx('my-1.5 leading-relaxed', h && 'font-semibold text-ink')}>{inline(h ? h[1] : line)}</p>)
    }
  }
  flush()
  return <div className="text-ink-2">{blocks}</div>
}

function inline(s: string): ReactNode {
  return s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
    part.startsWith('**') ? <b key={i} className="text-ink">{part.slice(2, -2)}</b>
      : part.startsWith('`') ? <code key={i} className="rounded bg-panel-2 px-1 font-mono text-xs">{part.slice(1, -1)}</code>
        : <Fragment key={i}>{part}</Fragment>)
}
