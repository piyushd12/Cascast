import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import { Pause, Play, Shuffle, X, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { GridMap, MapLegend } from '../components/GridMap'
import { Button, Panel, Segmented } from '../components/ui'
import { api, type Grid, type WhatIfResult } from '../lib/api'
import { useLive } from '../lib/live'

type View = 'replay' | 'gnn' | 'mc'

export default function LabPage({ grid }: { grid: Grid }) {
  const [params, setParams] = useSearchParams()
  const [lines, setLines] = useState<number[]>(() =>
    (params.get('lines') ?? '').split(',').filter(Boolean).map(Number).filter((n) => n >= 0 && n < grid.branches.length))
  const [scale, setScale] = useState(1)
  const [stage, setStage] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [mapView, setMapView] = useState<View>('gnn')
  const { state, risk } = useLive()

  const run = useMutation({
    mutationFn: () => api.whatif(lines, scale),
    onSuccess: () => { setStage(0); setPlaying(false); setMapView('gnn') },
  })
  const res = run.data

  // Run immediately when opened with ?lines=...
  useEffect(() => { if (lines.length) run.mutate() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const animating = playing && !!res && stage < res.sample.n_stages
  useEffect(() => {
    if (!animating) return
    const t = window.setTimeout(() => setStage((s) => s + 1), 1100)
    return () => window.clearTimeout(t)
  }, [animating, stage])

  const select = (next: number[]) => {
    setLines(next)
    setParams(next.length ? { lines: next.join(',') } : {}, { replace: true })
  }
  const toggle = (l: number) => select(lines.includes(l) ? lines.filter((x) => x !== l) : [...lines, l].slice(-5))
  const randomN2 = () => {
    const alive = grid.branches.filter((b) => state?.alive[b.id] ?? true).map((b) => b.id)
    select([...alive].sort(() => Math.random() - 0.5).slice(0, 2))
  }

  const view = res && res.lines.join(',') === [...lines].sort((a, b) => a - b).join(',') ? res : null
  const replay = view && mapView === 'replay'
  const stageLoading = replay ? view.sample.stages[Math.min(stage, view.sample.stages.length - 1)]?.loading : undefined
  const stageAlive = useMemo(() => {
    if (!state) return undefined
    if (!view) return state.alive
    if (!replay) return state.alive.map((a, e) => (a && !lines.includes(e) ? 1 : 0))
    return state.alive.map((a, e) => (a && !(view.sample.stage[e] >= 0 && view.sample.stage[e] <= stage) ? 1 : 0))
  }, [view, replay, state, stage, lines])

  const mode = !view ? 'loading' : replay ? 'cascade' : 'risk'
  const riskValues = view ? (mapView === 'mc' ? view.monte_carlo.prob : view.gnn.prob) : undefined

  return (
    <div className="grid gap-4 p-4 lg:p-5 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Panel title="What-If Lab"
        right={view
          ? <Segmented value={mapView} onChange={(v) => { setMapView(v); setStage(0); setPlaying(v === 'replay') }}
              options={[{ value: 'gnn', label: 'GNN probability' }, { value: 'mc', label: 'Monte Carlo' }, { value: 'replay', label: 'Replay one cascade' }]} />
          : <span className="text-xs text-muted">Click branches to select a contingency (up to 5)</span>}>
        <GridMap grid={grid} mode={mode}
          loading={stageLoading ?? state?.loading} alive={stageAlive} risk={riskValues}
          stage={replay ? view.sample.stage : undefined} currentStage={stage}
          predicted={replay ? view.gnn.likely : undefined}
          selected={lines} onLineClick={toggle} className="h-[62vh] min-h-[420px]" />
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-2.5">
          <MapLegend mode={mode} />
          {replay && view.sample.n_stages > 0 && (
            <div className="flex items-center gap-2">
              <Button onClick={() => {
                if (animating) return setPlaying(false)
                if (stage >= view.sample.n_stages) setStage(0)
                setPlaying(true)
              }} aria-label={animating ? 'Pause replay' : 'Play replay'}>
                {animating ? <Pause size={14} /> : <Play size={14} />}
              </Button>
              <input type="range" min={0} max={view.sample.n_stages} value={stage} aria-label="Cascade stage"
                onChange={(e) => { setPlaying(false); setStage(Number(e.target.value)) }} className="w-40 accent-[#3987e5]" />
              <span className="num w-20 text-xs text-ink-2">stage {stage}/{view.sample.n_stages}</span>
            </div>
          )}
        </div>
      </Panel>

      <div className="flex flex-col gap-4">
        <Panel title="Contingency">
          <div className="flex flex-col gap-4 p-4">
            <div>
              <div className="label mb-2">Branches to trip</div>
              <div className="flex min-h-8 flex-wrap gap-1.5">
                {lines.length === 0 && <span className="text-sm text-muted">None selected — click the map</span>}
                {lines.map((l) => (
                  <button key={l} onClick={() => toggle(l)}
                    className="flex items-center gap-1 rounded-md bg-panel-2 px-2 py-1 font-mono text-xs text-ink hover:bg-line">
                    #{l} <X size={12} />
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 flex justify-between"><span className="label">Demand</span>
                <span className="num text-xs text-ink-2">{scale === 1 ? 'current' : `${scale > 1 ? '+' : ''}${Math.round((scale - 1) * 100)}%`}</span></div>
              <input type="range" min={0.8} max={1.4} step={0.01} value={scale} aria-label="Demand multiplier"
                onChange={(e) => setScale(Number(e.target.value))} className="w-full accent-[#3987e5]" />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="primary" disabled={!lines.length || run.isPending} onClick={() => run.mutate()}>
                <Zap size={14} />{run.isPending ? 'Simulating…' : 'Run'}
              </Button>
              <Button onClick={randomN2}><Shuffle size={14} />Random N-2</Button>
              {risk?.top[0] && <Button onClick={() => select([risk.top[0].line])}>Riskiest N-1</Button>}
            </div>
            {run.isError && <div className="text-sm text-critical">{String(run.error)}</div>}
          </div>
        </Panel>

        {view ? <Results res={view} stage={replay ? stage : view.sample.n_stages} /> : (
          <Panel title="How it works">
            <div className="space-y-2 p-4 text-sm leading-relaxed text-ink-2">
              <p>The selected branches are tripped on the <b className="text-ink">current live operating point</b>.</p>
              <p>Protection is <b className="text-ink">stochastic</b>: relay trip thresholds vary, and relays next to a
                failure can misoperate (“hidden failures”). The same contingency can fizzle out or black out a region,
                so the right answer is a <b className="text-ink">probability</b>.</p>
              <p>The GNN predicts every branch’s failure probability in one pass. Physics needs a Monte Carlo of
                100 full cascade simulations to estimate the same thing — compare them side by side, then replay one
                random realisation stage by stage.</p>
            </div>
          </Panel>
        )}
      </div>
    </div>
  )
}

function Results({ res, stage }: { res: WhatIfResult; stage: number }) {
  const speedup = res.monte_carlo.latency_ms / Math.max(res.gnn.latency_ms, 0.01)
  const top = useMemo(() => {
    const ids = res.gnn.prob.map((_, e) => e).filter((e) => Math.max(res.gnn.prob[e], res.monte_carlo.prob[e]) >= 0.05)
    return ids.sort((a, b) => res.monte_carlo.prob[b] + res.gnn.prob[b] - res.monte_carlo.prob[a] - res.gnn.prob[a]).slice(0, 10)
  }, [res])
  const s = res.sample
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <Card title="CascadeGNN" tone="#3987e5"
          rows={[['Expected shed', `${res.gnn.shed_pct.toFixed(2)}%`], ['Likely trips (p ≥ 50%)', res.gnn.likely.length],
            ['Time', `${res.gnn.latency_ms.toFixed(1)} ms`]]} />
        <Card title={`Monte Carlo · ${res.monte_carlo.runs} runs`} tone="#d95926"
          rows={[['Expected shed', `${res.monte_carlo.expected_shed_pct.toFixed(2)}%`], ['Likely trips (p ≥ 50%)', res.monte_carlo.likely.length],
            ['Time', `${res.monte_carlo.latency_ms.toFixed(0)} ms`]]} />
      </div>

      <Panel title="Failure probability by branch" right={<span className="text-xs text-muted">GNN vs Monte Carlo</span>}>
        <div className="max-h-[260px] overflow-y-auto px-4 py-3">
          {top.length === 0 && <div className="py-3 text-center text-sm text-ink-2">No branch has a ≥5% chance of failing.</div>}
          {top.map((e) => (
            <div key={e} className="grid grid-cols-[48px_1fr] items-center gap-2 py-1">
              <span className="font-mono text-xs text-ink">#{e}</span>
              <div className="space-y-0.5">
                <Bar v={res.gnn.prob[e]} color="#3987e5" />
                <Bar v={res.monte_carlo.prob[e]} color="#d95926" />
              </div>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-line px-4 py-2.5 text-xs text-ink-2">
          <span>Probability error <b className="num text-ink">±{(res.agreement.prob_mae * 100).toFixed(1)} pp</b></span>
          <span>Expected-shed error <b className="num text-ink">{res.agreement.shed_error_pct.toFixed(2)} pp</b></span>
          <span>GNN <b className="text-ink">{speedup.toFixed(0)}×</b> faster</span>
        </div>
      </Panel>

      <Panel title="One random realisation" right={<span className="text-xs text-muted">P(any cascade) {(res.monte_carlo.cascade_prob * 100).toFixed(0)}% · p90 shed {res.monte_carlo.shed_p90_pct.toFixed(1)}%</span>}>
        {s.failed.length === 0
          ? <div className="p-4 text-sm text-ink-2">In this run the grid absorbed the contingency — nothing else tripped.</div>
          : (
            <ol className="space-y-1.5 p-4 text-sm">
              {s.stages.map((st, k) => st.tripped.length > 0 && (
                <li key={k} className={clsx('flex gap-3', k > stage && 'opacity-40')}>
                  <span className="num w-14 shrink-0 text-xs text-muted">{k === 0 ? 'Initial' : `Stage ${k}`}</span>
                  <span className="flex flex-wrap gap-1">
                    {st.tripped.map((l) => (
                      <span key={l} title={s.hidden.includes(l) ? 'relay misoperation (hidden failure)' : undefined}
                        className={clsx('rounded px-1.5 font-mono text-xs',
                          s.hidden.includes(l) ? 'bg-warn/15 text-warn' : 'bg-critical/15 text-critical')}>
                        #{l}{s.hidden.includes(l) && ' ⚡'}
                      </span>
                    ))}
                  </span>
                </li>
              ))}
            </ol>
          )}
        <div className="border-t border-line px-4 py-2 text-xs text-muted">
          This run shed {s.shed_pct.toFixed(2)}% of demand · ⚡ = relay misoperation · switch the map to “Replay” to watch it
        </div>
      </Panel>
    </>
  )
}

function Bar({ v, color }: { v: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 rounded-full bg-line">
        <div className="h-1.5 rounded-full" style={{ width: `${Math.max(1, v * 100)}%`, background: color }} />
      </div>
      <span className="num w-9 text-right text-[11px] text-ink-2">{Math.round(v * 100)}%</span>
    </div>
  )
}

function Card({ title, tone, rows }: { title: string; tone: string; rows: [string, string | number][] }) {
  return (
    <div className="panel p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-ink">
        <span className="h-2 w-2 rounded-full" style={{ background: tone }} />{title}
      </div>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2 py-0.5 text-sm"><span className="text-ink-2">{k}</span><span className="num">{v}</span></div>
      ))}
    </div>
  )
}
