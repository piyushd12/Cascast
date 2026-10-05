import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import { Pause, Play, Shuffle, X, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { GridMap, MapLegend } from '../components/GridMap'
import { Button, Panel } from '../components/ui'
import { api, type Grid, type WhatIfResult } from '../lib/api'
import { useLive } from '../lib/live'

export default function LabPage({ grid }: { grid: Grid }) {
  const [params, setParams] = useSearchParams()
  const [lines, setLines] = useState<number[]>(() =>
    (params.get('lines') ?? '').split(',').filter(Boolean).map(Number).filter((n) => n >= 0 && n < grid.branches.length))
  const [scale, setScale] = useState(1)
  const [stage, setStage] = useState(0)
  const [playing, setPlaying] = useState(false)
  const { state, risk } = useLive()

  const run = useMutation({ mutationFn: () => api.whatif(lines, scale), onSuccess: () => { setStage(0); setPlaying(true) } })
  const res = run.data

  // Run immediately when opened with ?lines=...
  useEffect(() => { if (lines.length) run.mutate() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const animating = playing && !!res && stage < res.physics.n_stages
  useEffect(() => {
    if (!animating) return
    const t = window.setTimeout(() => setStage((s) => s + 1), 1100)
    return () => window.clearTimeout(t)
  }, [animating, stage])

  const toggle = (l: number) => {
    const next = lines.includes(l) ? lines.filter((x) => x !== l) : [...lines, l].slice(-5)
    setLines(next)
    setParams(next.length ? { lines: next.join(',') } : {}, { replace: true })
  }

  const randomN2 = () => {
    const alive = grid.branches.filter((b) => state?.alive[b.id] ?? true).map((b) => b.id)
    const pick = [...alive].sort(() => Math.random() - 0.5).slice(0, 2)
    setLines(pick)
    setParams({ lines: pick.join(',') }, { replace: true })
  }

  const view = res && res.lines.join(',') === [...lines].sort((a, b) => a - b).join(',') ? res : null
  const stageLoading = view ? view.physics.stages[Math.min(stage, view.physics.stages.length - 1)]?.loading : undefined
  const stageAlive = useMemo(() => {
    if (!view || !state) return state?.alive
    return state.alive.map((a, e) => (a && !(view.physics.stage[e] >= 0 && view.physics.stage[e] <= stage) ? 1 : 0))
  }, [view, state, stage])

  return (
    <div className="grid gap-4 p-4 lg:p-5 xl:grid-cols-[minmax(0,1fr)_400px]">
      <Panel title="What-If Lab"
        right={<span className="text-xs text-muted">Click branches to select a contingency (up to 5)</span>}>
        <GridMap grid={grid} mode={view ? 'cascade' : 'loading'}
          loading={stageLoading ?? state?.loading} alive={stageAlive}
          stage={view?.physics.stage} currentStage={stage} predicted={view?.gnn.predicted}
          selected={lines} onLineClick={toggle} className="h-[62vh] min-h-[420px]" />
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-2.5">
          <MapLegend mode={view ? 'cascade' : 'loading'} />
          {view && view.physics.n_stages > 0 && (
            <div className="flex items-center gap-2">
              <Button onClick={() => {
                if (animating) return setPlaying(false)
                if (stage >= view.physics.n_stages) setStage(0)
                setPlaying(true)
              }} aria-label={animating ? 'Pause replay' : 'Play replay'}>
                {animating ? <Pause size={14} /> : <Play size={14} />}
              </Button>
              <input type="range" min={0} max={view.physics.n_stages} value={stage} aria-label="Cascade stage"
                onChange={(e) => { setPlaying(false); setStage(Number(e.target.value)) }} className="w-40 accent-[#3987e5]" />
              <span className="num w-20 text-xs text-ink-2">stage {stage}/{view.physics.n_stages}</span>
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
                <Zap size={14} />{run.isPending ? 'Simulating…' : 'Run cascade'}
              </Button>
              <Button onClick={randomN2}><Shuffle size={14} />Random N-2</Button>
              {risk?.top[0] && (
                <Button onClick={() => { setLines([risk.top[0].line]); setParams({ lines: String(risk.top[0].line) }, { replace: true }) }}>
                  Riskiest N-1
                </Button>
              )}
            </div>
            {run.isError && <div className="text-sm text-critical">{String(run.error)}</div>}
          </div>
        </Panel>

        {view && <Results res={view} stage={stage} />}
        {!view && (
          <Panel title="How it works">
            <p className="p-4 text-sm leading-relaxed text-ink-2">
              The selected branches are tripped on the <b className="text-ink">current live operating point</b>. The
              graph neural network predicts which other branches will overload and trip, and how much demand will be
              lost — in about a millisecond. The physics simulator then plays out the real cascade stage by stage so
              you can compare the two.
            </p>
          </Panel>
        )}
      </div>
    </div>
  )
}

function Results({ res, stage }: { res: WhatIfResult; stage: number }) {
  const speedup = res.physics.latency_ms / Math.max(res.gnn.latency_ms, 0.01)
  const st = res.physics.stages[Math.min(stage, res.physics.stages.length - 1)]
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <Card title="GNN prediction" tone="#fab219"
          rows={[['Cascading trips', res.gnn.predicted.length], ['Load shed', `${res.gnn.shed_pct.toFixed(2)}%`],
            ['Latency', `${res.gnn.latency_ms.toFixed(2)} ms`]]} />
        <Card title="Physics ground truth" tone="#e5484d"
          rows={[['Cascading trips', res.physics.failed.length], ['Load shed', `${res.physics.shed_pct.toFixed(2)}%`],
            ['Latency', `${res.physics.latency_ms.toFixed(1)} ms`]]} />
      </div>
      <Panel title="Agreement">
        <div className="grid grid-cols-4 gap-2 p-4 text-center">
          <Metric k="Jaccard" v={res.agreement.jaccard.toFixed(2)} />
          <Metric k="Correct" v={res.agreement.true_positive} />
          <Metric k="Missed" v={res.agreement.false_negative} bad={res.agreement.false_negative > 0} />
          <Metric k="False alarms" v={res.agreement.false_positive} bad={res.agreement.false_positive > 0} />
        </div>
        <div className="border-t border-line px-4 py-2.5 text-xs text-ink-2">
          GNN answered <b className="text-ink">{speedup.toFixed(0)}×</b> faster than the simulator
          {res.physics.n_stages > 0 && <> · stage {stage}: {st?.tripped.length ?? 0} branch(es) tripped, {Math.round(st?.served_mw ?? 0)} MW served</>}
        </div>
      </Panel>
      {res.physics.failed.length > 0 && (
        <Panel title="Cascade sequence">
          <ol className="space-y-1.5 p-4 text-sm">
            {res.physics.stages.map((s, k) => s.tripped.length > 0 && (
              <li key={k} className={clsx('flex gap-3', k > stage && 'opacity-40')}>
                <span className="num w-14 shrink-0 text-xs text-muted">{k === 0 ? 'Initial' : `Stage ${k}`}</span>
                <span className="flex flex-wrap gap-1">
                  {s.tripped.map((l) => (
                    <span key={l} className={clsx('rounded px-1.5 font-mono text-xs',
                      res.gnn.predicted.includes(l) || k === 0 ? 'bg-critical/15 text-critical' : 'bg-line text-ink-2')}>#{l}</span>
                  ))}
                </span>
              </li>
            ))}
          </ol>
          <div className="border-t border-line px-4 py-2 text-xs text-muted">Red = predicted by the GNN (or initial trip); grey = missed.</div>
        </Panel>
      )}
    </>
  )
}

function Card({ title, tone, rows }: { title: string; tone: string; rows: [string, string | number][] }) {
  return (
    <div className="panel p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-ink">
        <span className="h-2 w-2 rounded-full" style={{ background: tone }} />{title}
      </div>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between py-0.5 text-sm"><span className="text-ink-2">{k}</span><span className="num">{v}</span></div>
      ))}
    </div>
  )
}

function Metric({ k, v, bad }: { k: string; v: string | number; bad?: boolean }) {
  return (
    <div>
      <div className={clsx('num text-lg font-semibold', bad ? 'text-serious' : 'text-ink')}>{v}</div>
      <div className="text-[11px] text-muted">{k}</div>
    </div>
  )
}
