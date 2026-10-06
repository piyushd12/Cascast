import clsx from 'clsx'
import { CloudLightning, FlaskConical, RotateCcw, Sun, ThermometerSun, Zap } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { DemandChart, RiskChart } from '../components/Charts'
import { GridMap, MapLegend, type MapMode } from '../components/GridMap'
import { Button, Kind, Panel, Segmented, Stat } from '../components/ui'
import { api, type Grid } from '../lib/api'
import { fmtMinute, fmtMW, STATUS } from '../lib/format'
import { useLive } from '../lib/live'

export default function LivePage({ grid }: { grid: Grid }) {
  const { state, risk, events, history, live } = useLive()
  const [mode, setMode] = useState<MapMode>('loading')
  const [sel, setSel] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const navigate = useNavigate()

  const shedBuses = useMemo(() => {
    const s = new Set<number>()
    state?.load.forEach((l, i) => { if (l > 0 && state.served[i] < l * 0.99) s.add(i) })
    return s
  }, [state])

  const send = async (body: Parameters<typeof api.command>[0]) => {
    setBusy(true)
    try { await api.command(body) } finally { setBusy(false) }
  }

  if (!state) return <div className="p-6 text-muted">Waiting for the grid feed…</div>
  const status = risk ? STATUS[risk.status] : null
  const shed = state.demand_mw - state.served_mw
  const outLines = state.alive.filter((a) => !a).length
  const acc = risk?.live_accuracy

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-5">
      {!live && (
        <div className="panel border-warn/40 px-4 py-2.5 text-sm text-ink-2">
          Live simulation isn’t running, so this is the nominal operating point. Start the simulator and risk engine
          (<code className="font-mono">docker compose up</code>) to stream real-time data.
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Grid status" value={status?.label ?? '—'} tone={status?.color}
          sub={risk ? `${risk.n_critical} critical of ${risk.n_screened} contingencies` : 'screening…'} />
        <Stat label="Demand" value={fmtMW(state.demand_mw)} sub={`load factor ${state.load_factor.toFixed(2)}`} />
        <Stat label="Load shed now" value={fmtMW(shed)} tone={shed > 1 ? '#e5484d' : undefined}
          sub={`${outLines} branch${outLines === 1 ? '' : 'es'} out · ${state.n_islands} island${state.n_islands > 1 ? 's' : ''}`} />
        <Stat label="Worst N-1 risk" value={risk ? `${risk.worst_shed_pct.toFixed(2)}%` : '—'}
          tone={risk && risk.worst_shed_pct > 1 ? '#ec835a' : undefined} sub="expected demand lost, single failure" />
        <Stat label="GNN screen latency" value={risk ? `${risk.latency_ms.toFixed(1)} ms` : '—'}
          sub={risk ? `${risk.n_screened} cascades / tick` : undefined} />
        <Stat label="Live audit vs physics" value={acc && acc.checked ? `±${(acc.prob_mae * 100).toFixed(1)} pp` : '—'}
          sub={acc && acc.checked ? `probability error · ${acc.checked} Monte Carlo checks` : `vs ${acc?.mc_runs ?? 40}-run Monte Carlo`} />
      </div>

      <div className="grid min-h-0 gap-4 xl:h-[max(560px,calc(100vh-232px))] xl:grid-cols-[minmax(0,1fr)_380px] xl:grid-rows-[minmax(0,1fr)]">
        <Panel title="Network"
          right={<Segmented value={mode} onChange={setMode}
            options={[{ value: 'loading', label: 'Line loading' }, { value: 'risk', label: 'N-1 vulnerability' }]} />}>
          <GridMap grid={grid} mode={mode} loading={state.loading} alive={state.alive}
            risk={risk?.vulnerability} shedBuses={shedBuses}
            selected={sel !== null ? [sel] : []} onLineClick={(l) => setSel(l === sel ? null : l)}
            className="min-h-[380px] flex-1" />
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-2.5">
            <MapLegend mode={mode} />
            {sel !== null ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-ink-2">Branch {sel} · {(state.loading[sel] * 100).toFixed(0)}% loaded</span>
                <Button onClick={() => navigate(`/lab?lines=${sel}`)}><FlaskConical size={14} />Analyze</Button>
                <Button variant="danger" disabled={busy || !live || !state.alive[sel]}
                  onClick={() => send({ cmd: 'trip', lines: [sel] })}><Zap size={14} />Trip now</Button>
              </div>
            ) : <span className="text-xs text-muted">Click a branch to inspect or trip it · scroll to zoom</span>}
          </div>
        </Panel>

        <div className="flex min-h-0 flex-col gap-4">
          <Panel title="Riskiest contingencies" right={<span className="text-xs text-muted">GNN · N-1</span>}>
            <div className="max-h-[270px] overflow-y-auto">
              {risk && risk.top.length === 0 && (
                <div className="px-4 py-6 text-center text-sm text-ink-2">No single failure triggers a cascade. ✓</div>
              )}
              {risk?.top.map((c) => (
                <button key={c.line} onClick={() => navigate(`/lab?lines=${c.line}`)}
                  className="flex w-full items-center gap-3 border-b border-line/60 px-4 py-2 text-left text-sm hover:bg-panel-2">
                  <span className="w-16 font-mono text-ink">#{c.line}</span>
                  <span className="flex-1 text-xs text-ink-2">
                    {c.n_fail > 0 ? `${c.n_fail} likely trip${c.n_fail > 1 ? 's' : ''}` : `max P(trip) ${(c.max_prob * 100).toFixed(0)}%`}
                  </span>
                  <span className={clsx('num w-16 text-right font-medium', c.shed_pct > 2 ? 'text-critical' : 'text-serious')}>
                    {c.shed_pct.toFixed(2)}%
                  </span>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Operator controls">
            <div className="flex flex-wrap gap-2 p-3">
              <Button disabled={busy || !live} onClick={() => send({ cmd: 'weather', value: 'clear' })}><Sun size={14} />Clear</Button>
              <Button disabled={busy || !live} onClick={() => send({ cmd: 'weather', value: 'heatwave' })}><ThermometerSun size={14} />Heatwave</Button>
              <Button disabled={busy || !live} onClick={() => send({ cmd: 'weather', value: 'storm' })}><CloudLightning size={14} />Storm</Button>
              <Button disabled={busy || !live} onClick={() => send({ cmd: 'restore_all' })}><RotateCcw size={14} />Restore all</Button>
            </div>
          </Panel>

          <Panel title="Event log" className="min-h-[200px] flex-1">
            <ul className="max-h-[260px] min-h-0 flex-1 overflow-y-auto text-sm xl:max-h-none">
              {events.length === 0 && <li className="px-4 py-6 text-center text-muted">No events yet</li>}
              {events.map((e) => (
                <li key={`${e.id}-${e.minute}`} className="border-b border-line/60 px-4 py-2">
                  <div className="flex items-center gap-2">
                    <Kind kind={e.kind} /><span className="num text-[11px] text-muted">{fmtMinute(e.minute)}</span>
                    {!!e.shed_pct && <span className="num ml-auto text-xs text-critical">−{e.shed_pct}%</span>}
                  </div>
                  <div className="mt-0.5 text-ink-2">{e.title}</div>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Demand vs served" right={<ChartKey items={[['#3987e5', 'Served'], ['#a9b6c3', 'Demand']]} />}>
          <div className="h-52 p-2"><DemandChart data={history} /></div>
        </Panel>
        <Panel title="Worst single-failure expected load shed" right={<span className="text-xs text-muted">% of demand</span>}>
          <div className="h-52 p-2"><RiskChart data={history} /></div>
        </Panel>
      </div>
    </div>
  )
}

function ChartKey({ items }: { items: [string, string][] }) {
  return (
    <div className="flex gap-3 text-xs text-ink-2">
      {items.map(([c, l]) => <span key={l} className="flex items-center gap-1.5"><span className="h-0.5 w-4" style={{ background: c }} />{l}</span>)}
    </div>
  )
}
