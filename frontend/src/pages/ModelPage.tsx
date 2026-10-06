import { useQuery } from '@tanstack/react-query'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis } from 'recharts'
import { Panel, Stat } from '../components/ui'
import { api } from '../lib/api'

const TIP = { contentStyle: { background: '#16202b', border: '1px solid #223040', borderRadius: 8, fontSize: 12 } }
const AX = { stroke: '#6b7a8a', fontSize: 11, tickLine: false }

export default function ModelPage() {
  const q = useQuery({ queryKey: ['model'], queryFn: api.model })
  if (q.isLoading) return <div className="p-6 text-muted">Loading…</div>
  if (!q.data) return <div className="p-6 text-muted">Model metrics are not available yet. Run <code className="font-mono">make model</code>.</div>
  const m = q.data
  const b = m.benchmark
  const curve = m.monte_carlo.curve.map((c) => ({ ...c, mae: c.prob_mae * 100 }))

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Speed-up vs Monte Carlo" value={`${b.speedup_vs_mc100.toFixed(0)}×`}
          sub={`${(b.mc100_ms / 1000).toFixed(1)} s → ${b.gnn_ms.toFixed(0)} ms for ${b.contingencies} contingencies`} />
        <Stat label="Matches Monte Carlo with" value={`${b.equivalent_mc_runs.toFixed(0)} runs`}
          sub={`that budget takes ${(b.equivalent_mc_ms / 1000).toFixed(1)} s · ${b.speedup_vs_equivalent.toFixed(0)}× slower`} />
        <Stat label="Probability error" value={`±${(m.monte_carlo.prob_mae * 100).toFixed(1)} pp`}
          sub={`vs ${m.monte_carlo.reference_runs}-run reference · r = ${m.monte_carlo.prob_corr.toFixed(3)}`} />
        <Stat label="Branch ranking PR-AUC" value={m.branch.pr_auc.toFixed(3)}
          sub={`ROC-AUC ${m.branch.roc_auc.toFixed(3)} · base rate ${(m.branch.positive_rate * 100).toFixed(1)}%`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Accuracy vs compute" right={<span className="text-xs text-muted">full N-1 screen</span>}>
          <div className="h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={curve} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#1b2633" vertical={false} />
                <XAxis dataKey="n1_seconds" type="number" scale="log" domain={[0.1, 100]} ticks={[0.1, 0.3, 1, 3, 10, 30, 100]} {...AX}
                  tickFormatter={(v) => `${v < 1 ? v.toFixed(2) : v.toFixed(0)} s`} />
                <YAxis {...AX} width={44} tickFormatter={(v) => `${v.toFixed(0)}`} unit=" pp" />
                <Tooltip {...TIP} formatter={(v, n) => [`${Number(v).toFixed(2)} pp`, n === 'mae' ? 'Monte Carlo error' : 'GNN error']}
                  labelFormatter={(l) => `${Number(l).toFixed(2)} s`} />
                <Line dataKey="mae" stroke="#d95926" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                <Scatter data={[{ n1_seconds: b.gnn_ms / 1000, gnn: m.monte_carlo.prob_mae * 100 }]} dataKey="gnn" fill="#3987e5" shape="star" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="border-t border-line px-4 py-2.5 text-xs text-ink-2">
            Orange: Monte Carlo with 1–128 runs per contingency. Blue star: CascadeGNN — as accurate as
            ~{b.equivalent_mc_runs.toFixed(0)} runs, in {b.gnn_ms.toFixed(0)} ms.
          </div>
        </Panel>

        <Panel title="Validation PR-AUC by epoch">
          <div className="h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={m.history} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#1b2633" vertical={false} />
                <XAxis dataKey="epoch" {...AX} />
                <YAxis {...AX} width={44} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip {...TIP} formatter={(v) => [Number(v).toFixed(4), 'PR-AUC']} labelFormatter={(l) => `Epoch ${l}`} />
                <Line dataKey="val_pr_auc" stroke="#3987e5" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Test-set evaluation">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 p-4 text-sm">
            {[
              ['Test scenarios', m.test_scenarios.toLocaleString()],
              ['Brier score', m.branch.brier.toFixed(4)],
              ['Expected calibration error', m.branch.ece.toFixed(4)],
              ['Failure base rate', `${(m.branch.positive_rate * 100).toFixed(2)}%`],
              ['Monte Carlo scenarios', m.monte_carlo.scenarios.toLocaleString()],
              ['Expected-shed error', `${m.monte_carlo.shed_mae_pct.toFixed(2)} pp`],
              ['Expected-shed correlation', m.monte_carlo.shed_corr.toFixed(3)],
              ['One physics pass (N-1)', `${b.physics_pass_ms.toFixed(0)} ms`],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-line/60 pb-2">
                <dt className="text-ink-2">{k}</dt><dd className="num font-medium">{v}</dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel title="CascadeGNN architecture">
          <div className="space-y-3 p-4">
            <Block t="Inputs" lines={['Bus: demand, served, capacity, injection, served after trip',
              'Branch: loading before & right after the trip (exact LODF physics), rating, susceptance, outage & contingency flags']} />
            <Block t={`${m.model.layers} × message-passing blocks`} lines={['edge ← edge + W·GELU(edge, from-bus, to-bus, global)',
              'bus ← bus + W·GELU(bus, Σ out-edges, Σ in-edges, global)', 'global ← global + MLP(global, mean bus, mean edge)']} />
            <Block t="Outputs" lines={['P(branch fails) — Platt-calibrated', 'Expected fraction of demand shed']} />
          </div>
          <div className="border-t border-line px-4 py-2.5 text-xs text-ink-2">
            {m.model.params.toLocaleString()} parameters · hidden {m.model.hidden} · trained {m.model.epochs} epochs · served with ONNX Runtime on CPU
          </div>
        </Panel>
      </div>
    </div>
  )
}

function Block({ t, lines }: { t: string; lines: string[] }) {
  return (
    <div className="rounded-lg border border-line bg-panel-2 p-3">
      <div className="mb-1.5 text-sm font-semibold">{t}</div>
      <ul className="space-y-1 text-xs text-ink-2">{lines.map((l) => <li key={l}>{l}</li>)}</ul>
    </div>
  )
}
