import { useQuery } from '@tanstack/react-query'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Panel, Stat } from '../components/ui'
import { api } from '../lib/api'

const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`

export default function ModelPage() {
  const q = useQuery({ queryKey: ['model'], queryFn: api.model })
  if (q.isLoading) return <div className="p-6 text-muted">Loading…</div>
  if (!q.data) return <div className="p-6 text-muted">Model metrics are not available yet. Run <code className="font-mono">make train</code>.</div>
  const m = q.data

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Branch failure PR-AUC" value={m.branch.pr_auc.toFixed(3)} sub={`ROC-AUC ${m.branch.roc_auc.toFixed(3)}`} />
        <Stat label="Cascade detection" value={pct(m.scenario.cascade_detection_accuracy)}
          sub={`recall ${pct(m.scenario.cascade_recall)} · false alarms ${pct(m.scenario.false_alarm_rate)}`} />
        <Stat label="Load-shed error" value={`${m.load_shed.mae_pct.toFixed(2)} pp`} sub={`MAE · r = ${m.load_shed.corr.toFixed(3)}`} />
        <Stat label="Speed-up vs physics" value={`${m.benchmark.speedup.toFixed(0)}×`}
          sub={`${m.benchmark.contingencies} cascades in ${m.benchmark.gnn_onnx_ms.toFixed(1)} ms`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Validation PR-AUC by epoch">
          <div className="h-64 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={m.history} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#1b2633" vertical={false} />
                <XAxis dataKey="epoch" stroke="#6b7a8a" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7a8a" fontSize={11} tickLine={false} width={44} domain={['auto', 1]} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip contentStyle={{ background: '#16202b', border: '1px solid #223040', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => [Number(v).toFixed(4), 'PR-AUC']} labelFormatter={(l) => `Epoch ${l}`} />
                <Line dataKey="val_pr_auc" stroke="#3987e5" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Branch-level classification (test set)">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 p-4 text-sm">
            {[
              ['Precision', pct(m.branch.precision)], ['Recall', pct(m.branch.recall)],
              ['F1', m.branch.f1.toFixed(3)], ['Expected calibration error', m.branch.ece.toFixed(4)],
              ['Positive rate', pct(m.branch.positive_rate, 2)], ['Test scenarios', m.test_scenarios.toLocaleString()],
              ['Scenarios with a cascade', pct(m.scenario.cascade_rate)], ['Shed MAE on cascades', `${m.load_shed.mae_pct_on_cascades.toFixed(2)} pp`],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-line/60 pb-2">
                <dt className="text-ink-2">{k}</dt><dd className="num font-medium">{v}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      </div>

      <Panel title="CascadeGNN architecture">
        <div className="grid gap-4 p-4 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
          <Block t="Inputs" lines={['Bus: demand, served, capacity, injection', 'Branch: loading, rating, susceptance, outage & contingency flags']} />
          <Arrow />
          <Block t={`${m.model.layers} × message-passing blocks`} lines={['edge ← [edge, from-bus, to-bus, global]', 'bus ← [bus, Σ out-edges, Σ in-edges, global]', 'global ← [global, mean bus, mean edge]']} />
          <Arrow />
          <Block t="Outputs" lines={['P(branch fails) per branch, temperature-calibrated', 'Fraction of demand shed']} />
        </div>
        <div className="border-t border-line px-4 py-2.5 text-xs text-ink-2">
          {m.model.params.toLocaleString()} parameters · hidden {m.model.hidden} · temperature {m.model.temperature.toFixed(2)} ·
          served with ONNX Runtime on CPU
        </div>
      </Panel>
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

const Arrow = () => <div className="hidden text-center text-muted md:block">→</div>
