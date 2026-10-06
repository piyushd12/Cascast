import { useQuery } from '@tanstack/react-query'
import { FlaskConical } from 'lucide-react'
import { Link } from 'react-router'
import { Kind, Panel } from '../components/ui'
import { api } from '../lib/api'
import { fmtMinute } from '../lib/format'

export default function IncidentsPage() {
  const q = useQuery({ queryKey: ['incidents'], queryFn: () => api.incidents(200), refetchInterval: 5000 })
  const rows = q.data ?? []
  const cascades = rows.filter((r) => r.kind === 'cascade')
  const shed = rows.reduce((s, r) => s + (r.shed_mw ?? 0), 0)

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-5">
      <div className="grid grid-cols-3 gap-3">
        <Tile k="Incidents recorded" v={rows.length} />
        <Tile k="Cascades" v={cascades.length} />
        <Tile k="Load shed (sum of events)" v={`${Math.round(shed).toLocaleString()} MW`} />
      </div>
      <Panel title="Incident history" right={<span className="text-xs text-muted">Persisted in Postgres · refreshes every 5 s</span>}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-line text-left">
                {['Sim time', 'Type', 'Description', 'Initial', 'Cascaded', 'Shed', ''].map((h) => (
                  <th key={h} className="label px-4 py-2 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr><td colSpan={7} className="px-4 py-10 text-center text-muted">
                  {q.isLoading ? 'Loading…' : 'No incidents yet. Faults appear here as the simulation runs.'}
                </td></tr>
              )}
              {rows.map((r) => (
                <tr key={`${r.id}-${r.minute}`} className="border-b border-line/60 hover:bg-panel-2">
                  <td className="num whitespace-nowrap px-4 py-2 text-ink-2">{fmtMinute(r.minute)}</td>
                  <td className="px-4 py-2"><Kind kind={r.kind} /></td>
                  <td className="px-4 py-2 text-ink">{r.title}</td>
                  <td className="px-4 py-2 font-mono text-xs text-ink-2">{r.lines?.join(', ')}</td>
                  <td className="px-4 py-2 font-mono text-xs text-ink-2">{r.cascade?.length ? r.cascade.join(', ') : '—'}</td>
                  <td className="num px-4 py-2 text-right">{r.shed_pct ? <span className="text-critical">{r.shed_pct}%</span> : '—'}</td>
                  <td className="px-4 py-2 text-right">
                    {r.lines?.length ? (
                      <Link to={`/lab?lines=${r.lines.join(',')}`} title="Re-run on the current state"
                        className="inline-flex items-center gap-1 text-xs text-accent hover:underline">
                        <FlaskConical size={13} />Replay
                      </Link>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}

function Tile({ k, v }: { k: string; v: string | number }) {
  return (
    <div className="panel px-4 py-3">
      <div className="label">{k}</div>
      <div className="num mt-1 text-2xl font-semibold">{v}</div>
    </div>
  )
}
