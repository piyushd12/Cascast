import clsx from 'clsx'
import { Activity, BrainCircuit, CloudLightning, FlaskConical, ScrollText, Sparkles, Sun, ThermometerSun, Zap } from 'lucide-react'
import type { ReactNode } from 'react'
import { NavLink } from 'react-router'
import { STATUS } from '../lib/format'
import { useLive } from '../lib/live'

const NAV = [
  { to: '/', label: 'Live', icon: Activity },
  { to: '/lab', label: 'What-If Lab', icon: FlaskConical },
  { to: '/incidents', label: 'Incidents', icon: ScrollText },
  { to: '/model', label: 'Model', icon: BrainCircuit },
]

const WEATHER = { clear: Sun, heatwave: ThermometerSun, storm: CloudLightning } as const

export function Layout({ children, onCopilot }: { children: ReactNode; onCopilot: () => void }) {
  const { connected, live, state, risk } = useLive()
  const status = risk ? STATUS[risk.status] : null
  const W = state ? WEATHER[state.weather] : Sun

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line bg-panel px-4">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-[#1b2633]"><Zap size={18} className="text-[#f5a524]" fill="#f5a524" /></div>
          <div className="leading-tight">
            <div className="font-semibold tracking-tight">Cascadia</div>
            <div className="text-[11px] text-muted">IEEE 118-bus digital twin</div>
          </div>
        </div>

        <nav className="ml-4 hidden items-center gap-1 md:flex">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end
              className={({ isActive }) => clsx('flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition',
                isActive ? 'bg-panel-2 text-ink' : 'text-ink-2 hover:text-ink')}>
              <Icon size={16} />{label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3 text-sm">
          {state && live && (
            <span className="hidden items-center gap-2 text-ink-2 sm:flex">
              <W size={16} /> <span className="num">{state.clock}</span>
            </span>
          )}
          {status && (
            <span className="flex items-center gap-2 rounded-full border border-line px-2.5 py-1 text-xs font-medium">
              <span className="h-2 w-2 rounded-full" style={{ background: status.color }} />{status.label}
            </span>
          )}
          <span className={clsx('flex items-center gap-1.5 text-xs', connected && live ? 'text-ink-2' : 'text-muted')}>
            <span className={clsx('h-1.5 w-1.5 rounded-full', connected && live ? 'bg-good pulse' : 'bg-muted')} />
            {connected ? (live ? 'Live' : 'Offline sim') : 'Disconnected'}
          </span>
          <button onClick={onCopilot}
            className="flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:brightness-110">
            <Sparkles size={15} /> Copilot
          </button>
        </div>
      </header>

      <nav className="flex shrink-0 gap-1 overflow-x-auto border-b border-line bg-panel px-2 py-1 md:hidden">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end
            className={({ isActive }) => clsx('flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm',
              isActive ? 'bg-panel-2 text-ink' : 'text-ink-2')}>
            <Icon size={15} />{label}
          </NavLink>
        ))}
      </nav>

      <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
    </div>
  )
}
