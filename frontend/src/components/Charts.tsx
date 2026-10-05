import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Sample } from '../lib/live'

const AXIS = { stroke: '#6b7a8a', fontSize: 11, tickLine: false, axisLine: { stroke: '#223040' } }
const GRID = <CartesianGrid stroke="#1b2633" vertical={false} />
const TIP = {
  contentStyle: { background: '#16202b', border: '1px solid #223040', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#a9b6c3' },
  cursor: { stroke: '#6b7a8a', strokeDasharray: '3 3' },
}

export function DemandChart({ data }: { data: Sample[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="served" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3987e5" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#3987e5" stopOpacity={0} />
          </linearGradient>
        </defs>
        {GRID}
        <XAxis dataKey="clock" {...AXIS} minTickGap={40} />
        <YAxis {...AXIS} width={44} domain={['auto', 'auto']} tickFormatter={(v) => `${Math.round(v / 100) / 10}k`} />
        <Tooltip {...TIP} formatter={(v) => [`${Math.round(Number(v))} MW`]} />
        <Area type="monotone" dataKey="served" name="Served" stroke="#3987e5" strokeWidth={2} fill="url(#served)" isAnimationActive={false} />
        <Line type="monotone" dataKey="demand" name="Demand" stroke="#a9b6c3" strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function RiskChart({ data }: { data: Sample[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="risk" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#d95926" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#d95926" stopOpacity={0} />
          </linearGradient>
        </defs>
        {GRID}
        <XAxis dataKey="clock" {...AXIS} minTickGap={40} />
        <YAxis {...AXIS} width={44} tickFormatter={(v) => `${v}%`} />
        <Tooltip {...TIP} formatter={(v) => [`${Number(v).toFixed(2)}%`, 'Worst N-1 load shed']} />
        <Area type="stepAfter" dataKey="worst" stroke="#d95926" strokeWidth={2} fill="url(#risk)" isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}
