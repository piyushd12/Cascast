export const fmtPct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
export const fmtMW = (x: number) => `${Math.round(x).toLocaleString()} MW`

function mix(a: string, b: string, t: number) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(',')})`
}

/** Branch loading: calm blue → amber → orange → red at/above the thermal limit. */
export function loadingColor(l: number) {
  if (l >= 1) return '#e5484d'
  if (l >= 0.85) return mix('#f5a524', '#ec835a', (l - 0.85) / 0.15)
  if (l >= 0.6) return mix('#c08a1e', '#f5a524', (l - 0.6) / 0.25) // RGB blue→amber is mud-grey
  return mix('#2b4766', '#3b6ea8', l / 0.6)
}

/** Failure probability: slate → orange → red. */
export function riskColor(p: number) {
  if (p >= 0.5) return mix('#ec835a', '#e5484d', Math.min(1, (p - 0.5) / 0.4))
  return mix('#2a3746', '#ec835a', p / 0.5)
}

export const STATUS = {
  secure: { color: '#0ca30c', label: 'Secure' },
  alert: { color: '#fab219', label: 'Alert' },
  emergency: { color: '#e5484d', label: 'Emergency' },
} as const

export const fmtMinute = (m: number) => {
  const t = Math.floor(m)
  return `Day ${Math.floor(t / 1440) + 1} · ${String(Math.floor((t % 1440) / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`
}
