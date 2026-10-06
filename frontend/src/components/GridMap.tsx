import { select } from 'd3-selection'
import { zoom, zoomIdentity, type ZoomBehavior } from 'd3-zoom'
import { Maximize2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Grid } from '../lib/api'
import { fmtPct, loadingColor, riskColor } from '../lib/format'

export type MapMode = 'loading' | 'risk' | 'cascade'

interface Props {
  grid: Grid
  mode: MapMode
  loading?: number[]
  alive?: number[]
  /** risk mode: per-branch failure probability (0..1) */
  risk?: number[]
  /** cascade mode: stage at which each branch failed (-1 never, 0 initial trip) */
  stage?: number[]
  currentStage?: number
  /** branches to ring (e.g. GNN-predicted failures) */
  predicted?: number[]
  selected?: number[]
  shedBuses?: Set<number>
  onLineClick?: (line: number) => void
  className?: string
}

const W = 1000
const H = 720
const PAD = 36

interface Hover { line: number; x: number; y: number; w: number }

export function GridMap(p: Props) {
  const { grid, mode } = p
  const svgRef = useRef<SVGSVGElement>(null)
  const gRef = useRef<SVGGElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [hover, setHover] = useState<Hover | null>(null)

  const pos = useMemo(
    () => grid.buses.map((b) => [PAD + b.x * (W - 2 * PAD), PAD + b.y * (H - 2 * PAD)] as const),
    [grid],
  )
  const genBuses = useMemo(() => new Set(grid.gens.map((g) => g.bus)), [grid])
  const maxRating = useMemo(() => Math.max(...grid.branches.map((b) => b.rating_mw)), [grid])
  const selected = useMemo(() => new Set(p.selected ?? []), [p.selected])
  const predicted = useMemo(() => new Set(p.predicted ?? []), [p.predicted])

  useEffect(() => {
    const svg = select(svgRef.current!)
    const z = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 8])
      .on('zoom', (e) => gRef.current?.setAttribute('transform', e.transform.toString()))
    zoomRef.current = z
    svg.call(z)
    return () => { svg.on('.zoom', null) }
  }, [])

  const reset = () => {
    if (svgRef.current && zoomRef.current) select(svgRef.current).call(zoomRef.current.transform, zoomIdentity)
  }

  const lineStyle = (e: number) => {
    const dead = p.alive ? !p.alive[e] : false
    const width = 1.2 + 3.2 * Math.sqrt(grid.branches[e].rating_mw / maxRating)
    if (mode === 'cascade' && p.stage) {
      const s = p.stage[e]
      const cur = p.currentStage ?? 0
      if (s === 0) return { stroke: '#ffffff', width: width + 1, dash: '6 4', opacity: 1, pulse: cur === 0 }
      if (s > 0 && s <= cur) return { stroke: '#e5484d', width: width + 1, dash: '6 4', opacity: 1, pulse: s === cur }
      if (dead) return { stroke: '#3a4350', width, dash: '3 5', opacity: 0.6, pulse: false }
      return { stroke: loadingColor(p.loading?.[e] ?? 0), width, dash: undefined, opacity: 0.95, pulse: false }
    }
    if (dead) return { stroke: '#3a4350', width, dash: '3 5', opacity: 0.6, pulse: false }
    if (mode === 'risk') {
      const v = p.risk?.[e] ?? 0
      return { stroke: riskColor(v), width: width + (v > 0.5 ? 1.5 : 0), dash: undefined, opacity: 0.35 + 0.65 * Math.min(1, v * 1.5 + 0.2), pulse: v > 0.8 }
    }
    const l = p.loading?.[e] ?? 0
    return { stroke: loadingColor(l), width, dash: undefined, opacity: 0.95, pulse: l > 0.97 }
  }

  const h = hover !== null ? grid.branches[hover.line] : null

  return (
    <div className={`relative min-h-0 overflow-hidden ${p.className ?? ''}`}>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="h-full w-full cursor-grab select-none active:cursor-grabbing"
        role="img" aria-label={`${grid.name} network map`}>
        <g ref={gRef}>
          {/* selection halos */}
          {grid.branches.map((b) => selected.has(b.id) && (
            <line key={`sel${b.id}`} x1={pos[b.from][0]} y1={pos[b.from][1]} x2={pos[b.to][0]} y2={pos[b.to][1]}
              stroke="#ffffff" strokeOpacity={0.35} strokeWidth={12} strokeLinecap="round" />
          ))}
          {grid.branches.map((b) => predicted.has(b.id) && (
            <line key={`pred${b.id}`} x1={pos[b.from][0]} y1={pos[b.from][1]} x2={pos[b.to][0]} y2={pos[b.to][1]}
              stroke="#fab219" strokeOpacity={0.45} strokeWidth={9} strokeLinecap="round" />
          ))}
          {grid.branches.map((b) => {
            const s = lineStyle(b.id)
            return (
              <line key={b.id} x1={pos[b.from][0]} y1={pos[b.from][1]} x2={pos[b.to][0]} y2={pos[b.to][1]}
                stroke={s.stroke} strokeWidth={s.width} strokeOpacity={s.opacity} strokeDasharray={s.dash}
                strokeLinecap="round" className={s.pulse ? 'pulse' : undefined} />
            )
          })}
          {grid.buses.map((b) => {
            const [x, y] = pos[b.id]
            const shed = p.shedBuses?.has(b.id)
            const gen = genBuses.has(b.id)
            return gen
              ? <rect key={b.id} x={x - 5} y={y - 5} width={10} height={10} rx={2}
                  fill={shed ? '#e5484d' : '#0b0f14'} stroke="#3987e5" strokeWidth={2} />
              : <circle key={b.id} cx={x} cy={y} r={3.6 + Math.sqrt(b.load_mw) / 7}
                  fill={shed ? '#e5484d' : '#a9b6c3'} stroke="#0b0f14" strokeWidth={1.5} />
          })}
          {/* wide invisible hit targets */}
          {grid.branches.map((b) => (
            <line key={`hit${b.id}`} x1={pos[b.from][0]} y1={pos[b.from][1]} x2={pos[b.to][0]} y2={pos[b.to][1]}
              stroke="transparent" strokeWidth={14} className={p.onLineClick ? 'cursor-pointer' : undefined}
              onMouseMove={(ev) => {
                const r = svgRef.current!.getBoundingClientRect()
                setHover({ line: b.id, x: ev.clientX - r.left, y: ev.clientY - r.top, w: r.width })
              }}
              onMouseLeave={() => setHover(null)}
              onClick={() => p.onLineClick?.(b.id)} />
          ))}
        </g>
      </svg>

      <button onClick={reset} title="Reset zoom" aria-label="Reset zoom"
        className="absolute right-3 top-3 rounded-md border border-line bg-panel-2/80 p-1.5 text-ink-2 hover:text-ink">
        <Maximize2 size={14} />
      </button>

      {h && hover && (
        <div className="pointer-events-none absolute z-10 min-w-44 rounded-lg border border-line bg-panel-2/95 px-3 py-2 text-xs shadow-xl backdrop-blur"
          style={{ left: Math.min(hover.x + 14, hover.w - 190), top: hover.y + 14 }}>
          <div className="mb-1 font-semibold text-ink">
            {h.trafo ? 'Transformer' : 'Line'} {h.id}
            <span className="ml-1.5 font-normal text-muted">bus {grid.buses[h.from].name} → {grid.buses[h.to].name}</span>
          </div>
          <Row k="Rating" v={`${h.rating_mw.toFixed(0)} MW`} />
          {p.loading && <Row k="Loading" v={fmtPct(p.loading[h.id])} />}
          {p.alive && !p.alive[h.id] && <Row k="Status" v="Out of service" />}
          {mode === 'risk' && p.risk && <Row k="Failure probability" v={fmtPct(p.risk[h.id])} />}
          {mode === 'cascade' && p.stage && p.stage[h.id] >= 0 &&
            <Row k="Failed at" v={p.stage[h.id] === 0 ? 'initial trip' : `stage ${p.stage[h.id]}`} />}
          {predicted.has(h.id) && <Row k="GNN" v="predicted to fail" />}
        </div>
      )}
    </div>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 text-ink-2">
      <span>{k}</span><span className="num text-ink">{v}</span>
    </div>
  )
}

export function MapLegend({ mode }: { mode: MapMode }) {
  const items =
    mode === 'risk'
      ? [['#2a3746', 'Low'], ['#ec835a', 'Elevated'], ['#e5484d', 'Likely to fail (p ≥ 50%)']]
      : mode === 'cascade'
        ? [['#ffffff', 'Initial trip'], ['#e5484d', 'Failed in this run'], ['#fab219', 'GNN: likely to fail (halo)']]
        : [['#3b6ea8', '< 60%'], ['#f5a524', '60–85%'], ['#ec835a', '85–100%'], ['#e5484d', 'Overloaded']]
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map(([c, l]) => (
        <span key={l} className="flex items-center gap-1.5">
          <span className="inline-block h-1 w-5 rounded-full" style={{ background: c }} />{l}
        </span>
      ))}
      <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm border-2 border-accent" />Generator</span>
      <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full bg-critical" />Load shed</span>
    </div>
  )
}
