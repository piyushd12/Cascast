import { create } from 'zustand'
import type { GridEvent, GridState, Risk } from './api'

export interface Sample { tick: number; clock: string; demand: number; served: number; worst: number; critical: number }

interface LiveStore {
  connected: boolean
  live: boolean
  state: GridState | null
  risk: Risk | null
  events: GridEvent[]
  history: Sample[]
}

const HISTORY = 360

export const useLive = create<LiveStore>(() => ({
  connected: false, live: false, state: null, risk: null, events: [], history: [],
}))

function sample(state: GridState, risk: Risk | null): Sample {
  return {
    tick: state.tick, clock: state.clock.split('· ')[1] ?? state.clock,
    demand: state.demand_mw, served: state.served_mw,
    worst: risk?.worst_shed_pct ?? 0, critical: risk?.n_critical ?? 0,
  }
}

type Msg =
  | { type: 'snapshot'; data: { state: GridState; risk: Risk | null; events: GridEvent[]; live: boolean } }
  | { type: 'state'; data: GridState }
  | { type: 'risk'; data: Risk }
  | { type: 'event'; data: GridEvent }

function onMessage(msg: Msg) {
  const set = useLive.setState
  switch (msg.type) {
    case 'snapshot':
      set({ state: msg.data.state, risk: msg.data.risk, events: msg.data.events.slice().reverse(), live: msg.data.live })
      break
    case 'state':
      set({ state: msg.data, live: true })
      break
    case 'risk': {
      const { state, risk, history } = useLive.getState()
      if (risk && msg.data.tick < risk.tick) return // out-of-order result from another engine
      const next = state && state.tick === msg.data.tick
        ? [...history.slice(-(HISTORY - 1)), sample(state, msg.data)]
        : history
      set({ risk: msg.data, history: next })
      break
    }
    case 'event':
      set((s) => ({ events: [msg.data, ...s.events].slice(0, 100) }))
      break
  }
}

/** Connect to the live feed with exponential-backoff reconnects. Returns a disposer. */
export function connectLive(): () => void {
  let ws: WebSocket | null = null
  let retry = 0
  let stopped = false
  let timer: number | undefined

  const open = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws/live`)
    ws.onopen = () => { retry = 0; useLive.setState({ connected: true }) }
    ws.onmessage = (e) => onMessage(JSON.parse(e.data as string) as Msg)
    ws.onclose = () => {
      useLive.setState({ connected: false })
      if (!stopped) timer = window.setTimeout(open, Math.min(10_000, 500 * 2 ** retry++))
    }
  }
  open()
  return () => { stopped = true; window.clearTimeout(timer); ws?.close() }
}
