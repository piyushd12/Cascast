export interface Bus { id: number; name: string; load_mw: number; x: number; y: number }
export interface Branch {
  id: number; from: number; to: number; x: number; trafo: boolean
  rating_mw: number; base_flow_mw: number
}
export interface Grid { name: string; buses: Bus[]; branches: Branch[]; gens: { bus: number; pmax_mw: number }[] }

export interface GridState {
  tick: number; minute: number; clock: string; weather: 'clear' | 'heatwave' | 'storm'
  load_factor: number; demand_mw: number; served_mw: number; n_islands: number
  alive: number[]; flow: number[]; loading: number[]; load: number[]; served: number[]
  repairs: Record<string, number>
}

export interface Contingency { line: number; shed_pct: number; n_fail: number; cascade: number[]; max_prob: number }
export interface Risk {
  tick: number; minute: number; status: 'secure' | 'alert' | 'emergency'
  worst_shed_pct: number; n_critical: number; n_screened: number; latency_ms: number
  top: Contingency[]; vulnerability: number[]
  live_accuracy?: { checked: number; precision: number; recall: number; shed_mae_pct: number }
}

export interface GridEvent {
  id: number; minute: number; kind: 'fault' | 'cascade' | 'restored' | 'weather'; title: string
  lines?: number[]; cascade?: number[]; stages?: number[][]; shed_mw?: number; shed_pct?: number
  cause?: string; weather?: string; created_at?: string
}

export interface WhatIfResult {
  lines: number[]; load_scale: number; pre_overloaded: number[]
  gnn: { prob: number[]; shed_pct: number; predicted: number[]; latency_ms: number }
  physics: {
    failed: number[]; stage: number[]; shed_pct: number; shed_mw: number; n_stages: number
    islands: number; latency_ms: number
    stages: { tripped: number[]; loading: number[]; served_mw: number }[]
  }
  agreement: { jaccard: number; true_positive: number; false_positive: number; false_negative: number }
}

export interface Metrics {
  test_scenarios: number
  branch: Record<'pr_auc' | 'roc_auc' | 'precision' | 'recall' | 'f1' | 'ece' | 'positive_rate', number>
  scenario: Record<'cascade_rate' | 'cascade_detection_accuracy' | 'cascade_recall' | 'false_alarm_rate', number>
  load_shed: { mae_pct: number; mae_pct_on_cascades: number; corr: number }
  benchmark: { contingencies: number; physics_ms: number; gnn_onnx_ms: number; speedup: number }
  model: { params: number; hidden: number; layers: number; temperature: number; epochs: number }
  history: { epoch: number; train_loss: number; val_pr_auc: number; val_shed_mae: number }[]
}

export interface TimelinePoint {
  tick: number; minute: number; demand_mw: number; served_mw: number
  worst_shed_pct: number; n_critical: number; max_loading: number
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export const api = {
  grid: () => req<Grid>('/api/grid'),
  state: () => req<{ state: GridState; risk: Risk; events: GridEvent[]; live: boolean }>('/api/state'),
  whatif: (lines: number[], load_scale = 1) =>
    req<WhatIfResult>('/api/whatif', { method: 'POST', body: JSON.stringify({ lines, load_scale }) }),
  incidents: (limit = 100) => req<GridEvent[]>(`/api/incidents?limit=${limit}`),
  timeline: (limit = 720) => req<TimelinePoint[]>(`/api/timeline?limit=${limit}`),
  model: () => req<Metrics>('/api/model'),
  health: () => req<{ live: boolean; redis: boolean; postgres: boolean; copilot: boolean }>('/api/health'),
  command: (body: { cmd: 'trip' | 'restore_all' | 'weather'; lines?: number[]; value?: string }) =>
    req<{ queued: boolean }>('/api/commands', { method: 'POST', body: JSON.stringify(body) }),
}
