import { useQuery } from '@tanstack/react-query'
import { lazy, Suspense, useEffect, useState } from 'react'
import { Route, Routes } from 'react-router'
import { Copilot } from './components/Copilot'
import { Layout } from './components/Layout'
import { api } from './lib/api'
import { connectLive } from './lib/live'
import LivePage from './pages/LivePage'

const LabPage = lazy(() => import('./pages/LabPage'))
const IncidentsPage = lazy(() => import('./pages/IncidentsPage'))
const ModelPage = lazy(() => import('./pages/ModelPage'))

export default function App() {
  const [copilotOpen, setCopilotOpen] = useState(false)
  const grid = useQuery({ queryKey: ['grid'], queryFn: api.grid, staleTime: Infinity })

  useEffect(() => connectLive(), [])

  return (
    <Layout onCopilot={() => setCopilotOpen((o) => !o)}>
      {grid.isError && (
        <div className="panel m-6 p-6 text-ink-2">
          Can’t reach the Cascast API. Start the backend (<code className="font-mono">make dev</code> or{' '}
          <code className="font-mono">docker compose up</code>) and reload.
        </div>
      )}
      {grid.data && (
        <Suspense fallback={<div className="p-6 text-muted">Loading…</div>}>
          <Routes>
            <Route path="/" element={<LivePage grid={grid.data} />} />
            <Route path="/lab" element={<LabPage grid={grid.data} />} />
            <Route path="/incidents" element={<IncidentsPage />} />
            <Route path="/model" element={<ModelPage />} />
          </Routes>
        </Suspense>
      )}
      <Copilot open={copilotOpen} onClose={() => setCopilotOpen(false)} />
    </Layout>
  )
}
