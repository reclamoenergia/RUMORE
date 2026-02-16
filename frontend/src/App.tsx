import { useMemo, useState } from 'react'
import MapEditor from './components/MapEditor'
import { getContribution, loadProject, runCalculation, runSection, saveProject, updateScenario } from './lib/api'
import { Feature, Project, defaultProject } from './lib/types'

function featureKind(mode: string) {
  if (mode === 'point') return 'point_sources'
  if (mode === 'line') return 'line_sources'
  if (mode === 'barrier') return 'barriers'
  if (mode === 'section') return 'sections'
  return ''
}

export default function App() {
  const [projectId, setProjectId] = useState('demo')
  const [project, setProject] = useState<Project>(defaultProject)
  const [mode, setMode] = useState<'point' | 'line' | 'barrier' | 'section' | 'select'>('point')
  const [outputRaster, setOutputRaster] = useState('')
  const [sectionPng, setSectionPng] = useState('')

  const allSources = useMemo(() => [...project.point_sources.features, ...project.line_sources.features], [project])

  const onCreate = (feature: Feature) => {
    const kind = featureKind(mode)
    if (!kind) return
    setProject((prev) => ({
      ...prev,
      [kind]: {
        ...prev[kind as keyof Project],
        features: [...(prev[kind as keyof Project] as any).features, feature],
      },
    }))
  }

  const setOnlySource = async (id: string) => {
    const active = [id]
    const res = await updateScenario(projectId, active)
    setOutputRaster(res.scenario_raster)
  }

  const toggle = async (id: string) => {
    const modified = allSources.map((s) =>
      s.properties.id === id ? { ...s, properties: { ...s.properties, active: !s.properties.active } } : s,
    )
    const point = modified.filter((f) => f.geometry.type === 'Point')
    const line = modified.filter((f) => f.geometry.type === 'LineString' && !project.barriers.features.some((b) => b.properties.id === f.properties.id))
    setProject((prev) => ({ ...prev, point_sources: { type: 'FeatureCollection', features: point }, line_sources: { type: 'FeatureCollection', features: line } }))
    const active = modified.filter((s) => s.properties.active).map((s) => s.properties.id)
    const res = await updateScenario(projectId, active)
    setOutputRaster(res.scenario_raster)
  }

  return (
    <div className="app">
      <aside className="panel">
        <h1>Rumore</h1>
        <input value={projectId} onChange={(e) => setProjectId(e.target.value)} placeholder="project id" />
        <div className="toolbar">
          {['point', 'line', 'barrier', 'section', 'select'].map((m) => (
            <button key={m} onClick={() => setMode(m as any)} className={mode === m ? 'active' : ''}>{m}</button>
          ))}
        </div>
        <button onClick={() => saveProject(projectId, project)}>Salva progetto</button>
        <button onClick={async () => setProject(await loadProject(projectId))}>Carica progetto</button>
        <button onClick={async () => {
          const res = await runCalculation(projectId)
          setOutputRaster(res.scenario_raster)
        }}>Calcola</button>
        <h3>Sorgenti</h3>
        {allSources.map((s) => (
          <div key={s.properties.id} className="row">
            <label>
              <input type="checkbox" checked={!!s.properties.active} onChange={() => toggle(s.properties.id)} />
              {s.properties.name || s.properties.id}
            </label>
            <button onClick={() => setOnlySource(s.properties.id)}>Solo questa</button>
            <button onClick={async () => {
              const c = await getContribution(projectId, s.properties.id)
              setOutputRaster(c.raster)
            }}>Contributo</button>
          </div>
        ))}
        <h3>Sezione</h3>
        <button onClick={async () => {
          const f = project.sections.features[0]
          if (!f) return
          const res = await runSection(projectId, f.properties.id)
          setSectionPng(res.png)
        }}>Calcola sezione</button>
        {sectionPng && <img src={`http://localhost:8000/files?path=${encodeURIComponent(sectionPng)}`} alt="section" className="preview" />}
        {outputRaster && <p>Raster: {outputRaster}</p>}
      </aside>
      <main>
        <MapEditor mode={mode} onCreate={onCreate} />
      </main>
    </div>
  )
}
