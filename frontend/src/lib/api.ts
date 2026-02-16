import { Project } from './types'

const API = 'http://localhost:8000'

export async function saveProject(projectId: string, project: Project) {
  return fetch(`${API}/projects/${projectId}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(project),
  }).then((r) => r.json())
}

export async function loadProject(projectId: string): Promise<Project> {
  return fetch(`${API}/projects/${projectId}`).then((r) => r.json())
}

export async function runCalculation(projectId: string) {
  return fetch(`${API}/projects/${projectId}/calculate`, { method: 'POST' }).then((r) => r.json())
}

export async function updateScenario(projectId: string, active_source_ids: string[]) {
  return fetch(`${API}/projects/${projectId}/scenario`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ active_source_ids }),
  }).then((r) => r.json())
}

export async function getContribution(projectId: string, sourceId: string) {
  return fetch(`${API}/projects/${projectId}/contribution/${sourceId}`).then((r) => r.json())
}

export async function runSection(projectId: string, section_feature_id: string) {
  return fetch(`${API}/projects/${projectId}/section`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ section_feature_id }),
  }).then((r) => r.json())
}
