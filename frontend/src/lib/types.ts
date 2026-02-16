export type GeometryType = 'Point' | 'LineString'

export interface Feature {
  type: 'Feature'
  geometry: {
    type: GeometryType
    coordinates: any
  }
  properties: Record<string, any>
}

export interface FeatureCollection {
  type: 'FeatureCollection'
  features: Feature[]
}

export interface Project {
  schema_version: string
  name: string
  crs_epsg: number
  background?: {
    kind: 'georef' | 'calibrated'
    path: string
    opacity: number
    visible: boolean
    locked: boolean
    affine?: number[]
    bbox?: number[]
  }
  dem?: {
    path: string
    nodata?: number
    clamp_sources: boolean
  }
  point_sources: FeatureCollection
  line_sources: FeatureCollection
  barriers: FeatureCollection
  sections: FeatureCollection
  settings: {
    extent: [number, number, number, number]
    resolution: number
    receiver_height: number
    ground_factor: number
    humidity: number
    temperature_c: number
    pressure_hpa: number
  }
}

export const defaultProject: Project = {
  schema_version: '1.0',
  name: 'demo',
  crs_epsg: 32633,
  point_sources: { type: 'FeatureCollection', features: [] },
  line_sources: { type: 'FeatureCollection', features: [] },
  barriers: { type: 'FeatureCollection', features: [] },
  sections: { type: 'FeatureCollection', features: [] },
  settings: {
    extent: [0, 0, 2000, 2000],
    resolution: 25,
    receiver_height: 1.5,
    ground_factor: 0.5,
    humidity: 70,
    temperature_c: 20,
    pressure_hpa: 1013.25,
  },
}
