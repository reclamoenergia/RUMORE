import { useEffect } from 'react'
import L from 'leaflet'
import 'leaflet-draw'
import { Feature } from '../lib/types'

type Props = {
  mode: 'point' | 'line' | 'barrier' | 'section' | 'select'
  onCreate: (f: Feature) => void
}

export function setupDraw(map: L.Map, mode: Props['mode']) {
  if (mode === 'select') return
  if (mode === 'point') {
    const drawer = new L.Draw.Marker(map)
    drawer.enable()
    return
  }
  const polylineDrawer = new L.Draw.Polyline(map)
  polylineDrawer.enable()
}

export default function MapEditor({ mode, onCreate }: Props) {
  useEffect(() => {
    const map = L.map('map', {
      crs: L.CRS.Simple,
      center: [1000, 1000],
      zoom: 0,
      minZoom: -4,
    })

    const bounds = [
      [0, 0],
      [2000, 2000],
    ] as L.LatLngBoundsExpression

    L.rectangle(bounds, { color: '#777', weight: 1 }).addTo(map)
    map.fitBounds(bounds)

    const drawn = new L.FeatureGroup()
    map.addLayer(drawn)

    map.on(L.Draw.Event.CREATED, (e: any) => {
      const layer = e.layer
      drawn.addLayer(layer)
      const gj = layer.toGeoJSON()
      const id = crypto.randomUUID()
      const props: Record<string, any> = { id, name: id, active: true }
      if (mode === 'point') props.lwa = 95
      if (mode === 'line') props.lwa_per_m = 80
      onCreate({ ...gj, properties: props } as Feature)
    })

    setupDraw(map, mode)

    return () => map.remove()
  }, [mode, onCreate])

  return <div id="map" style={{ height: '70vh', width: '100%' }} />
}
