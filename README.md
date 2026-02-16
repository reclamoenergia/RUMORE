# RUMORE Monorepo

Applicazione open-source per editing sorgenti/barriere e calcolo rumore outdoor con backend FastAPI e frontend React/Leaflet.

## Struttura repo

- `frontend/` React + Vite + TypeScript + Leaflet + Leaflet.Draw
- `backend/` FastAPI + motore acustico ISO 9613-2 (MVP ingegneristico)
- `backend/data/demo/` progetto demo e sfondo placeholder (SVG testuale PR-safe)
- `docs/` note architetturali

## Checklist feature (milestone)

- [x] **M1** Editor 2D con toolbox (sorgenti puntuali/lineari, barriere, sezione), salvataggio/caricamento `project.json`.
- [x] **M1** Sfondo: config georeferenziata o calibrata (2/3 punti) nel modello progetto.
- [x] **M2** DEM opzionale nel modello dati con clamp configurabile e fallback terreno piatto.
- [x] **M3** Calcolo griglia 2D e output GeoTIFF scenario + export isofone GeoJSON.
- [x] **M4** Cache per-feature in energia (`.npy` float32), update scenario rapido via ON/OFF sorgenti, raster contributo singolo.
- [x] **M5** Sezione verticale lungo polilinea con output PNG + CSV.

## Esecuzione locale (dev)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Apri `http://localhost:5173`.

## Formato `project.json`

Campi principali:

- `crs_epsg`: CRS metrico progetto (es. 32633).
- `background`: `{kind: georef|calibrated, path, affine?, bbox?, opacity, visible, locked}`.
- `dem`: `{path, nodata, clamp_sources}` opzionale.
- `point_sources` / `line_sources` / `barriers` / `sections`: FeatureCollection GeoJSON.
- `settings`: extent, resolution, receiver_height, meteo/suolo.

### Properties GeoJSON usate

- Point source: `id`, `name`, `active`, `lwa`, opzionale `octave_lw`.
- Line source: `id`, `name`, `active`, `lwa_per_m`, opzionale `octave_lw_per_m`.
- Barrier: `id`, `name`, `active`, opzionale `height`, `base_elevation`.
- Section: `id`, `name`, `active`, `step_s`, `z_min`, `z_max`, `z_step`.

## API principali

- `POST /projects/{id}` salva progetto
- `GET /projects/{id}` carica progetto
- `POST /upload` upload file (immagini/DEM)
- `POST /projects/{id}/calculate` calcolo completo + cache + scenario + isofone
- `POST /projects/{id}/scenario` update rapido da lista sorgenti attive
- `GET /projects/{id}/contribution/{source_id}` raster contributo singola feature
- `POST /projects/{id}/section` calcolo sezione verticale

## Note modello acustico

Implementazione ISO 9613-2 in forma MVP:

- divergenza geometrica,
- assorbimento aria parametrico,
- termine suolo semplificato,
- attenuazione barriera conservativa (occlusione LOS con barriera dominante),
- somma energetica in dominio energia e conversione dB.

Parametri meteo/suolo sono configurabili in `settings`.

## Demo

Usa `backend/data/demo/project.json` come base iniziale (3 sorgenti puntuali, 1 lineare, 2 barriere, 1 sezione) con sfondo `backend/data/demo/background_placeholder.svg` (asset testuale, senza binari).


## Note PR-safe

Per evitare errori su tool che non supportano file binari durante la creazione PR, la demo include uno sfondo SVG testuale versionato nel repository invece di PNG/JPG.
