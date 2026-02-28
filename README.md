# RUMORE - ISO 9613 V1 (desktop)

Applicazione desktop in **PySide6** per calcolo propagazione acustica semplificata ISO 9613 (V1):
- divergenza geometrica
- assorbimento aria (alpha fisso)
- schermatura barriere lineari (approccio engineering semplificato)

## Struttura

- `app/main.py` entrypoint UI
- `app/ui/` canvas e widget
- `app/model/` dataclass + parsing/validazioni
- `app/calc/` pipeline calcolo + isofone
- `app/io/` export GeoTIFF/PNG e progetto JSON
- `tests/` test minimi
- `build.bat`, `rumore.spec` packaging `.exe`

## Avvio in sviluppo

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Build EXE Windows

```bat
build.bat
```

Output in `dist/RUMORE/` con eseguibile windowed (`console=False`).

## Input tabella sorgenti

Colonne:
1. ID (auto)
2. Coord `"X, Y, Z"` (accetta anche `;`)
3. 63Hz
4. 125Hz
5. 250Hz
6. 500Hz
7. 1kHz
8. 2kHz
9. 4kHz
10. 8kHz
11. LwA_tot

Regola:
- compilare `LwA_tot`, oppure bande, oppure entrambi.
- se entrambi: controllo mismatch > 1.0 dB (errore nel log).

Esempio riga:

```text
1 | 500000, 5032000, 5 | 95 | 94 | 92 | 90 | 89 | 86 | 83 | 80 | 92
```

## Output

- **GeoTIFF** georeferenziato (EPSG scelto), banda singola dBA.
- **PNG layout** con mappa, sorgenti, barriere, isofone (step 2 dB), legenda, scala, freccia nord, metadati.

## Test

```bash
pytest -q
```

## Reset rapido locale

Per cancellare modifiche locali e file non tracciati senza operazioni manuali:

```bat
reset_local.bat
```

> Attenzione: operazione distruttiva (`git reset --hard` + `git clean -fd`).


## Test core ISO9613 plugin QGIS

Nel plugin `iso9613_lpa_raster` il motore numerico è ora separato in un modulo puro Python (`iso9613_lpa_raster/core`).
Per eseguire i test unitari fuori da QGIS:

```bash
pytest -q tests/test_core.py tests/test_regressions.py
```

Requisiti minimi: `numpy` e `pytest` (opzionale ma consigliato).
