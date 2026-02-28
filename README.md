# RUMORE - QGIS Plugin Only

Questo repository contiene solo il plugin QGIS `iso9613_lpa_raster` e i test del core numerico.

## Contenuto repo

- `iso9613_lpa_raster/` plugin QGIS (Processing provider, algoritmi raster e ricettori, metadata, dialog).
- `tests/` test unitari/regressione del core numerico indipendente da QGIS.
- `pytest.ini` configurazione test.

## Sviluppo e test

Prerequisiti minimi per i test locali:

- Python 3.10+
- `numpy`
- `pytest`

Esecuzione:

```bash
pytest -q tests/test_core.py tests/test_regressions.py
```

## Documentazione plugin

Per installazione, parametri e note d'uso del plugin:

- `iso9613_lpa_raster/README.md`
