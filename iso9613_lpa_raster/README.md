# ISO9613 LpA Raster (v1)

Plugin QGIS 3.x per calcolare un raster GeoTIFF `LpA_dB` da sorgenti puntuali con `LwA` su una griglia coincidente con il DEM (window limitata da BAF), usando distanza 3D con orografia esplicita.

## Funzionalità principali

## Nuove funzioni (v0.2.0)

- **Auto-load output raster in progetto** dal dialog tramite checkbox "Aggiungi output raster al progetto".
- **Uso sole sorgenti selezionate** dal layer input (con fallback automatico a tutte se non c'è selezione o layer non selezionabile).
- **Nuovo algoritmo Processing** `Acoustics -> ISO9613 LpA Receptors (v1)` per calcolo su layer di ricettori puntuali con output vettoriale e campo `LpA_dB`.

- Algoritmo in **Processing Toolbox**: `Acoustics -> ISO9613 LpA Raster (v1)`.
- Plugin classico con menu/toolbar e dialog minimale per avviare il calcolo.
- Input:
  - DEM raster (metri, CRS metrico)
  - Sorgenti puntuali con campi numerici `h_src` e `LwA` (+ `name` opzionale)
- Modello v1:
  - `Adiv = 20*log10(d)+11`
  - `Aatm = alpha_atm * d`
  - `Agr` opzionale semplificato: `clip(k(G)*(1-exp(-d/200)),0,3)` con `k(G)=3*G`
  - Nessuna barriera (v1)
- Somma energetica su tutte le sorgenti.
- Tiling NumPy (tile size interno 512) + progress bar + cancel.

## Limitazioni v1

- `alpha_atm` è un coefficiente manuale costante (dB/m).
- `Agr` è un modello semplificato, **non** full ISO 9613.
- Nessun termine di schermatura/barriera.

## Installazione

1. Copiare la cartella `iso9613_lpa_raster` nella cartella plugin utente QGIS.
2. Riavviare QGIS.
3. Abilitare il plugin da **Plugin > Gestisci e Installa Plugin**.

## Parametri Processing

- `DEM`
- `Sorgenti`
- `Campo name` (opzionale)
- `Campo h_src` (numerico)
- `Campo LwA` (numerico)
- `h_rec`
- `BAF`
- `alpha_atm`
- `abilita_suolo`
- `G`
- `d_min`
- `OUTPUT` GeoTIFF

## Note output

- Output: GeoTIFF single-band float32 `LpA_dB`.
- NoData output: `-9999`.
- Viene calcolata solo la window derivata da `bbox(union(buffer(sorgenti, BAF))) ∩ DEM extent`.

## Test minimi consigliati

1. **DEM piano + 1 sorgente + alpha=0 + suolo off**
   - Verificare decrescita `LpA` con distanza.
   - Verificare effetto delle altezze `h_src` / `h_rec` (distanza 3D).

2. **DEM con pendenza/gradino**
   - Confrontare output con caso planimetrico: i risultati devono variare per effetto della quota (`z_r - z_s`).

## Dipendenze

- QGIS 3.22+
- Python in ambiente QGIS
- `numpy`
- `osgeo.gdal`

## Licenza

MIT
