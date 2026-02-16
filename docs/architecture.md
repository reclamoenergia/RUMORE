# Architecture notes

- Cache invalidation: hash SHA256 del `ProjectModel` serializzato (`fingerprint`) salvato in `backend/data/cache/<project>/meta.json`.
- Cache contributi: un file `.npy` per feature sorgente in energia (Ei = 10^(Li/10)).
- Update scenario: somma rapida dei soli `.npy` attivi + export GeoTIFF.
- Sezione verticale: campionamento lungo polilinea, matrice (s,z), export PNG heatmap e CSV long format.
