# -*- coding: utf-8 -*-

import math

import numpy as np
from osgeo import gdal

from ..core.iso9613_core import compute_lpa_from_sources_grid

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSource,
    QgsField,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProject,
    QgsUnitTypes,
    Qgis,
)


class ISO9613LpaRasterAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    SOURCES = "SOURCES"
    FIELD_NAME = "FIELD_NAME"
    FIELD_HSRC = "FIELD_HSRC"
    FIELD_LWA = "FIELD_LWA"
    H_REC = "H_REC"
    BAF = "BAF"
    ALPHA_ATM = "ALPHA_ATM"
    ENABLE_GROUND = "ENABLE_GROUND"
    G = "G"
    D_MIN = "D_MIN"
    OUTPUT = "OUTPUT"

    TILE_SIZE = 512
    OUTPUT_NODATA = -9999.0

    def name(self):
        return "iso9613_lpa_raster_v1"

    def displayName(self):
        return "ISO9613 LpA Raster (v1)"

    def group(self):
        return "Acoustics"

    def groupId(self):
        return "acoustics"

    def shortHelpString(self):
        return (
            "Calcola un raster LpA (dB) su griglia DEM con distanza 3D, "
            "assorbimento atmosferico manuale e termine suolo semplificato (opzionale)."
        )

    def createInstance(self):
        return ISO9613LpaRasterAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "DEM (elevazione, metri)")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SOURCES,
                "Sorgenti puntuali",
                [QgsProcessing.TypeVectorPoint],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_NAME,
                "Campo nome sorgente (opzionale)",
                parentLayerParameterName=self.SOURCES,
                type=QgsProcessingParameterField.String,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_HSRC,
                "Campo h_src (m)",
                parentLayerParameterName=self.SOURCES,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_LWA,
                "Campo LwA (dB re 1 pW)",
                parentLayerParameterName=self.SOURCES,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.H_REC,
                "h_rec (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=4.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BAF,
                "BAF (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALPHA_ATM,
                "alpha_atm (dB/m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_GROUND,
                "Abilita attenuazione suolo semplificata",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.G,
                "G (0..1)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.D_MIN,
                "d_min (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.001,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                "Output GeoTIFF (LpA_dB)",
                fileFilter="GeoTIFF (*.tif *.tiff)",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        src_source = self.parameterAsSource(parameters, self.SOURCES, context)
        field_name = self.parameterAsString(parameters, self.FIELD_NAME, context)
        field_hsrc = self.parameterAsString(parameters, self.FIELD_HSRC, context)
        field_lwa = self.parameterAsString(parameters, self.FIELD_LWA, context)
        h_rec = self.parameterAsDouble(parameters, self.H_REC, context)
        baf = self.parameterAsDouble(parameters, self.BAF, context)
        alpha_atm = self.parameterAsDouble(parameters, self.ALPHA_ATM, context)
        enable_ground = self.parameterAsBool(parameters, self.ENABLE_GROUND, context)
        g_value = self.parameterAsDouble(parameters, self.G, context)
        d_min = self.parameterAsDouble(parameters, self.D_MIN, context)
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)

        if dem_layer is None:
            raise QgsProcessingException("DEM non valido.")
        if src_source is None:
            raise QgsProcessingException("Layer sorgenti non valido.")

        if dem_layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
            raise QgsProcessingException("Il CRS del DEM deve avere unità in metri.")

        wkb = src_source.wkbType()
        if QgsWkbTypes.geometryType(wkb) != QgsWkbTypes.PointGeometry:
            raise QgsProcessingException("Il layer sorgenti deve essere di tipo Point.")

        fields = src_source.fields()
        h_field = fields.lookupField(field_hsrc)
        lwa_field = fields.lookupField(field_lwa)
        if h_field < 0 or lwa_field < 0:
            raise QgsProcessingException("Campi h_src/LwA non trovati.")

        if not self._is_numeric_field(fields[h_field]) or not self._is_numeric_field(fields[lwa_field]):
            raise QgsProcessingException("I campi h_src e LwA devono essere numerici.")

        dem_path = dem_layer.source()
        dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
        if dem_ds is None:
            raise QgsProcessingException(f"Impossibile aprire il DEM con GDAL: {dem_path}")

        dem_band = dem_ds.GetRasterBand(1)
        dem_nodata = dem_band.GetNoDataValue()
        gt = dem_ds.GetGeoTransform()
        if gt is None:
            raise QgsProcessingException("GeoTransform DEM non disponibile.")

        cols_dem = dem_ds.RasterXSize
        rows_dem = dem_ds.RasterYSize
        dem_extent = dem_layer.extent()

        transform = None
        if src_source.sourceCrs().isValid() and src_source.sourceCrs() != dem_layer.crs():
            transform = QgsCoordinateTransform(
                src_source.sourceCrs(), dem_layer.crs(), QgsProject.instance()
            )

        all_sources = []
        n_total = 0
        for feat in src_source.getFeatures():
            if feedback.isCanceled():
                break
            n_total += 1
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if transform is not None:
                geom = QgsGeometry(geom)
                geom.transform(transform)
            pt = geom.asPoint()
            name = feat[field_name] if field_name and field_name in fields.names() else f"src_{feat.id()}"
            try:
                h_src = float(feat[field_hsrc])
                lwa = float(feat[field_lwa])
            except (TypeError, ValueError):
                feedback.pushWarning(f"Sorgente {name}: h_src/LwA non numerici, scartata.")
                continue
            all_sources.append((name, float(pt.x()), float(pt.y()), h_src, lwa))

        if not all_sources:
            raise QgsProcessingException("Nessuna sorgente valida trovata.")

        domain = self._compute_domain_bbox(all_sources, baf)
        domain = domain.intersect(dem_extent)
        if domain.isEmpty():
            raise QgsProcessingException("Il dominio BAF non interseca il DEM.")

        col_min, col_max, row_min, row_max = self._extent_to_window(domain, gt, cols_dem, rows_dem)
        win_cols = col_max - col_min + 1
        win_rows = row_max - row_min + 1
        if win_cols <= 0 or win_rows <= 0:
            raise QgsProcessingException("Window DEM non valida.")

        dem_window = dem_band.ReadAsArray(col_min, row_min, win_cols, win_rows)
        if dem_window is None:
            raise QgsProcessingException("Errore lettura window DEM.")
        dem_window = dem_window.astype(np.float64, copy=False)

        valid_sources = []
        discarded = 0
        for name, x_s, y_s, h_src, lwa in all_sources:
            c_s, r_s = self._xy_to_colrow(x_s, y_s, gt)
            if c_s < col_min or c_s > col_max or r_s < row_min or r_s > row_max:
                feedback.pushWarning(f"Sorgente {name} fuori DEM/window, scartata.")
                discarded += 1
                continue
            z_dem = dem_band.ReadAsArray(c_s, r_s, 1, 1)
            if z_dem is None:
                feedback.pushWarning(f"Sorgente {name}: impossibile campionare DEM, scartata.")
                discarded += 1
                continue
            z_dem_val = float(z_dem[0, 0])
            if self._is_nodata_value(z_dem_val, dem_nodata):
                feedback.pushWarning(f"Sorgente {name} su NoData DEM, scartata.")
                discarded += 1
                continue
            z_s = z_dem_val + h_src
            valid_sources.append((name, x_s, y_s, z_s, lwa))

        if not valid_sources:
            raise QgsProcessingException("Tutte le sorgenti sono state scartate.")

        feedback.pushInfo(f"Sorgenti totali: {n_total}")
        feedback.pushInfo(f"Sorgenti usate: {len(valid_sources)}")
        feedback.pushInfo(f"Sorgenti scartate: {discarded}")
        feedback.pushInfo(
            f"Dominio: xmin={domain.xMinimum():.3f}, xmax={domain.xMaximum():.3f}, "
            f"ymin={domain.yMinimum():.3f}, ymax={domain.yMaximum():.3f}"
        )
        feedback.pushInfo(f"Window: cols={win_cols}, rows={win_rows}, tile_size={self.TILE_SIZE}")
        feedback.pushInfo(
            f"Parametri: h_rec={h_rec}, BAF={baf}, alpha={alpha_atm}, "
            f"abilita_suolo={enable_ground}, G={g_value}, d_min={d_min}"
        )

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, win_cols, win_rows, 1, gdal.GDT_Float32)
        if out_ds is None:
            raise QgsProcessingException(f"Impossibile creare output: {output_path}")

        out_gt = list(gt)
        out_gt[0] = gt[0] + col_min * gt[1] + row_min * gt[2]
        out_gt[3] = gt[3] + col_min * gt[4] + row_min * gt[5]
        out_ds.SetGeoTransform(tuple(out_gt))
        out_ds.SetProjection(dem_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(self.OUTPUT_NODATA)

        nodata_mask = self._build_nodata_mask(dem_window, dem_nodata)

        ntr = math.ceil(win_rows / self.TILE_SIZE)
        ntc = math.ceil(win_cols / self.TILE_SIZE)
        total_tiles = ntr * ntc
        done = 0

        for r0 in range(0, win_rows, self.TILE_SIZE):
            r1 = min(r0 + self.TILE_SIZE, win_rows)
            for c0 in range(0, win_cols, self.TILE_SIZE):
                if feedback.isCanceled():
                    out_ds = None
                    dem_ds = None
                    return {self.OUTPUT: output_path}

                c1 = min(c0 + self.TILE_SIZE, win_cols)
                dem_tile = dem_window[r0:r1, c0:c1]
                nodata_tile = nodata_mask[r0:r1, c0:c1]

                rows_idx = np.arange(r0, r1, dtype=np.float64) + row_min
                cols_idx = np.arange(c0, c1, dtype=np.float64) + col_min
                cc, rr = np.meshgrid(cols_idx, rows_idx)

                x = gt[0] + (cc + 0.5) * gt[1] + (rr + 0.5) * gt[2]
                y = gt[3] + (cc + 0.5) * gt[4] + (rr + 0.5) * gt[5]

                z_r = dem_tile + h_rec
                lpa_tile = compute_lpa_from_sources_grid(
                    x_grid=x,
                    y_grid=y,
                    z_rec=z_r,
                    sources=valid_sources,
                    alpha_atm=alpha_atm,
                    enable_ground=enable_ground,
                    g_value=g_value,
                    d_min=d_min,
                    nodata_mask=nodata_tile,
                )

                out_tile = np.full((r1 - r0, c1 - c0), self.OUTPUT_NODATA, dtype=np.float32)
                calc_mask = ~np.isnan(lpa_tile)
                out_tile[calc_mask] = lpa_tile[calc_mask].astype(np.float32)

                out_band.WriteArray(out_tile, xoff=c0, yoff=r0)

                done += 1
                feedback.setProgress(int(100.0 * done / total_tiles))

        out_band.FlushCache()
        out_ds = None
        dem_ds = None

        return {self.OUTPUT: output_path}

    @staticmethod
    def _is_numeric_field(field: QgsField):
        return field.isNumeric()

    @staticmethod
    def _is_nodata_value(v, nodata):
        if nodata is None:
            return np.isnan(v)
        return np.isclose(v, nodata)

    @staticmethod
    def _build_nodata_mask(arr, nodata):
        if nodata is None:
            return np.isnan(arr)
        return np.isclose(arr, nodata)

    @staticmethod
    def _xy_to_colrow(x, y, gt):
        det = gt[1] * gt[5] - gt[2] * gt[4]
        if det == 0:
            raise QgsProcessingException("GeoTransform DEM non invertibile.")
        dx = x - gt[0]
        dy = y - gt[3]
        col = int(math.floor((gt[5] * dx - gt[2] * dy) / det))
        row = int(math.floor((-gt[4] * dx + gt[1] * dy) / det))
        return col, row

    @staticmethod
    def _compute_domain_bbox(sources, baf):
        from qgis.core import QgsRectangle

        xmin = min(s[1] for s in sources) - baf
        xmax = max(s[1] for s in sources) + baf
        ymin = min(s[2] for s in sources) - baf
        ymax = max(s[2] for s in sources) + baf
        return QgsRectangle(xmin, ymin, xmax, ymax)

    @staticmethod
    def _extent_to_window(extent, gt, max_cols, max_rows):
        x_min = extent.xMinimum()
        x_max = extent.xMaximum()
        y_min = extent.yMinimum()
        y_max = extent.yMaximum()

        c0, r0 = ISO9613LpaRasterAlgorithm._xy_to_colrow(x_min, y_max, gt)
        c1, r1 = ISO9613LpaRasterAlgorithm._xy_to_colrow(x_max, y_min, gt)

        col_min = max(0, min(c0, c1))
        col_max = min(max_cols - 1, max(c0, c1))
        row_min = max(0, min(r0, r1))
        row_max = min(max_rows - 1, max(r0, r1))
        return col_min, col_max, row_min, row_max


from qgis.core import QgsWkbTypes  # noqa: E402
