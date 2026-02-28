# -*- coding: utf-8 -*-

import math

import numpy as np

from ..core.iso9613_core import compute_lpa_for_receptors_points
from osgeo import gdal

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProject,
    QgsRectangle,
    QgsUnitTypes,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


class ISO9613LpaReceptorsAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    SOURCES = "SOURCES"
    FIELD_NAME = "FIELD_NAME"
    FIELD_HSRC = "FIELD_HSRC"
    FIELD_LWA = "FIELD_LWA"
    RECEPTORS = "RECEPTORS"
    H_REC = "H_REC"
    BAF = "BAF"
    ALPHA_ATM = "ALPHA_ATM"
    ENABLE_GROUND = "ENABLE_GROUND"
    G = "G"
    D_MIN = "D_MIN"
    OUTPUT = "OUTPUT"

    def name(self):
        return "iso9613_lpa_receptors"

    def displayName(self):
        return "ISO9613 LpA Receptors (v1)"

    def group(self):
        return "Acoustics"

    def groupId(self):
        return "acoustics"

    def shortHelpString(self):
        return "Calcola LpA (dB) sui ricettori puntuali con output vettoriale e campo LpA_dB."

    def createInstance(self):
        return ISO9613LpaReceptorsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.DEM, "DEM (elevazione, metri)"))
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
            QgsProcessingParameterFeatureSource(
                self.RECEPTORS,
                "Ricettori puntuali",
                [QgsProcessing.TypeVectorPoint],
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
                "BAF filtro ricettori (m, opzionale)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output ricettori con LpA_dB"))

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        src_source = self.parameterAsSource(parameters, self.SOURCES, context)
        rec_source = self.parameterAsSource(parameters, self.RECEPTORS, context)
        field_name = self.parameterAsString(parameters, self.FIELD_NAME, context)
        field_hsrc = self.parameterAsString(parameters, self.FIELD_HSRC, context)
        field_lwa = self.parameterAsString(parameters, self.FIELD_LWA, context)
        h_rec = self.parameterAsDouble(parameters, self.H_REC, context)
        alpha_atm = self.parameterAsDouble(parameters, self.ALPHA_ATM, context)
        enable_ground = self.parameterAsBool(parameters, self.ENABLE_GROUND, context)
        g_value = self.parameterAsDouble(parameters, self.G, context)
        d_min = self.parameterAsDouble(parameters, self.D_MIN, context)

        baf_value = None
        if self.BAF in parameters and parameters[self.BAF] not in (None, ""):
            baf_value = self.parameterAsDouble(parameters, self.BAF, context)
            if baf_value <= 0:
                baf_value = None

        if dem_layer is None:
            raise QgsProcessingException("DEM non valido.")
        if src_source is None:
            raise QgsProcessingException("Layer sorgenti non valido.")
        if rec_source is None:
            raise QgsProcessingException("Layer ricettori non valido.")

        if dem_layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
            raise QgsProcessingException("Il CRS del DEM deve avere unità in metri.")
        if QgsWkbTypes.geometryType(src_source.wkbType()) != QgsWkbTypes.PointGeometry:
            raise QgsProcessingException("Il layer sorgenti deve essere Point.")
        if QgsWkbTypes.geometryType(rec_source.wkbType()) != QgsWkbTypes.PointGeometry:
            raise QgsProcessingException("Il layer ricettori deve essere Point.")

        fields = src_source.fields()
        h_field = fields.lookupField(field_hsrc)
        lwa_field = fields.lookupField(field_lwa)
        if h_field < 0 or lwa_field < 0:
            raise QgsProcessingException("Campi h_src/LwA non trovati.")
        if not fields[h_field].isNumeric() or not fields[lwa_field].isNumeric():
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

        transform_src = None
        if src_source.sourceCrs().isValid() and src_source.sourceCrs() != dem_layer.crs():
            transform_src = QgsCoordinateTransform(src_source.sourceCrs(), dem_layer.crs(), QgsProject.instance())

        sources = []
        for feat in src_source.getFeatures():
            if feedback.isCanceled():
                return {self.OUTPUT: None}
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if transform_src is not None:
                geom = QgsGeometry(geom)
                geom.transform(transform_src)
            pt = geom.asPoint()
            name = feat[field_name] if field_name and field_name in fields.names() else f"src_{feat.id()}"
            try:
                h_src = float(feat[field_hsrc])
                lwa = float(feat[field_lwa])
            except (TypeError, ValueError):
                feedback.pushWarning(f"Sorgente {name}: h_src/LwA non numerici, scartata.")
                continue

            z_dem = self._sample_dem_nearest(dem_band, gt, pt.x(), pt.y())
            if z_dem is None or self._is_nodata_value(z_dem, dem_nodata):
                feedback.pushWarning(f"Sorgente {name} su NoData DEM/fuori raster, scartata.")
                continue
            sources.append((name, pt.x(), pt.y(), z_dem + h_src, lwa))

        if not sources:
            raise QgsProcessingException("Nessuna sorgente valida trovata.")

        domain = None
        if baf_value is not None:
            domain = self._compute_domain_bbox(sources, baf_value)

        rec_fields = QgsFields(rec_source.fields())
        rec_fields.append(QgsField("LpA_dB", QVariant.Double))

        output_crs = rec_source.sourceCrs() if rec_source.sourceCrs().isValid() else dem_layer.crs()
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            rec_fields,
            rec_source.wkbType(),
            output_crs,
        )
        if sink is None:
            raise QgsProcessingException("Impossibile creare output ricettori.")

        transform_rec = None
        if rec_source.sourceCrs().isValid() and rec_source.sourceCrs() != dem_layer.crs():
            transform_rec = QgsCoordinateTransform(rec_source.sourceCrs(), dem_layer.crs(), QgsProject.instance())

        total_rec = rec_source.featureCount() or 0
        done = 0
        nodata_receptors = 0
        skipped_by_baf = 0


        for rec_feat in rec_source.getFeatures():
            if feedback.isCanceled():
                break

            rec_geom = rec_feat.geometry()
            if rec_geom is None or rec_geom.isEmpty():
                continue

            calc_geom = QgsGeometry(rec_geom)
            if transform_rec is not None:
                calc_geom.transform(transform_rec)
            rec_pt = calc_geom.asPoint()

            if domain is not None and not domain.contains(rec_pt):
                skipped_by_baf += 1
                done += 1
                feedback.setProgress(int(100.0 * done / max(1, total_rec)))
                continue

            z_rec_dem = self._sample_dem_nearest(dem_band, gt, rec_pt.x(), rec_pt.y())
            lpa_db = None
            if z_rec_dem is None or self._is_nodata_value(z_rec_dem, dem_nodata):
                nodata_receptors += 1
                feedback.pushWarning(f"Ricettore {rec_feat.id()} su DEM NoData/fuori raster: LpA_dB = NULL")
            else:
                rec_xy = np.array([[rec_pt.x(), rec_pt.y()]], dtype=np.float64)
                rec_z = np.array([z_rec_dem + h_rec], dtype=np.float64)
                lpa_vals = compute_lpa_for_receptors_points(
                    rec_xy=rec_xy,
                    rec_z=rec_z,
                    sources=sources,
                    alpha_atm=alpha_atm,
                    enable_ground=enable_ground,
                    g_value=g_value,
                    d_min=d_min,
                )
                if np.isfinite(lpa_vals[0]):
                    lpa_db = float(lpa_vals[0])

            out_feat = QgsFeature(rec_fields)
            out_feat.setGeometry(rec_feat.geometry())
            attrs = rec_feat.attributes()
            attrs.append(lpa_db)
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            done += 1
            feedback.setProgress(int(100.0 * done / max(1, total_rec)))

        feedback.pushInfo(f"Sorgenti usate: {len(sources)}")
        feedback.pushInfo(f"Ricettori totali: {total_rec}")
        feedback.pushInfo(f"Ricettori filtrati da BAF: {skipped_by_baf}")
        feedback.pushInfo(f"Ricettori con LpA_dB NULL (NoData): {nodata_receptors}")

        dem_ds = None
        return {self.OUTPUT: dest_id}

    @staticmethod
    def _compute_domain_bbox(sources, baf):
        xmin = min(s[1] for s in sources) - baf
        xmax = max(s[1] for s in sources) + baf
        ymin = min(s[2] for s in sources) - baf
        ymax = max(s[2] for s in sources) + baf
        return QgsRectangle(xmin, ymin, xmax, ymax)

    @staticmethod
    def _sample_dem_nearest(dem_band, gt, x, y):
        col, row = ISO9613LpaReceptorsAlgorithm._xy_to_colrow(x, y, gt)
        if col < 0 or row < 0 or col >= dem_band.XSize or row >= dem_band.YSize:
            return None
        px = dem_band.ReadAsArray(col, row, 1, 1)
        if px is None:
            return None
        return float(px[0, 0])

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
    def _is_nodata_value(v, nodata):
        if nodata is None:
            return np.isnan(v)
        return np.isclose(v, nodata)
