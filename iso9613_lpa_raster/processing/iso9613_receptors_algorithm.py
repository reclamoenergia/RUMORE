# -*- coding: utf-8 -*-

import math

import numpy as np

from ..core.iso9613_core import (
    BANDS,
    A_WEIGHT_DB,
    alpha_iso9613_1,
    compute_adiv,
    compute_agr_iso9613_2_octave,
    compute_agr_simplified,
    dz_iso_single_screen,
    abar_from_dz,
    build_source_spectrum,
    compute_barrier_attenuation_broadband,
    compute_lpa_for_receptors_points,
)
from osgeo import gdal

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
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
    BARRIERS = "BARRIERS"
    BARRIER_HEIGHT_FIELD = "BARRIER_HEIGHT_FIELD"
    BARRIER_HEIGHT_DEFAULT = "BARRIER_HEIGHT_DEFAULT"
    H_REC = "H_REC"
    BAF = "BAF"
    ALPHA_ATM = "ALPHA_ATM"
    ENABLE_GROUND = "ENABLE_GROUND"
    G = "G"
    D_MIN = "D_MIN"
    USE_OCTAVE_BANDS = "USE_OCTAVE_BANDS"
    SPECTRUM_MODE = "SPECTRUM_MODE"
    OFFSETS = "OFFSETS"
    TEMPERATURE_C = "TEMPERATURE_C"
    RELATIVE_HUMIDITY = "RELATIVE_HUMIDITY"
    PRESSURE_KPA = "PRESSURE_KPA"
    OUTPUT = "OUTPUT"

    SPECTRUM_MODES = [
        "Flat (from LwA)",
        "Aerogeneratore (shape scaled to LwA)",
        "Use band fields (if provided)",
    ]

    FIELD_LW_MAP = {
        63: "FIELD_LW_63",
        125: "FIELD_LW_125",
        250: "FIELD_LW_250",
        500: "FIELD_LW_500",
        1000: "FIELD_LW_1000",
        2000: "FIELD_LW_2000",
        4000: "FIELD_LW_4000",
        8000: "FIELD_LW_8000",
    }

    def name(self):
        return "iso9613_lpa_receptors"

    def displayName(self):
        return "ISO9613 LpA Receptors (v2.2)"

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
        for freq, field_name in self.FIELD_LW_MAP.items():
            self.addParameter(
                QgsProcessingParameterField(
                    field_name,
                    f"Campo Lw {freq} Hz (opzionale)",
                    parentLayerParameterName=self.SOURCES,
                    type=QgsProcessingParameterField.Numeric,
                    optional=True,
                )
            )

        self.addParameter(QgsProcessingParameterFeatureSource(self.RECEPTORS, "Ricettori puntuali", [QgsProcessing.TypeVectorPoint]))
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.BARRIERS,
                "Barriere lineari (opzionale)",
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.BARRIER_HEIGHT_FIELD,
                "Campo altezza barriera (m, opzionale)",
                parentLayerParameterName=self.BARRIERS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BARRIER_HEIGHT_DEFAULT,
                "Altezza barriera default (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.0,
            )
        )
        self.addParameter(QgsProcessingParameterBoolean(self.USE_OCTAVE_BANDS, "USE_OCTAVE_BANDS", defaultValue=False))
        self.addParameter(QgsProcessingParameterEnum(self.SPECTRUM_MODE, "SPECTRUM_MODE", options=self.SPECTRUM_MODES, defaultValue=0))
        self.addParameter(QgsProcessingParameterString(self.OFFSETS, "OFFSETS (es. 63:-3;125:-2;...)", optional=True, defaultValue=""))

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
            QgsProcessingParameterNumber(
                self.TEMPERATURE_C,
                "Temperatura aria (°C)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RELATIVE_HUMIDITY,
                "Umidità relativa (%)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=70.0,
                minValue=0.1,
                maxValue=100.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PRESSURE_KPA,
                "Pressione atmosferica (kPa)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=101.325,
                minValue=10.0,
            )
        )
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_GROUND, "Abilita attenuazione suolo semplificata", defaultValue=False))
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
        use_bands = self.parameterAsBool(parameters, self.USE_OCTAVE_BANDS, context)
        spectrum_idx = self.parameterAsEnum(parameters, self.SPECTRUM_MODE, context)
        offsets = self._parse_offsets(self.parameterAsString(parameters, self.OFFSETS, context))
        h_rec = self.parameterAsDouble(parameters, self.H_REC, context)
        alpha_atm = self.parameterAsDouble(parameters, self.ALPHA_ATM, context)
        enable_ground = self.parameterAsBool(parameters, self.ENABLE_GROUND, context)
        g_value = self.parameterAsDouble(parameters, self.G, context)
        d_min = self.parameterAsDouble(parameters, self.D_MIN, context)
        temperature_c = self.parameterAsDouble(parameters, self.TEMPERATURE_C, context)
        relative_humidity = self.parameterAsDouble(parameters, self.RELATIVE_HUMIDITY, context)
        pressure_kpa = self.parameterAsDouble(parameters, self.PRESSURE_KPA, context)
        barriers_source = self.parameterAsSource(parameters, self.BARRIERS, context)
        barrier_height_field = self.parameterAsString(parameters, self.BARRIER_HEIGHT_FIELD, context)
        barrier_height_default = self.parameterAsDouble(parameters, self.BARRIER_HEIGHT_DEFAULT, context)

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

        barriers = self._load_barriers(
            barriers_source=barriers_source,
            barrier_height_field=barrier_height_field,
            barrier_height_default=barrier_height_default,
            target_crs=dem_layer.crs(),
            feedback=feedback,
        )

        fields = src_source.fields()
        h_field = fields.lookupField(field_hsrc)
        lwa_field = fields.lookupField(field_lwa)
        if h_field < 0 or lwa_field < 0:
            raise QgsProcessingException("Campi h_src/LwA non trovati.")
        if not fields[h_field].isNumeric() or not fields[lwa_field].isNumeric():
            raise QgsProcessingException("I campi h_src e LwA devono essere numerici.")

        band_field_names = {freq: self.parameterAsString(parameters, key, context) for freq, key in self.FIELD_LW_MAP.items()}
        has_complete_band_fields = all(name and fields.lookupField(name) >= 0 for name in band_field_names.values())
        mode_key = self._resolve_mode_key(spectrum_idx)
        if has_complete_band_fields and use_bands:
            mode_key = "BANDS_FROM_FIELDS"

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
        sources_spectra = []
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

            lw_band_fields = None
            if has_complete_band_fields:
                lw_band_fields = {}
                try:
                    for freq, f_name in band_field_names.items():
                        lw_band_fields[freq] = float(feat[f_name])
                except (TypeError, ValueError):
                    lw_band_fields = None

            z_dem = self._sample_dem_nearest(dem_band, gt, pt.x(), pt.y())
            if z_dem is None or self._is_nodata_value(z_dem, dem_nodata):
                feedback.pushWarning(f"Sorgente {name} su NoData DEM/fuori raster, scartata.")
                continue
            sources.append((name, pt.x(), pt.y(), z_dem + h_src, lwa))
            sources_spectra.append(
                {
                    "x": pt.x(),
                    "y": pt.y(),
                    "z": z_dem + h_src,
                    "h_src": h_src,
                    "lwa": lwa,
                    "lw_band": build_source_spectrum(lwa_total=lwa, mode=mode_key, lw_band_fields=lw_band_fields, user_offsets_db=offsets),
                }
            )

        if not sources:
            raise QgsProcessingException("Nessuna sorgente valida trovata.")

        if use_bands:
            feedback.pushInfo(
                f"Mode bande: preset={self.SPECTRUM_MODES[spectrum_idx]}, "
                f"T={temperature_c:.2f}°C, RH={relative_humidity:.1f}%, p={pressure_kpa:.3f} kPa"
            )
            feedback.pushInfo("alpha_atm manuale ignorato in modalità bande.")
            feedback.pushInfo(f"Ground effect: ISO9613-2 octave bands, G={g_value:.3f} ({'enabled' if enable_ground else 'disabled'})")

        domain = None
        if baf_value is not None:
            domain = self._compute_domain_bbox(sources, baf_value)

        rec_fields = QgsFields(rec_source.fields())
        rec_fields.append(QgsField("LpA_dB", QVariant.Double))

        output_crs = rec_source.sourceCrs() if rec_source.sourceCrs().isValid() else dem_layer.crs()
        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, rec_fields, rec_source.wkbType(), output_crs)
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
                if barriers:
                    lpa_db = self._compute_lpa_with_barriers(
                        rec_pt,
                        float(rec_z[0]),
                        sources_spectra,
                        dem_band,
                        gt,
                        dem_nodata,
                        barriers,
                        use_bands,
                        alpha_atm,
                        enable_ground,
                        g_value,
                        d_min,
                        temperature_c,
                        relative_humidity,
                        pressure_kpa,
                        h_rec,
                        feedback if done == 0 else None,
                    )
                else:
                    lpa_vals = compute_lpa_for_receptors_points(
                        rec_xy=rec_xy,
                        rec_z=rec_z,
                        sources=sources,
                        alpha_atm=alpha_atm,
                        enable_ground=enable_ground,
                        g_value=g_value,
                        d_min=d_min,
                        use_bands=use_bands,
                        sources_spectra=sources_spectra,
                        temperature_c=temperature_c,
                        relative_humidity=relative_humidity,
                        pressure_kpa=pressure_kpa,
                        receiver_height_m=h_rec,
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
    def _compute_lpa_with_barriers(
        rec_pt,
        rec_z,
        sources_spectra,
        dem_band,
        gt,
        dem_nodata,
        barriers,
        use_bands,
        alpha_atm,
        enable_ground,
        g_value,
        d_min,
        temperature_c,
        relative_humidity,
        pressure_kpa,
        h_rec,
        feedback=None,
    ):
        alpha_per_band = {freq: alpha_iso9613_1(freq, temperature_c, relative_humidity, pressure_kpa) for freq in BANDS}
        p_tot = 0.0
        intersections_total = 0
        barriers_applied = 0

        for src in sources_spectra:
            result = ISO9613LpaReceptorsAlgorithm._compute_source_contribution_with_barriers(
                src=src,
                rec_pt=rec_pt,
                rec_z=rec_z,
                dem_band=dem_band,
                gt=gt,
                dem_nodata=dem_nodata,
                barriers=barriers,
                use_bands=use_bands,
                alpha_atm=alpha_atm,
                alpha_per_band=alpha_per_band,
                enable_ground=enable_ground,
                g_value=g_value,
                d_min=d_min,
                h_rec=h_rec,
            )
            p_tot += result["power"]
            intersections_total += result["intersections"]
            if result["barrier_applied"]:
                barriers_applied += 1

        if feedback is not None:
            feedback.pushInfo(f"Barriere caricate: {len(barriers)}")
            feedback.pushInfo(f"Intersezioni sorgente-recettore/barriera valutate: {intersections_total}")
            feedback.pushInfo(f"Contributi sorgente con attenuazione barriera applicata: {barriers_applied}")
            if not use_bands:
                feedback.pushInfo("USE_OCTAVE_BANDS=False: attenuazione barriera broadband derivata da attenuazioni per banda.")

        if p_tot <= 0.0:
            return None
        return 10.0 * math.log10(p_tot)

    @staticmethod
    def _compute_source_contribution_with_barriers(
        src,
        rec_pt,
        rec_z,
        dem_band,
        gt,
        dem_nodata,
        barriers,
        use_bands,
        alpha_atm,
        alpha_per_band,
        enable_ground,
        g_value,
        d_min,
        h_rec,
    ):
        x_s = float(src["x"])
        y_s = float(src["y"])
        z_s = float(src["z"])
        dxy = math.hypot(rec_pt.x() - x_s, rec_pt.y() - y_s)
        dz_sr = rec_z - z_s
        d = max(d_min, math.sqrt(dxy * dxy + dz_sr * dz_sr))
        adiv = float(compute_adiv(d))

        best, intersection_count = ISO9613LpaReceptorsAlgorithm._best_barrier_profile(
            x_s, y_s, z_s, rec_pt.x(), rec_pt.y(), rec_z, d, dem_band, gt, dem_nodata, barriers
        )

        h_src = float(src.get("h_src", 1.0))
        abar_by_band = {freq: 0.0 for freq in BANDS}
        if best is not None and best["z"] > 0.0:
            for freq in BANDS:
                agr_band = compute_agr_iso9613_2_octave(freq, d, h_src, h_rec, g_value) if enable_ground else 0.0
                dz = dz_iso_single_screen(freq, best["dss"], best["dsr"], d, best["z"])
                abar_by_band[freq] = abar_from_dz(dz, agr_band)

        power = 0.0
        barrier_applied = False
        if use_bands:
            for freq in BANDS:
                aatm = alpha_per_band[freq] * d
                agr = compute_agr_iso9613_2_octave(freq, d, h_src, h_rec, g_value) if enable_ground else 0.0
                abar = abar_by_band[freq]
                if abar > 0.0:
                    barrier_applied = True
                lp_band = float(src["lw_band"][freq]) - (adiv + aatm + agr + abar)
                power += 10.0 ** ((lp_band + A_WEIGHT_DB[freq]) / 10.0)
        else:
            aatm = alpha_atm * d
            agr = compute_agr_simplified(enable_ground, g_value, d)
            abar_bb = compute_barrier_attenuation_broadband(src.get("lw_band"), abar_by_band)
            barrier_applied = abar_bb > 0.0
            lp = float(src["lwa"]) - (adiv + aatm + agr + abar_bb)
            power += 10.0 ** (lp / 10.0)

        return {"power": power, "intersections": intersection_count, "barrier_applied": barrier_applied}

    @staticmethod
    def _best_barrier_profile(x_s, y_s, z_s, x_r, y_r, z_r, d, dem_band, gt, dem_nodata, barriers):
        line = QgsGeometry.fromWkt(f"LINESTRING({x_s} {y_s}, {x_r} {y_r})")
        best = None
        intersections = 0
        for barrier in barriers:
            inter = line.intersection(barrier["geom"])
            if inter is None or inter.isEmpty():
                continue
            for point in ISO9613LpaReceptorsAlgorithm._extract_intersection_points(inter):
                intersections += 1
                z_ground = ISO9613LpaReceptorsAlgorithm._sample_dem_nearest(dem_band, gt, point.x(), point.y())
                if z_ground is None or ISO9613LpaReceptorsAlgorithm._is_nodata_value(z_ground, dem_nodata):
                    continue
                z_edge = z_ground + barrier["h"]
                dss = math.sqrt((point.x() - x_s) ** 2 + (point.y() - y_s) ** 2 + (z_edge - z_s) ** 2)
                dsr = math.sqrt((x_r - point.x()) ** 2 + (y_r - point.y()) ** 2 + (z_r - z_edge) ** 2)
                z_diff = max(0.0, dss + dsr - d)
                if best is None or z_diff > best["z"]:
                    best = {"dss": dss, "dsr": dsr, "z": z_diff, "h": barrier["h"]}
        return best, intersections

    @staticmethod
    def _extract_intersection_points(intersection_geom):
        points = []
        geom_type = QgsWkbTypes.geometryType(intersection_geom.wkbType())
        if geom_type == QgsWkbTypes.PointGeometry:
            if intersection_geom.isMultipart():
                points.extend(intersection_geom.asMultiPoint())
            else:
                points.append(intersection_geom.asPoint())
        elif geom_type == QgsWkbTypes.LineGeometry:
            lines = intersection_geom.asMultiPolyline() if intersection_geom.isMultipart() else [intersection_geom.asPolyline()]
            for line in lines:
                if not line:
                    continue
                points.append(line[0])
                points.append(line[-1])
        return points

    @staticmethod
    def _load_barriers(barriers_source, barrier_height_field, barrier_height_default, target_crs, feedback):
        barriers = []
        if barriers_source is None:
            return barriers
        if QgsWkbTypes.geometryType(barriers_source.wkbType()) != QgsWkbTypes.LineGeometry:
            raise QgsProcessingException("Il layer barriere deve essere LineString/MultiLineString.")

        barrier_transform = None
        if barriers_source.sourceCrs().isValid() and barriers_source.sourceCrs() != target_crs:
            barrier_transform = QgsCoordinateTransform(barriers_source.sourceCrs(), target_crs, QgsProject.instance())

        barrier_fields = barriers_source.fields()
        h_idx = barrier_fields.lookupField(barrier_height_field) if barrier_height_field else -1
        for feat in barriers_source.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if barrier_transform is not None:
                geom = QgsGeometry(geom)
                geom.transform(barrier_transform)
            if not geom.isGeosValid():
                geom = geom.makeValid()
            if geom is None or geom.isEmpty():
                continue

            h_bar = barrier_height_default
            if h_idx >= 0:
                raw_h = feat[h_idx]
                if raw_h in (None, ""):
                    feedback.pushWarning(f"Barriera {feat.id()}: altezza mancante, uso default {barrier_height_default} m")
                else:
                    try:
                        h_bar = float(raw_h)
                    except (TypeError, ValueError):
                        feedback.pushWarning(f"Barriera {feat.id()}: altezza non valida ({raw_h}), uso default {barrier_height_default} m")
                        h_bar = barrier_height_default
            barriers.append({"geom": geom, "h": max(0.0, h_bar)})

        feedback.pushInfo(f"Barriere lineari valide caricate: {len(barriers)}")
        return barriers

    @staticmethod
    def _resolve_mode_key(idx):
        if idx == 1:
            return "TURBINE_SHAPE_SCALED"
        if idx == 2:
            return "BANDS_FROM_FIELDS"
        return "FLAT_FROM_LWA"

    @staticmethod
    def _parse_offsets(offsets_text):
        if not offsets_text:
            return {}
        out = {}
        for chunk in offsets_text.split(";"):
            if not chunk.strip() or ":" not in chunk:
                continue
            f_txt, v_txt = chunk.split(":", 1)
            try:
                out[int(f_txt.strip())] = float(v_txt.strip())
            except ValueError:
                continue
        return out

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
