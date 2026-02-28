# -*- coding: utf-8 -*-

from qgis import processing
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
)
from qgis.core import (
    QgsFeature,
    QgsFeatureRequest,
    QgsFieldProxyModel,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsFieldComboBox, QgsMapLayerComboBox


class ISO9613LpaRasterDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("ISO9613 LpA Raster (v2)")
        self.setMinimumWidth(520)
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)

        self.dem_combo = QgsMapLayerComboBox()
        self.dem_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout.addRow("DEM", self.dem_combo)

        self.src_combo = QgsMapLayerComboBox()
        self.src_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        layout.addRow("Sorgenti", self.src_combo)

        self.hsrc_field = QgsFieldComboBox()
        self.hsrc_field.setFilters(QgsFieldProxyModel.Numeric)
        layout.addRow("Campo h_src", self.hsrc_field)

        self.lwa_field = QgsFieldComboBox()
        self.lwa_field.setFilters(QgsFieldProxyModel.Numeric)
        layout.addRow("Campo LwA", self.lwa_field)

        self.src_combo.layerChanged.connect(self._on_source_changed)

        self.chk_use_selected_sources = QCheckBox("Usa solo sorgenti selezionate")
        self.chk_use_selected_sources.setChecked(False)
        layout.addRow("Selezione sorgenti", self.chk_use_selected_sources)

        self.h_rec = self._make_spin(0.0, 1000.0, 4.0)
        layout.addRow("h_rec (m)", self.h_rec)

        self.baf = self._make_spin(0.0, 100000.0, 1000.0)
        layout.addRow("BAF (m)", self.baf)

        self.alpha = self._make_spin(0.0, 1.0, 0.0, 6)
        layout.addRow("alpha_atm (dB/m)", self.alpha)

        self.d_min = self._make_spin(0.001, 1000.0, 1.0)
        layout.addRow("d_min (m)", self.d_min)

        self.use_bands = QCheckBox("Usa bande d’ottava")
        self.use_bands.setChecked(False)
        layout.addRow("Bande", self.use_bands)

        self.spectrum_combo = QComboBox()
        self.spectrum_combo.addItems(["Flat", "Aerogeneratore standard", "Campi per banda (se disponibili)", "Offset personalizzati"])
        layout.addRow("Spettro", self.spectrum_combo)

        self.wind_bin = self._make_spin(4, 14, 10, 0)
        layout.addRow("Wind bin", self.wind_bin)

        self.offsets_edit = QLineEdit()
        self.offsets_edit.setPlaceholderText("63:-3;125:-2;250:0;500:1;1000:2;2000:1;4000:-1;8000:-2")
        self.offsets_edit.setEnabled(False)
        layout.addRow("Offsets", self.offsets_edit)

        self.spectrum_combo.currentIndexChanged.connect(self._on_spectrum_mode_changed)

        self.enable_ground = QCheckBox("Abilita suolo semplificato")
        layout.addRow("Suolo", self.enable_ground)

        self.g_value = self._make_spin(0.0, 1.0, 0.5)
        self.g_value.setEnabled(False)
        self.enable_ground.toggled.connect(self.g_value.setEnabled)
        layout.addRow("G (0..1)", self.g_value)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        out_btn = QPushButton("Sfoglia…")
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(out_btn)
        layout.addRow("Output GeoTIFF", out_row)

        self.chk_add_raster_to_project = QCheckBox("Aggiungi output raster al progetto")
        self.chk_add_raster_to_project.setChecked(True)
        layout.addRow("Import raster", self.chk_add_raster_to_project)

        self.chk_run_receptors = QCheckBox("Esegui anche calcolo su ricettori")
        self.chk_run_receptors.setChecked(False)
        layout.addRow("Ricettori", self.chk_run_receptors)

        self.receptors_combo = QgsMapLayerComboBox()
        self.receptors_combo.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.receptors_combo.setEnabled(False)
        self.chk_run_receptors.toggled.connect(self.receptors_combo.setEnabled)
        layout.addRow("Layer ricettori", self.receptors_combo)

        rec_out_row = QHBoxLayout()
        self.receptors_output_edit = QLineEdit()
        self.receptors_output_edit.setEnabled(False)
        self.rec_out_btn = QPushButton("Sfoglia…")
        self.rec_out_btn.setEnabled(False)
        self.chk_run_receptors.toggled.connect(self.receptors_output_edit.setEnabled)
        self.chk_run_receptors.toggled.connect(self.rec_out_btn.setEnabled)
        self.rec_out_btn.clicked.connect(self._pick_receptors_output)
        rec_out_row.addWidget(self.receptors_output_edit)
        rec_out_row.addWidget(self.rec_out_btn)
        layout.addRow("Output ricettori", rec_out_row)

        self.chk_add_receptors_to_project = QCheckBox("Aggiungi output ricettori al progetto")
        self.chk_add_receptors_to_project.setChecked(True)
        self.chk_add_receptors_to_project.setEnabled(False)
        self.chk_run_receptors.toggled.connect(self.chk_add_receptors_to_project.setEnabled)
        layout.addRow("Import ricettori", self.chk_add_receptors_to_project)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Esegui")
        self.buttons.accepted.connect(self._run)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)

        self._on_source_changed(self.src_combo.currentLayer())
        self._on_spectrum_mode_changed(self.spectrum_combo.currentIndex())

    @staticmethod
    def _make_spin(min_v, max_v, default_v, decimals=3):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(min_v, max_v)
        spin.setValue(default_v)
        spin.setSingleStep(0.1)
        return spin

    def _on_source_changed(self, layer):
        self.hsrc_field.setLayer(layer)
        self.lwa_field.setLayer(layer)


    def _on_spectrum_mode_changed(self, idx):
        self.wind_bin.setEnabled(idx == 1)
        self.offsets_edit.setEnabled(idx == 3)

    def _pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva output GeoTIFF",
            "",
            "GeoTIFF (*.tif *.tiff)",
        )
        if path:
            self.output_edit.setText(path)

    def _pick_receptors_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva output ricettori",
            "",
            "GeoPackage (*.gpkg);;ESRI Shapefile (*.shp)",
        )
        if path:
            self.receptors_output_edit.setText(path)

    def _selected_sources_layer(self, src):
        if not self.chk_use_selected_sources.isChecked():
            return src

        if not isinstance(src, QgsVectorLayer):
            QMessageBox.warning(self, "Selezione sorgenti", "Layer sorgenti non selezionabile: uso tutte le sorgenti.")
            return src

        total = src.featureCount()
        selected_ids = src.selectedFeatureIds()
        if not selected_ids:
            QMessageBox.warning(self, "Selezione sorgenti", "Nessuna sorgente selezionata → uso tutte.")
            return src

        crs_authid = src.crs().authid() or "EPSG:4326"
        memory_uri = f"Point?crs={crs_authid}"
        selected_layer = QgsVectorLayer(memory_uri, "selected_sources", "memory")
        dp = selected_layer.dataProvider()
        dp.addAttributes(src.fields())
        selected_layer.updateFields()

        request = QgsFeatureRequest().setFilterFids(selected_ids)
        selected_feats = []
        for feat in src.getFeatures(request):
            new_feat = QgsFeature(selected_layer.fields())
            new_feat.setGeometry(feat.geometry())
            new_feat.setAttributes(feat.attributes())
            selected_feats.append(new_feat)

        dp.addFeatures(selected_feats)
        selected_layer.updateExtents()

        QMessageBox.information(
            self,
            "Selezione sorgenti",
            f"Uso {len(selected_feats)} sorgenti selezionate su {total} totali.",
        )
        return selected_layer

    def _run(self):
        dem = self.dem_combo.currentLayer()
        src = self.src_combo.currentLayer()
        out_path = self.output_edit.text().strip()

        if dem is None or src is None:
            QMessageBox.critical(self, "Errore", "Seleziona DEM e layer sorgenti.")
            return

        if QgsWkbTypes.geometryType(src.wkbType()) != QgsWkbTypes.PointGeometry:
            QMessageBox.critical(self, "Errore", "Il layer sorgenti deve essere Point.")
            return

        field_hsrc = self.hsrc_field.currentField().strip()
        field_lwa = self.lwa_field.currentField().strip()
        if not field_hsrc or not field_lwa:
            QMessageBox.critical(self, "Errore", "Seleziona i campi h_src e LwA prima di eseguire.")
            return

        if not out_path:
            QMessageBox.critical(self, "Errore", "Seleziona un file output raster.")
            return

        src_to_use = self._selected_sources_layer(src)

        params = {
            "DEM": dem,
            "SOURCES": src_to_use,
            "FIELD_NAME": None,
            "FIELD_HSRC": field_hsrc,
            "FIELD_LWA": field_lwa,
            "H_REC": self.h_rec.value(),
            "BAF": self.baf.value(),
            "ALPHA_ATM": self.alpha.value(),
            "ENABLE_GROUND": self.enable_ground.isChecked(),
            "G": self.g_value.value(),
            "D_MIN": self.d_min.value(),
            "USE_OCTAVE_BANDS": self.use_bands.isChecked(),
            "SPECTRUM_MODE": self.spectrum_combo.currentIndex(),
            "WIND_BIN": int(self.wind_bin.value()),
            "OFFSETS": self.offsets_edit.text().strip(),
            "OUTPUT": out_path,
        }

        try:
            raster_result = processing.run("iso9613_lpa_raster:iso9613_lpa_raster_v1", params)
            raster_out_path = raster_result.get("OUTPUT", out_path)

            if self.chk_add_raster_to_project.isChecked():
                rl = QgsRasterLayer(raster_out_path, "LpA ISO9613")
                if rl.isValid():
                    QgsProject.instance().addMapLayer(rl)
                else:
                    QMessageBox.warning(self, "Import raster", "Output raster creato ma layer non valido.")

            if self.chk_run_receptors.isChecked():
                receptors_layer = self.receptors_combo.currentLayer()
                receptors_output = self.receptors_output_edit.text().strip()
                if receptors_layer is None:
                    QMessageBox.critical(self, "Errore", "Seleziona il layer ricettori.")
                    return
                if not receptors_output:
                    QMessageBox.critical(self, "Errore", "Seleziona un file output per i ricettori.")
                    return

                receptors_params = {
                    "DEM": dem,
                    "SOURCES": src_to_use,
                    "FIELD_NAME": None,
                    "FIELD_HSRC": field_hsrc,
                    "FIELD_LWA": field_lwa,
                    "RECEPTORS": receptors_layer,
                    "H_REC": self.h_rec.value(),
                    "BAF": self.baf.value(),
                    "ALPHA_ATM": self.alpha.value(),
                    "ENABLE_GROUND": self.enable_ground.isChecked(),
                    "G": self.g_value.value(),
                    "D_MIN": self.d_min.value(),
                    "USE_OCTAVE_BANDS": self.use_bands.isChecked(),
                    "SPECTRUM_MODE": self.spectrum_combo.currentIndex(),
                    "WIND_BIN": int(self.wind_bin.value()),
                    "OFFSETS": self.offsets_edit.text().strip(),
                    "OUTPUT": receptors_output,
                }
                receptors_result = processing.run("iso9613_lpa_raster:iso9613_lpa_receptors", receptors_params)
                receptors_out_path = receptors_result.get("OUTPUT", receptors_output)

                if self.chk_add_receptors_to_project.isChecked():
                    vl = QgsVectorLayer(receptors_out_path, "LpA Receptors", "ogr")
                    if vl.isValid():
                        QgsProject.instance().addMapLayer(vl)
                    else:
                        QMessageBox.warning(self, "Import ricettori", "Output ricettori creato ma layer non valido.")

            QMessageBox.information(self, "Completato", "Calcolo completato con successo.")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc))
