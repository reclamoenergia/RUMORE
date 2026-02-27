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
    QgsMapLayerProxyModel,
    QgsWkbTypes,
    QgsFieldProxyModel,
)
from qgis.gui import QgsFieldComboBox, QgsMapLayerComboBox


class ISO9613LpaRasterDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("ISO9613 LpA Raster (v1)")
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

        self.h_rec = self._make_spin(0.0, 1000.0, 4.0)
        layout.addRow("h_rec (m)", self.h_rec)

        self.baf = self._make_spin(0.0, 100000.0, 1000.0)
        layout.addRow("BAF (m)", self.baf)

        self.alpha = self._make_spin(0.0, 1.0, 0.0, 6)
        layout.addRow("alpha_atm (dB/m)", self.alpha)

        self.d_min = self._make_spin(0.001, 1000.0, 1.0)
        layout.addRow("d_min (m)", self.d_min)

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

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Esegui")
        self.buttons.accepted.connect(self._run)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)

        self._on_source_changed(self.src_combo.currentLayer())

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

    def _pick_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva output GeoTIFF",
            "",
            "GeoTIFF (*.tif *.tiff)",
        )
        if path:
            self.output_edit.setText(path)

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

        if not out_path:
            QMessageBox.critical(self, "Errore", "Seleziona un file output.")
            return

        params = {
            "DEM": dem,
            "SOURCES": src,
            "FIELD_NAME": None,
            "FIELD_HSRC": self.hsrc_field.currentField(),
            "FIELD_LWA": self.lwa_field.currentField(),
            "H_REC": self.h_rec.value(),
            "BAF": self.baf.value(),
            "ALPHA_ATM": self.alpha.value(),
            "ENABLE_GROUND": self.enable_ground.isChecked(),
            "G": self.g_value.value(),
            "D_MIN": self.d_min.value(),
            "OUTPUT": out_path,
        }

        try:
            processing.run("iso9613_lpa_raster:iso9613_lpa_raster_v1", params)
            QMessageBox.information(self, "Completato", "Calcolo completato con successo.")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc))
