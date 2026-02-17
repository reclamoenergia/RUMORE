from __future__ import annotations

import os
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.calc.iso9613 import calculate_grid_map
from app.io.exporters import export_geotiff, export_layout_png
from app.io.project_io import load_project, save_project
from app.model.entities import BAND_ORDER, GridSettings, ProjectData
from app.model.parsing import rows_to_sources
from app.ui.canvas2d import Canvas2D

EPSG_LIST = ["EPSG:32632", "EPSG:32633", "EPSG:32634", "EPSG:32632", "EPSG:32732", "EPSG:3857"]


class CalcWorker(QThread):
    progress = Signal(int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, sources, barriers, grid):
        super().__init__()
        self.sources = sources
        self.barriers = barriers
        self.grid = grid

    def run(self):
        try:
            result = calculate_grid_map(self.sources, self.barriers, self.grid, progress_cb=self.progress.emit)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RUMORE - ISO 9613 V1")
        self.sources = []
        self.barriers = []
        self.grid = GridSettings()
        self.last_result = None
        self._build_ui()
        self._create_menu()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        self.table = QTableWidget(0, 11)
        headers = ["ID", "Coord", "63Hz", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "8kHz", "LwA_tot"]
        self.table.setHorizontalHeaderLabels(headers)
        left.addWidget(QLabel("Sorgenti"))
        left.addWidget(self.table)

        btn_parse = QPushButton("Valida/Importa tabella")
        btn_parse.clicked.connect(self.parse_table)
        left.addWidget(btn_parse)

        form = QFormLayout()
        self.epsg_search = QLineEdit()
        self.epsg_combo = QComboBox()
        self.epsg_combo.addItems(EPSG_LIST)
        self.epsg_search.textChanged.connect(self.filter_epsg)
        self.cell_size = QLineEdit("10")
        self.buffer = QLineEdit("100")
        self.z_receiver = QLineEdit("4")
        self.nx = QLineEdit(); self.nx.setReadOnly(True)
        self.ny = QLineEdit(); self.ny.setReadOnly(True)
        form.addRow("Cerca EPSG", self.epsg_search)
        form.addRow("EPSG", self.epsg_combo)
        form.addRow("cell_size [m]", self.cell_size)
        form.addRow("buffer [m]", self.buffer)
        form.addRow("Z_ricevitore [m]", self.z_receiver)
        form.addRow("Nx", self.nx)
        form.addRow("Ny", self.ny)
        left.addLayout(form)

        self.cell_size.editingFinished.connect(self.update_grid_from_inputs)
        self.buffer.editingFinished.connect(self.update_grid_from_inputs)

        btns = QHBoxLayout()
        b_add_source = QPushButton("Aggiungi sorgente")
        b_add_source.clicked.connect(lambda: self.canvas.set_mode("add_source"))
        b_add_bar = QPushButton("Disegna barriera")
        b_add_bar.clicked.connect(lambda: self.canvas.set_mode("add_barrier"))
        b_calc = QPushButton("Calcola")
        b_calc.clicked.connect(self.run_calc)
        b_export = QPushButton("Export")
        b_export.clicked.connect(self.export_outputs)
        btns.addWidget(b_add_source); btns.addWidget(b_add_bar); btns.addWidget(b_calc); btns.addWidget(b_export)
        left.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4caf50, stop:1 #2196f3); }")
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        left.addWidget(self.progress)
        left.addWidget(QLabel("Log"))
        left.addWidget(self.log)

        right = QVBoxLayout()
        self.canvas = Canvas2D()
        right.addWidget(self.canvas)

        layout.addLayout(left, 2)
        layout.addLayout(right, 3)

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("File")
        save_action = QAction("Salva progetto", self)
        load_action = QAction("Carica progetto", self)
        save_action.triggered.connect(self.save_project)
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)

    def log_msg(self, msg: str):
        self.log.appendPlainText(msg)

    def filter_epsg(self, text: str):
        self.epsg_combo.clear()
        self.epsg_combo.addItems([e for e in EPSG_LIST if text.lower() in e.lower()])

    def parse_table(self):
        rows = []
        for r in range(self.table.rowCount()):
            rows.append([self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(self.table.columnCount())])
        sources, errors = rows_to_sources(rows)
        self.sources = sources
        self.barriers = self.canvas.barriers
        for e in errors:
            self.log_msg(e)
        if not errors:
            self.log_msg(f"Importate {len(self.sources)} sorgenti.")
        self.update_grid_from_inputs()
        self.canvas.set_data(self.sources, self.barriers, self.grid)

    def update_grid_from_inputs(self):
        try:
            self.grid.cell_size = float(self.cell_size.text())
            self.grid.buffer = float(self.buffer.text())
            self.grid.z_receiver = float(self.z_receiver.text())
        except ValueError:
            self.log_msg("Parametri griglia non validi.")
            return
        self.grid.update_extent(self.sources or self.canvas.sources)
        self.nx.setText(str(self.grid.nx))
        self.ny.setText(str(self.grid.ny))
        self.canvas.set_data(self.sources or self.canvas.sources, self.canvas.barriers, self.grid)

    def run_calc(self):
        if not (self.sources or self.canvas.sources):
            self.log_msg("Nessuna sorgente definita.")
            return
        self.sources = self.sources or self.canvas.sources
        self.barriers = self.canvas.barriers
        self.update_grid_from_inputs()
        self.worker = CalcWorker(self.sources, self.barriers, self.grid)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.completed.connect(self.on_calc_done)
        self.worker.failed.connect(lambda e: self.log_msg(f"Errore calcolo: {e}"))
        self.worker.start()
        self.log_msg("Calcolo avviato...")

    def on_calc_done(self, result):
        self.last_result = result
        self.canvas.redraw(contours=result.contours)
        self.log_msg("Calcolo completato.")

    def export_outputs(self):
        if self.last_result is None:
            self.log_msg("Eseguire prima il calcolo.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Seleziona cartella output")
        if not out_dir:
            return
        epsg = self.epsg_combo.currentText() or "EPSG:32632"
        tif = os.path.join(out_dir, "rumore_dba.tif")
        png = os.path.join(out_dir, "rumore_layout.png")
        export_geotiff(tif, self.last_result, self.grid, epsg)
        export_layout_png(png, self.last_result, self.grid, epsg, self.sources, self.barriers)
        self.log_msg(f"Export completato: {tif}, {png}")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salva progetto", filter="JSON (*.json)")
        if not path:
            return
        p = ProjectData(epsg=self.epsg_combo.currentText(), grid=self.grid, sources=self.sources or self.canvas.sources, barriers=self.canvas.barriers)
        save_project(path, p)
        self.log_msg(f"Progetto salvato: {path}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Carica progetto", filter="JSON (*.json)")
        if not path:
            return
        p = load_project(path)
        self.grid = p.grid
        self.sources = p.sources
        self.barriers = p.barriers
        self.canvas.set_data(self.sources, self.barriers, self.grid)
        self.nx.setText(str(self.grid.nx)); self.ny.setText(str(self.grid.ny))
        self.log_msg(f"Progetto caricato: {path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
