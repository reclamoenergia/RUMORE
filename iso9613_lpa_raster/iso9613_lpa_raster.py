# -*- coding: utf-8 -*-

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsApplication

from .iso9613_lpa_raster_dialog import ISO9613LpaRasterDialog
from .processing.provider import ISO9613LpaRasterProvider


class ISO9613LpaRasterPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.provider = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "icon.png")
        self.action = QAction(QIcon(icon_path), "ISO9613 LpA Raster (v1)", self.iface.mainWindow())
        self.action.triggered.connect(self.run)

        self.iface.addPluginToMenu("&ISO9613 LpA Raster", self.action)
        self.iface.addToolBarIcon(self.action)

        self.provider = ISO9613LpaRasterProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("&ISO9613 LpA Raster", self.action)
            self.iface.removeToolBarIcon(self.action)
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

    def run(self):
        dlg = ISO9613LpaRasterDialog(self.iface)
        dlg.exec_()
