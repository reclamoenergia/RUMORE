# -*- coding: utf-8 -*-

from qgis.core import QgsProcessingProvider

from .iso9613_algorithm import ISO9613LpaRasterAlgorithm


class ISO9613LpaRasterProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(ISO9613LpaRasterAlgorithm())

    def id(self):
        return "iso9613_lpa_raster"

    def name(self):
        return "ISO9613 LpA Raster"

    def longName(self):
        return self.name()
