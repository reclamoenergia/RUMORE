# -*- coding: utf-8 -*-

from qgis.core import QgsProcessingProvider

from .iso9613_algorithm import ISO9613LpaRasterAlgorithm
from .iso9613_receptors_algorithm import ISO9613LpaReceptorsAlgorithm


class ISO9613LpaRasterProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(ISO9613LpaRasterAlgorithm())
        self.addAlgorithm(ISO9613LpaReceptorsAlgorithm())

    def id(self):
        return "iso9613_lpa_raster"

    def name(self):
        return "ISO9613 LpA Raster"

    def longName(self):
        return self.name()
