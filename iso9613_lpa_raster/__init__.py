# -*- coding: utf-8 -*-

def classFactory(iface):
    from .iso9613_lpa_raster import ISO9613LpaRasterPlugin
    return ISO9613LpaRasterPlugin(iface)
