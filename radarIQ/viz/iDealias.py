from iPyart import iPyart
from ..proc import splashDealias
import pyart
import netCDF4

def iDealias(cfradDir, startFilenum):
    dealiasCorrectionVar = {
        "name": "dealias_correction",
        "fields": {
            "type": "i2",
            "fill_value": 0,
            "dims": ("time", "range"),
            "long_name": "velocity_dealias_correction_offset",
            "units": "2x nyqist velocity m/s",
            "grid_mapping": "grid_mapping",
            "coordinates": "time range",
            "data": "tbd"
        },
    }

    editHistory = []
    