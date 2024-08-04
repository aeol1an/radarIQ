import numpy as np

def cart2polar(x: float, y: float):
    r = np.sqrt(x**2 + y**2)
    az = -np.arctan2(y, x) + (np.pi / 2)
    az = np.degrees(az)
    az = (az + 360) % 360

    return r, az

def polar2cart(r: float, az: float):
    x = r*np.cos(np.radians(-(az - 90)))
    y = r*np.sin(np.radians(-(az - 90)))

    return x, y