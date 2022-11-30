import math
import numpy as np


def get_random_bright_hsv(seed:int = 1):
    """ Generate random bright color in the HSV color space."""
    low_sat = 0.75
    low_val = 0.75
    rng = np.random.default_rng(seed=seed)
    
    hue = rng.integers(360, size=1)[0] # 0 <= h <= 360
    sat = rng.uniform(low_sat, 1.0, size=1)[0]
    val = rng.uniform(low_val, 1.0, size=1)[0]
    
    return (hue, sat, val)


def hsv2rgb(hsv_color:tuple = None):
    """ """
    hue = hsv_color[0]
    sat = hsv_color[1]
    val = hsv_color[2]
    
    chroma = sat * val
    H = hue / 60
    X = chroma * (1 - abs(H % 2 - 1))
    
    R = 0.0
    G = 0.0
    B = 0.0
    
    if 0 <= H < 1:
        R = chroma
        G = X
    elif 1 <= H < 2:
        R = X
        G = chroma
    elif 2 <= H < 3:
        G = chroma
        B = X
    elif 3 <= H < 4:
        G = X
        B = chroma
    elif 4 <= H < 5:
        R = X
        B = chroma
    elif 5 <= H < 6:
        R = chroma
        G = X
    
    m = val - chroma
    R = int((R + m) * 255)
    G = int((G + m) * 255)
    B = int((B + m) * 255)
    
    return (R, G, B)

