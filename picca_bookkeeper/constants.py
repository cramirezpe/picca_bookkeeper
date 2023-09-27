"""Defining constant values to be used in other modules"""
from picca.constants import ABSORBER_IGM

forest_regions = {
    "lya": {
        "lambda-rest-min": 1040.0,
        "lambda-rest-max": 1200.0,
    },
    "lyb": {
        "lambda-rest-min": 920.0,
        "lambda-rest-max": 1020.0,
    },
    "mgii_r": {
        "lambda-rest-min": 2900.0,
        "lambda-rest-max": 3120.0,
    },
    "ciii": {
        "lambda-rest-min": 1600.0,
        "lambda-rest-max": 1850.0,
    },
    "civ": {
        "lambda-rest-min": 1410.0,
        "lambda-rest-max": 1520.0,
    },
    "siv": {
        "lambda-rest-min": 1260.0,
        "lambda-rest-max": 1375.0,
    },
    "SiII": {
        "lambda-rest-min": 1225.0,
        "lambda-rest-max": 1255.0,
    },
    "mgii_11": {
        "lambda-rest-min": 2600.0,
        "lambda-rest-max": 2760.0,
    },
    "mgii_h": {
        "lambda-rest-min": 2100.0,
        "lambda-rest-max": 2760.0,
    },
}

# Get absorbers in lowercase.
# For historical reasons I'm using lowercase for lya and lyb.
absorber_igm = dict(
    (absorber, absorber) for absorber in ABSORBER_IGM if absorber not in ("LYA", "LYB")
)
absorber_igm["lya"] = "LYA"
absorber_igm["lyb"] = "LYB"
