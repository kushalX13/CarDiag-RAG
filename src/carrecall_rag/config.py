"""Paths and NHTSA API URLs."""

import os

NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
NHTSA_RECALLS_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_COMPLAINTS_DIR = os.path.join(RAW_DIR, "complaints")
RAW_RECALLS_DIR = os.path.join(RAW_DIR, "recalls")
RAW_RECALLS_GLOBAL_DIR = os.path.join(RAW_DIR, "recalls_global")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS = [
    {"make": "HONDA", "model": "CIVIC", "year_start": 2012, "year_end": 2018},
    {"make": "TOYOTA", "model": "CAMRY", "year_start": 2012, "year_end": 2018},
    {"make": "FORD", "model": "F-150", "year_start": 2013, "year_end": 2019},
    {"make": "JEEP", "model": "GRAND CHEROKEE", "year_start": 2014, "year_end": 2020},
    {"make": "BMW", "model": "3 SERIES", "year_start": 2012, "year_end": 2018},
    {"make": "MERCEDES-BENZ", "model": "C-CLASS", "year_start": 2012, "year_end": 2018},
]
