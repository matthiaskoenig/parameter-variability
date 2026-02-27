"""Definitions for all models."""

from typing import Any

definitions_minimal: dict[str, dict[str, Any]] = {
    "timepoints": {
        "prior_types": ["prior_biased", "exact_prior"],
        "timepoints": [5, 11],
    },
}

definitions_all: dict[str, dict[str, Any]] = {
    "all": {
        "prior_types": ["no_prior", "prior_biased", "exact_prior"],
        "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
        "timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
    "prior_types": {
        "prior_types": ["no_prior", "prior_biased", "exact_prior"],
    },
    "samples": {
        "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80],
    },
    "timepoints": {
        "timepoints": [2, 3, 4, 5, 11, 21, 41, 81],
    },
    "cvs": {
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    },
}
