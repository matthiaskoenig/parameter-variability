"""Definitions for all models."""

from typing import Any

definitions_minimal: dict[str, dict[str, Any]] = {
    "timepoints": {
        "prior_types": ["exact_prior", "prior_biased", "prior_incorrect"],
        "timepoints": [5, 10],
    },
}

definitions_all: dict[str, dict[str, Any]] = {
    "all": {
        "prior_types": ["exact_prior", "prior_biased", "prior_incorrect"],
        "samples": [1, 2, 3, 4, 5, 10, 20, 40, 80, 160],  # n=10
        "timepoints": [2, 3, 4, 5, 10, 20, 40, 80, 160],  # n=9
        "noise_cvs": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],  # n=8
    }
}
