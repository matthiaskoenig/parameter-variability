"""Plotting helpers."""

import matplotlib
import matplotlib.pyplot as plt

DPI = 360

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 25

matplotlib.rc("font", size=SMALL_SIZE)  # controls default text sizes
matplotlib.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
matplotlib.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
matplotlib.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
matplotlib.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = {
    "MALE": "tab:blue",
    "FEMALE": "tab:red",
}

# Latex Labels
parameter_labels = {
    "k1": r"$k_1$",
    "CL": r"$CL$",
    "k_abs": r"$k$",
    "BW": r"$Bw$",
    "LI__ICGIM_Vmax": r"$V_{max}$",
}

# Value Labels
value_labels = {
    "prior_type": {
        "prior_biased_1": "Biased 1",
        "prior_biased_2": "Biased 2",
        "exact_prior": "Exact",
        "prior_biased": "Biased",
        "prior_incorrect": "Incorrect",
    }
}

# Axis Labels
axis_labels = {
    "prior_type": "Prior Types",
    "samples": "Samples",
    "timepoints": "Timepoints",
    "noise_cv": "Coefficient of variation",
}

__all__ = [
    "plt",
    "colors",
    "parameter_labels",
    "value_labels",
    "axis_labels",
]
