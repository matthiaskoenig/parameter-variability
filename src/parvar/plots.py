"""Plotting helpers."""

import matplotlib
import matplotlib.pyplot as plt

DPI = 300

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

__all__ = ["plt"]

colors = {
    "MALE": "tab:blue",
    "FEMALE": "tab:red",
}
