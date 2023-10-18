import pandas as pd
import roadrunner
from sbmlutils.console import console

from matplotlib import pyplot as plt
import numpy as np


r: roadrunner.RoadRunner = roadrunner.RoadRunner("model2.xml")

ks = np.linspace(0, 10, num=11)
cls = np.linspace(0, 10, num=11)
qs = np.linspace(0, 10, num=11)

results = {}

for parameter, par_name in zip([ks, cls, qs], ['k', 'CL', 'Q']):
    results_par = []
    for value in parameter:
        # reset to a clean state
        r.resetAll()
        r.setValue(par_name, value)
        s = r.simulate(start=0, end=10, steps=400)
        # pretty slow (memory copy)
        df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
        # console.print(df)
        results_par.append(df)

    results[par_name] = results_par


f, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 13))
for parameter, ax in zip(results, axes):
    for sid in ["[y_cent]", "[y_gut]", "[y_peri]"]:
        for df in results[parameter]:
            ax.plot(
                    df.time, df[sid], label=sid,
                    linestyle="-",
                    # marker="o",
                    markeredgecolor="black"
                )
    # ax.legend()
    ax.set_title(parameter)
    ax.set_xlabel("time [A.U.]")
    ax.set_ylabel("concentration [substance_units/compartment_units]")
plt.tight_layout()
plt.show()
