import pandas as pd
import roadrunner
from sbmlutils.console import console

from matplotlib import pyplot as plt
import numpy as np


r: roadrunner.RoadRunner = roadrunner.RoadRunner("model1.xml")

ks = np.linspace(0, 10, num=11)

results = []
for k1 in ks:
    # reset to a clean state
    r.resetAll()
    r.setValue("k1", k1)
    s = r.simulate(start=0, end=10, steps=400)
    # pretty slow (memory copy)
    df: pd.DataFrame = pd.DataFrame(s, columns=s.colnames)
    # console.print(df)
    results.append(df)


f, ax = plt.subplots(nrows=1, ncols=1)
for sid in ["[S1]", "[S2]"]:
    for df in results:
        ax.plot(
            df.time, df[sid], label=sid,
            linestyle="-",
            # marker="o",
            markeredgecolor="black"
        )
    # ax.legend()
ax.set_xlabel("time [A.U.]")
ax.set_ylabel("concentration [substance_units/compartment_units]")
plt.show()






