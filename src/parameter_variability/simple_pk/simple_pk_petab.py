from pathlib import Path
from matplotlib import pyplot as plt
import roadrunner
import pandas as pd
import numpy as np
from dataclasses import dataclass

from parameter_variability import BAYES_DIR, MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN
from parameter_variability.console import console


def example_simulation(model_path: Path, fig_folder: Path, **kwargs) -> pd.DataFrame:
    np.random.seed(1234)
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    console.print(f"True k: {r.getValue('kabs')}, True CL: {r.getValue('CL')}")
    sim = r.simulate(start=0, end=20, steps=100, **kwargs)
    df = pd.DataFrame(sim, columns=sim.colnames)

    console.print(df)

    f, ax = plt.subplots()
    cols = df.drop('time', axis=1).columns.tolist()

    # Error adding
    mu = 0
    sigma = 0.01

    for col in cols:
        ax.plot(df['time'], df[col], alpha=0.7, color="black")

        col_wo_brackets = col[1:-1]
        df[f"{col_wo_brackets}_data"] = (df[f"{col}"] +
                                         np.random.normal(mu, sigma, len(df)))
        df[f"{col_wo_brackets}_data"] = (df[f"{col}"] +
                                         np.random.normal(mu, sigma, len(df)))

        plot_kwargs = {
            "linestyle": "",
            "marker": "o",
            "markeredgecolor": "black",
        }

        ax.plot(df['time'], df[f"{col_wo_brackets}_data"],
                color="tab:blue", **plot_kwargs)

    ax.set_xlabel("time")
    ax.set_ylabel("concentration")

    plt.savefig(str(fig_path) + '/00_simulation.png')
    plt.show()

    return df

def create_petab_example(model_path: Path, fig_path: Path):
    df = example_simulation(model_path, fig_path)

    data = []
    for k, row in df.iterrows():
        for col in ['y_gut', 'y_cent', 'y_peri']:
            data.append({
                "observableId": f"{col}_observable",
                "preequilibrationConditionId": None,
                "simulationConditionId": "model1_data1",
                "measurement": 	row[f"{col}_data"],
                MEASUREMENT_UNIT_COLUMN: "mmole/l",
                "time": row["time"],
                MEASUREMENT_TIME_UNIT_COLUMN: "second",
                "observableParameters": None,
                "noiseParameters": None,
            })
    measurement_df = pd.DataFrame(data)
    measurement_df.to_csv(model_path.parent / "measurements_simple_pk.tsv", sep="\t", index=False)


if __name__ == '__main__':
    model_path: Path = Path(__file__).parent / "simple_pk.xml"
    fig_path: Path = Path(__file__).parent / "results"
    fig_path.mkdir(exist_ok=True)
    create_petab_example(model_path, fig_path)

    import petab

    yaml_path = BAYES_DIR / "pypesto" / "simple_pk" / "simple_pk.yaml"
    problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
    console.print(problem)
    errors_exist = petab.lint.lint_problem(problem)
    console.print(f"PEtab errors: {errors_exist}")










