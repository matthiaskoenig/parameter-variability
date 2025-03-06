from pathlib import Path
from typing import Optional, List, Union
import yaml
import shutil
import xarray
from matplotlib import pyplot as plt
import roadrunner
import pandas as pd
import numpy as np
from dataclasses import dataclass
import xarray as xr
from parameter_variability import BAYES_DIR, MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN
from parameter_variability.console import console
from parameter_variability.bayes.pypesto.simple_chain.experiment_factory import Group
from dataclasses import dataclass
from enum import Enum
from scipy.stats import lognorm


class Category(str, Enum):
    """Categories."""
    MALE = "male"
    FEMALE = "female"
    OLD = "old"
    YOUNG = "young"


colors = {
    Category.MALE: "tab:blue",
    Category.FEMALE: "tab:red",
    Category.OLD: "tab:orange",
    Category.YOUNG: "tab:green",
}

@dataclass
class LognormParameters:
    """Lognormal parameters"""
    n: int  # number of samples
    sigma: float  # standard deviation
    mu: float  # mean

@dataclass
class SimulationSettings:
    """Lognormal parameters"""
    start: float
    end: float
    steps: int

def create_male_female_samples(parameters: dict[Category, LognormParameters], seed: Optional[int] = 1234):
    """Create the male and female samples."""
    if seed is not None:
        np.random.seed(seed)

    # samples
    samples: dict[Category, np.ndarray] = {}
    for category, lnpars in parameters.items():
        s = lnpars.sigma
        m = lnpars.mu

        sigma_ln = np.sqrt(np.log(1 + (s / m) ** 2))
        mu_ln = np.log(m) - sigma_ln ** 2 / 2
        samples[category] =np.random.lognormal(mu_ln, sigma_ln, size=lnpars.n)

    return samples

def plot_samples(samples: dict[Category, np.ndarray]):
    console.print(samples_k1)

    # plot distributions
    f, ax = plt.subplots(dpi=300, layout="constrained")
    for category, data in samples.items():
        ax.hist(
            data, density=True, bins='auto', histtype='stepfilled', alpha=0.5,
            color=colors[category], label=f"hist {category.name}"
        )
    for category, data in samples.items():
        mean = np.mean(data)
        std = np.std(data)
        ax.axvline(x=mean, color=colors[category], label=f"mean {category.name}")
        ax.axvline(x=mean + std, color=colors[category], linestyle="--",
                   label=f"__nolabel__")
        ax.axvline(x=mean - std, color=colors[category], linestyle="--",
                   label=f"__nolabel__")

    ax.set_xlabel("parameter")
    ax.set_ylabel("density")
    ax.legend()
    # plt.show()

class ODESampleSimulator:
    """Performs simulations with given model and samples."""

    def __init__(self, model_path: Path, abs_tol: float = 1E-6, rel_tol: float = 1E-6):
        """Load model and integrator settings."""
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
        # console.print(self.r.getInfo())
        integrator: roadrunner.Integrator = self.r.integrator
        integrator.setSetting("absolute_tolerance", abs_tol)
        integrator.setSetting("relative_tolerance", rel_tol)


    def simulate_samples(self, parameters: pd.DataFrame, simulation_settings: SimulationSettings) -> xr.Dataset:
        """Simulate samples with roadrunner."""
        n = parameters.shape[0]
        pids = parameters.columns
        dfs = []
        for _, row in parameters.iterrows():
            self.r.resetAll()

            # set the parameter values
            for pid in pids:
                value = row[pid]
                # console.print(f"{pid}: {value}")
                self.r.setValue(pid, value)

            # simulate
            s = self.r.simulate(
                start=simulation_settings.start,
                end=simulation_settings.end,
                steps=simulation_settings.steps,
            )
            # convert result to data frame
            df = pd.DataFrame(s, columns=s.colnames).set_index("time")
            dfs.append(df)

        dset = xr.concat([df.to_xarray() for df in dfs], dim=pd.Index(np.arange(n), name='sim'))
        return dset


def plot_simulations(dsets: dict[Category, xarray.Dataset], fig_path: Optional[Path] = None):
    """Plot simulations."""
    # console.print(dsets)
    # console.print(list(dsets[list(dsets.keys())[0]].data_vars))

    # plot distributions
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=300, layout="constrained")

    alpha = 0.7
    for category, dset in dsets.items():
        color = colors[category]
        nsim = len(dset["sim"])
        for k in range(nsim):
            ax1.plot(
                dset["time"],
                dset["[S1]"].isel(sim=k),
                alpha=alpha,
                color=color
            )
            ax2.plot(
                dset["time"],
                dset["[S2]"].isel(sim=k),
                alpha=alpha,
                color=color
            )

    ax1.set_ylabel("[S1]")
    ax2.set_ylabel("[S2]")
    for ax in ax1, ax2:
        ax.set_xlabel("time")
    if fig_path is not None:
        plt.show()
        f.savefig(fig_path, bbox_inches="tight")


def create_petab_example(petab_path: Path, dfs: dict[Category, xarray.Dataset],
                         param: Union[str, List[str]], compartment_starting_values: dict[str, int],
                         groups: List[Group],
                         sbml_path: Path) -> Path:
    """Create PETab problem for given information.

    Returns path to petab yaml.
    """
    # ensure output folder exists
    petab_path.mkdir(parents=True, exist_ok=True)

    # Create all files and copy all the files
    measurement_ls: List[pd.DataFrame] = []
    condition_ls: List[dict[str, Optional[str, float, int]]] = []
    parameter_ls: List[dict[str, Optional[str, float, int]]] = []
    observable_ls: List[dict[str, Optional[str, float, int]]] = []

    if isinstance(param, str):
        param = [param]

    #for j, gen in enumerate(sim_dfs['gender'].values):
    for j, (cat, data) in enumerate(dfs.items()):

        measurement_pop: List[pd.DataFrame] = []

        sim_df = data

        condition_ls.append({
            'conditionId': cat.name,
            'conditionName': '',
            # 'k1': f'k1_{cat.name}'

        })

        for par in param:
            condition_ls[-1].update({par: f'{par}_{cat.name}'})

        data_names = [name[1:-1] for name in list(data.data_vars)]

        for col in data_names:
            condition_ls[-1].update({col: compartment_starting_values[col]})

        for sim in sim_df['sim'].values:
            df_s = sim_df.isel(sim=sim).to_dataframe().reset_index()
            unique_measurement = []

            for col in data_names:
                if sim == sim_df['sim'].values[0] and j == 0:
                    observable_ls.append({
                        'observableId': f'{col}_observable',
                        'observableFormula': col,
                        'observableName': col,
                        'noiseDistribution': 'normal',
                        'noiseFormula': 1,
                        'observableTransformation': 'lin',
                        'observableUnit': 'mmol/l'
                    })
                col_brackets = '[' + col + ']'
                for k, row in df_s.iterrows():
                    unique_measurement.append({
                        "observableId": f"{col}_observable",
                        "preequilibrationConditionId": None,
                        "simulationConditionId": cat.name,  # f"model{j}_data{sim}",
                        "measurement": row[col_brackets],  # !
                        MEASUREMENT_UNIT_COLUMN: "mmole/l",
                        "time": row["time"],  # !
                        MEASUREMENT_TIME_UNIT_COLUMN: "second",
                        "observableParameters": None,
                        "noiseParameters": None,
                    })

            measurement_sim_df = pd.DataFrame(unique_measurement)

            measurement_pop.append(measurement_sim_df)

        measurement_df = pd.concat(measurement_pop)
        measurement_ls.append(measurement_df)

    parameters: List[str] = ['k1']  # FIXME: add the SBML parameters
    console.print(parameters)
    for par in parameters:
        if par in param:
            for cat, group in zip(dfs.keys(), groups):
                parameter_ls.append({
                    'parameterId': f'{par}_{cat.name}',
                    'parameterName': f'{par}_{cat.name}',
                    # 'parameterScale': 'log10',
                    'parameterScale': 'lin',
                    'lowerBound': 0.01,
                    'upperBound': 100,
                    'nominalValue': 1,
                    'estimate': 1,
                    'parameterUnit': 'l/min',
                    'objectivePriorType': 'parameterScaleNormal',
                    'objectivePriorParameters': f"{group.get_parameter('estimation', par, 'loc')};"
                                                f"{group.get_parameter('estimation', par, 'scale')}"
                })

        else:
            parameter_ls.append({
                'parameterId': par,
                'parameterName': par,
                'parameterScale': 'log10',
                'lowerBound': 0.01,
                'upperBound': 100,
                'nominalValue': 1,
                'estimate': 1,
                'parameterUnit': 'l/min'
            })

    measurement_df = pd.concat(measurement_ls)
    condition_df = pd.DataFrame(condition_ls)
    parameter_df = pd.DataFrame(parameter_ls)
    observable_df = pd.DataFrame(observable_ls)

    measurement_df.to_csv(petab_path / "measurements_simple_chain.tsv",
                          sep="\t", index=False)

    condition_df.to_csv(petab_path / "conditions_simple_chain.tsv",
                        sep="\t", index=False)

    parameter_df.to_csv(petab_path / "parameters_simple_chain.tsv",
                        sep='\t', index=False)

    observable_df.to_csv(petab_path / "observables_simple_chain.tsv",
                         sep='\t', index=False)

    # Create Petab YAML
    petab_path_rel = petab_path.relative_to(petab_path.parents[0])

    petab_yaml: dict[str, Optional[str, List[dict[str, List]]]] = {}
    petab_yaml['format_version'] = 1
    petab_yaml['parameter_file'] = str(petab_path_rel / "parameters_simple_chain.tsv")

    # copy model
    shutil.copy(sbml_path, petab_path / sbml_path.name)
    petab_yaml['problems'] = [{
        'condition_files': [str(petab_path_rel / "conditions_simple_chain.tsv")],
        'measurement_files': [str(petab_path_rel / "measurements_simple_chain.tsv")],
        'observable_files': [str(petab_path_rel / "observables_simple_chain.tsv")],
        'sbml_files': [str(petab_path_rel / sbml_path.name)],
    }]

    yaml_dest = petab_path.parents[0] / 'petab.yaml'
    with open(yaml_dest, 'w') as outfile:
        yaml.dump(petab_yaml, outfile, default_flow_style=False)

    # # create the measurement table from given data !
    # df = example_simulation(model_path, fig_path)
    #
    # data = []
    # for k, row in df.iterrows():
    #     for col in ['y_gut', 'y_cent', 'y_peri']:
    #         data.append({
    #             "observableId": f"{col}_observable",
    #             "preequilibrationConditionId": None,
    #             "simulationConditionId": "model1_data1",
    #             "measurement": 	row[f"{col}_data"],
    #             MEASUREMENT_UNIT_COLUMN: "mmole/l",
    #             "time": row["time"],
    #             MEASUREMENT_TIME_UNIT_COLUMN: "second",
    #             "observableParameters": None,
    #             "noiseParameters": None,
    #         })
    # measurement_df = pd.DataFrame(data)
    # measurement_df.to_csv(model_path.parent / "measurements_simple_pk.tsv", sep="\t", index=False)
    #
    # model_path: Path = Path(__file__).parent / "simple_pk.xml"
    # fig_path: Path = Path(__file__).parent / "results"
    # fig_path.mkdir(exist_ok=True)
    # create_petab_example(model_path, fig_path)
    #
    # import petab
    #
    # yaml_path = BAYES_DIR / "pypesto" / "simple_pk" / "simple_pk.yaml"
    # problem: petab.Problem = petab.Problem.from_yaml(yaml_path)
    # console.print(problem)
    # errors_exist = petab.lint.lint_problem(problem)
    # console.print(f"PEtab errors: {errors_exist}")

    return yaml_dest


if __name__ == '__main__':
    from parameter_variability import MODEL_SIMPLE_CHAIN

    fig_path: Path = Path(__file__).parent / "results"
    fig_path.mkdir(parents=True, exist_ok=True)

    petab_path = fig_path / "petab"
    petab_path.mkdir(parents=True, exist_ok=True)
    sbml_path = Path("../../../models/sbml/simple_chain.xml")

    # prior_par = {'k1_MALE': [1.5, 1.0], 'k1_FEMALE': [3.0, 0.5]}
    prior_par = {'k1_MALE': [1.0, 0.2], 'k1_FEMALE': [10.0, 0.2]}

    # samples
    samples_k1: dict[Category, np.ndarray] = create_male_female_samples(
        {
            # Category.MALE: LognormParameters(mu=1.5, sigma=1.0, n=50),  # mu_ln=0.2216, sigma_ln=0.60640
            # Category.FEMALE: LognormParameters(mu=3.0, sigma=0.5, n=100),  # mu_ln=1.0849, sigma_ln=0.16552
            Category.MALE: LognormParameters(mu=prior_par["k1_MALE"][0], sigma=prior_par["k1_MALE"][1], n=100),
            Category.FEMALE: LognormParameters(mu=prior_par["k1_FEMALE"][0], sigma=prior_par["k1_FEMALE"][1], n=100),

            # Category.OLD: LognormParameters(mu=10.0, sigma=3, n=20),
            # Category.YOUNG: LognormParameters(mu=1.5, sigma=1, n=40),
        }
    )
    plot_samples(samples_k1)
    plt.savefig(str(fig_path) + '/01-plot_samples.png')

    # simulations
    simulator = ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
    sim_settings = SimulationSettings(start=0.0, end=20.0, steps=300)
    dsets: dict[Category, xarray.Dataset] = {}
    for category, data in samples_k1.items():
        # simulate samples for category
        parameters = pd.DataFrame({"k1": data})
        dset = simulator.simulate_samples(parameters, simulation_settings=sim_settings)
        dsets[category] = dset

    plot_simulations(dsets)
    plt.savefig(str(fig_path) + '/02-plot_simulations.png')

    prior_par = {'k1_MALE': [0.0, 10.0], 'k1_FEMALE': [0.0, 10.0]}
    create_petab_example(petab_path, dsets, param='k1',
                         compartment_starting_values={'S1': 1, 'S2': 0},
                         prior_par=prior_par,
                         sbml_path=sbml_path)

