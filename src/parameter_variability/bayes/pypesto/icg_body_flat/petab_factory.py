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
from parameter_variability.bayes.pypesto.experiment import Group, DistributionType
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

class PKPDParameters(str, Enum):
    """ICG Parameters"""
    LI__ICGIM_Vmax = "LI__ICGIM_Vmax"
    BW = "BW"


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
    dosage: Optional[dict[str, float]] # TODO: Change to dosage and add skip error column

def create_samples_parameters(
    parameters: dict[Category, dict[PKPDParameters, LognormParameters]],
    seed: Optional[int] = 1234
) -> dict[Category, dict[PKPDParameters, np.ndarray]]:
    """Create the male and female samples."""
    if seed is not None:
        np.random.seed(seed)

    # samples
    samples: dict[Category, dict[PKPDParameters, np.ndarray]] = {}
    for category, pars in parameters.items():
        sample_par: dict[PKPDParameters, np.ndarray] = {}
        for par, lnpars in pars.items():
            s = lnpars.sigma
            m = lnpars.mu

            sigma_ln = np.sqrt(np.log(1 + (s / m) ** 2))
            mu_ln = np.log(m) - sigma_ln ** 2 / 2
            sample_par[par] = np.random.lognormal(mu_ln, sigma_ln, size=lnpars.n)
        samples[category] = sample_par

    return samples


def plot_samples(
    samples: dict[Category, dict[PKPDParameters, np.ndarray]],
    fig_path: Optional[Path]
) -> None:
    n_rows = np.max(np.array([len(v) for k, v in samples.items()]))
    # plot distributions
    f, axs = plt.subplots(n_rows, dpi=300, layout="constrained")

    for category, parameters in samples.items():
        for ax, (par, data) in zip(axs, parameters.items()):
            ax.hist(
                data, density=True, bins='auto', histtype='stepfilled', alpha=0.5,
                color=colors[category], label=f"{category.name}-{par.name} (n={len(data)})"
            )
    for category, data in samples.items():
        for ax, (par, data) in zip(axs, parameters.items()):
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
    if fig_path:
        plt.savefig(str(fig_path))
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
        self.ids: List[str] = self.r.getIds()


    def simulate_samples(self, parameters: pd.DataFrame, simulation_settings: SimulationSettings) -> xr.Dataset:
        """Simulate samples with roadrunner."""
        n = parameters.shape[0]
        pids = parameters.columns
        dfs = []
        for _, row in parameters.iterrows():
            self.r.resetAll()

            if simulation_settings.dosage:
                for key, value in simulation_settings.dosage.items():
                    self.r.setValue(key, value)

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
    """Plot simulations which were used for the PETab problem."""
    vars = dsets[next(iter(dsets))].data_vars
    # plot distributions
    f, axs = plt.subplots(nrows=len(vars), ncols=1, figsize=(4, 4*len(vars)),
                          dpi=200, layout="constrained")

    alpha = 0.5
    for category, dset in dsets.items():
        color = colors[category]
        nsim = len(dset["sim"])
        for k in range(nsim):
            t = dset["time"]
            Nt = len(t)
            kwargs = dict(
                alpha=alpha,
                color=color,
                marker='o',
                markeredgecolor="black",
                label=f"{category.name} (n={nsim}, Nt={Nt})" if k == 0 else "__nolabel__",
            )
            for var, ax in zip(vars, axs):
                ax.plot(t, dset[var].isel(sim=k), **kwargs)
                ax.set_ylabel(var)
                ax.set_xlabel("time")
                ax.legend()

    if fig_path is not None:
        plt.show()
        f.savefig(fig_path, bbox_inches="tight")

def add_errors(df_sim: pd.Dataframe,
               seed: int,
               skip: Optional[List[str]]
               # dsn_type: DistributionType
               ) -> pd.Dataframe:
    if skip:
        df_sim = df_sim.drop(skip, axis=1)

    np.random.normal(0, 1, df_sim.shape)

    pass


def create_petab_example(dfs: dict[Category, xarray.Dataset],
                         groups: List[Group],
                         petab_path: Path,
                         param: Union[str, List[str]],
                         sbml_path: Path,
                         initial_values: Optional[dict[str, int]] = None) -> Path:
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

        data_names = []

        for name in list(data.data_vars):
            if name[0] == '[':
                data_names.append(name[1:-1])
            else:
                data_names.append(name)

        for col in data_names:
            if initial_values:
                try:
                    condition_ls[-1].update({col: initial_values[col]})
                except KeyError:
                    continue
            else:
                condition_ls[-1].update({col: 0.0})

        for sim in sim_df['sim'].values:
            df_s = sim_df.isel(sim=sim).to_dataframe().reset_index()

            # TODO: Add errors in df_s
            #   df_s = add_errors(df_s)

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

                if col in df_s.columns:
                    col_brackets = col
                else:
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

    parameters = param

    for par in parameters:
        if par in param:
            for cat, group in zip(dfs.keys(), groups):
                # define parameters
                p = {
                    'parameterId': f'{par}_{cat.name}',
                    'parameterName': f'{par}_{cat.name}',
                    # 'parameterScale': 'log10',
                    'parameterScale': 'lin',
                    'lowerBound': 0.0,
                    'upperBound': 1E8,
                    'nominalValue': 1,
                    'estimate': 1,
                    'parameterUnit': 'l/min',
                }
                # objective priors
                if group.estimation.parameters:
                    p['objectivePriorType'] = 'parameterScaleNormal'

                    mean = group.get_parameter('estimation', par, 'loc')
                    std = group.get_parameter('estimation', par, 'scale')
                    p['objectivePriorParameters'] = f"{mean};{std}"

                parameter_ls.append(p)
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

    measurement_df.to_csv(petab_path / f"measurements_{sbml_path.stem}.tsv",
                          sep="\t", index=False)

    condition_df.to_csv(petab_path / f"conditions_{sbml_path.stem}.tsv",
                        sep="\t", index=False)

    parameter_df.to_csv(petab_path / f"parameters_{sbml_path.stem}.tsv",
                        sep='\t', index=False)

    observable_df.to_csv(petab_path / f"observables_{sbml_path.stem}.tsv",
                         sep='\t', index=False)

    # Create Petab YAML
    petab_path_rel = petab_path.relative_to(petab_path.parents[0])

    petab_yaml: dict[str, Optional[str, List[dict[str, List]]]] = {}
    petab_yaml['format_version'] = 1
    petab_yaml['parameter_file'] = str(petab_path_rel / f"parameters_{sbml_path.stem}.tsv")

    # copy model
    shutil.copy(sbml_path, petab_path / sbml_path.name)
    petab_yaml['problems'] = [{
        'condition_files': [str(petab_path_rel / f"conditions_{sbml_path.stem}.tsv")],
        'measurement_files': [str(petab_path_rel / f"measurements_{sbml_path.stem}.tsv")],
        'observable_files': [str(petab_path_rel / f"observables_{sbml_path.stem}.tsv")],
        'sbml_files': [str(petab_path_rel / sbml_path.name)],
    }]

    yaml_dest = petab_path.parents[0] / 'petab.yaml'
    with open(yaml_dest, 'w') as outfile:
        yaml.dump(petab_yaml, outfile, default_flow_style=False)

    return yaml_dest


# if __name__ == '__main__':
#     from parameter_variability import MODEL_SIMPLE_CHAIN
#
#     fig_path: Path = Path(__file__).parent / "results"
#     fig_path.mkdir(parents=True, exist_ok=True)
#
#     petab_path = fig_path / "petab"
#     petab_path.mkdir(parents=True, exist_ok=True)
#     sbml_path = Path("../../../models/sbml/simple_chain.xml")
#
#
#
#     pid: str = "k1"
#     prior_par = {f'{pid}_MALE': [1.0, 0.2], f'{pid}_FEMALE': [10.0, 0.2]}
#
#     # samples
#     samples_k1: dict[Category, np.ndarray] = create_male_female_samples(
#         {
#             Category.MALE: LognormParameters(mu=prior_par["k1_MALE"][0], sigma=prior_par["k1_MALE"][1], n=100),
#             Category.FEMALE: LognormParameters(mu=prior_par["k1_FEMALE"][0], sigma=prior_par["k1_FEMALE"][1], n=100),
#         }
#     )
#     plot_samples(samples_k1)
#     plt.savefig(str(fig_path) + '/01-plot_samples.png')
#
#     # simulations
#     simulator = ODESampleSimulator(model_path=MODEL_SIMPLE_CHAIN)
#     sim_settings = SimulationSettings(start=0.0, end=20.0, steps=300)
#     dsets: dict[Category, xarray.Dataset] = {}
#
#
#     for category, data in samples_k1.items():
#         # simulate samples for category
#         parameters = pd.DataFrame({pid: data})
#         dset = simulator.simulate_samples(parameters, simulation_settings=sim_settings)
#         dsets[category] = dset
#
#     plot_simulations(dsets)
#     plt.savefig(str(fig_path) + '/02-plot_simulations.png')
#
#     prior_par = {'k1_MALE': [0.0, 10.0], 'k1_FEMALE': [0.0, 10.0]}
#
#
#     create_petab_example(petab_path, dsets, param='k1',
#                          initial_values={'S1': 1, 'S2': 0},
#                          prior_par=prior_par,
#                          sbml_path=sbml_path)

