from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
import roadrunner
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal, Covariance, gaussian_kde
from matplotlib import pyplot as plt

from parameter_variability import BAYES_DIR, MEASUREMENT_TIME_UNIT_COLUMN, MEASUREMENT_UNIT_COLUMN
from parameter_variability.console import console

FIG_PATH: Path = Path(__file__).parent / "results"

_LOG_2PI = np.log(2 * np.pi)

# Create class for distributions


class BivariateLogNormal:
    """
    Bivariate log normal distribution

    Attributes
    ----------
    parameter_names:
        names of the variables for each dimension
    mean:
        array of mean values
    cov:
        covariate matrix
    """
    def __init__(self,
                 parameter_names: List[str],
                 mean: np.array,
                 cov: Covariance
                 ) -> None:
        self.parameter_names = parameter_names
        self.mean = mean
        self.cov = cov

    def rvs(self,
            size: int,
            seed: Optional[int]
            ) -> Dict[str, np.array]:
        """
        Sample from the defined distribution

        Parameters
        ----------
        size:
            number of samples
        seed:
            defined for reproducibility

        Returns
        -------
        result:
            array with random sample

        """

        if seed:
            np.random.seed(seed)

        multi_norm = multivariate_normal(self.mean, self.cov)
        sample = np.exp(multi_norm.rvs(size=size))

        result = {}
        for j, par in enumerate(self.parameter_names):
            result[par] = sample[:, j]

        return result

    def logpdf(self,
               x: np.array
               ) -> np.array:
        """
        Logarithm of the original probability density function
        Parameters
        ----------
        x:
            array with values in [0,1]

        Returns
        -------
            probability density values in the log scale
        """

        log_det_cov, rank = self.cov.log_pdet, self.cov.rank
        dev = np.log(x) - self.mean
        if dev.ndim > 1:
            log_det_cov = log_det_cov[..., np.newaxis]
            rank = rank[..., np.newaxis]
        maha = np.sum(np.square(self.cov.whiten(dev)) + 2 * np.log(x), axis=-1)

        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    def pdf(self,
            x: np.array
            ) -> np.array:
        """
        Probability Density Function on the original scale

        Parameters
        ----------
        x:
            array with values in [0,1]

        Returns
        -------
            probability density values

        """
        return np.exp(self.logpdf(x))

    @staticmethod
    def plot_distributions(dsns,
                           samples: List[Dict[str, np.array]],
                           log_scale: bool = True) -> None:
        """
        Plot the Bivariate Log Normal Distribution(s) with their samples

        Parameters
        ----------
        dsns:
            list of distributions to plot with logpdf or pdf method
        samples:
            array of samples from the distribution
        log_scale:
            determines the axes of the samples and the pdf to draw the contours from

        Returns
        -------
            matplotlib plot
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        colors = ["tab:blue", "tab:red"]
        cmaps = ["Blues", "Reds"]

        def calculate_limits(samples):
            return xlims, ylims

        def plot_samples():
            for k, samples_data in enumerate(samples):
                df = pd.DataFrame.from_dict(samples_data)
                ax.plot(
                    df['kabs'], df['CL'],
                    'o',
                    color=colors[k],
                    markeredgecolor='k',
                    markersize=4,
                    alpha=0.8
                )

                # Gaussian KDE
                # xmin = df['kabs'].min()
                # xmax = df['kabs'].max()
                # ymin = df['CL'].min()
                # ymax = df['CL'].max()
                # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                # positions = np.vstack([X.ravel(), Y.ravel()])
                # values = np.vstack([df['kabs'], df['CL']])
                # kernel = gaussian_kde(values)
                # Z = np.reshape(kernel(positions).T, X.shape)
                # ax.imshow(np.rot90(Z), cmap=cmaps[k], extent=[xmin, xmax, ymin, ymax])

        plot_samples()

        # plot pdf
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        for k, dsn in enumerate(dsns):
            xvec = np.linspace(start=xlims[0], stop=xlims[1], num=100)
            yvec = np.linspace(start=ylims[0], stop=ylims[1], num=100)
            x, y = np.meshgrid(xvec, yvec)

            xy = np.dstack((x, y))
            z = dsn.pdf(xy)

            cs = ax.contour(
                x, y, z,
                colors=colors[k],
                # cmap=cmaps[k],
                levels=20, alpha=1.0
            )

        # plot_samples()
        scale = "log"
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlabel('kabs')
        ax.set_ylabel('CL')


# Create class for simulation

class ODESimulation:
    """
    Create simulation using the samples from a multivariate normal distribution

    Parameters
    ----------
    model_path:
        SBML-based model path location
    samples:
        sample from a multivariate normal distribution with their names
    pop_vars:
        name of the variable and its levels
    compartment_starting_values:
        initial values to start the simulations
    parameters_var:
        parameters in the SBML model to

    """

    def __init__(self,
                 model_path: Path,
                 samples: List[Dict[str, np.array]],
                 pop_vars: Dict[str, List[str]],
                 compartment_starting_values: Dict[str, int],
                 parameters_var: Union[List[str], str]
                 ):
        self.model_path = model_path
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))

        integrator: roadrunner.Integrator = self.r.integrator
        integrator.setSetting("absolute_tolerance", 1e-6)
        integrator.setSetting("relative_tolerance", 1e-6)

        self.samples = samples
        self.pop_vars = pop_vars
        self.compartment_starting_values = compartment_starting_values

        if isinstance(parameters_var, str):
            parameters_var = [parameters_var]
        self.parameters_var = parameters_var

    def sim(self,
            sim_start: int = 0,
            sim_end: int = 10,
            sim_steps: int = 100,
            **kwargs) -> xr.Dataset:
        """
        Run the Roadrunner simulation

        Parameters
        ----------
        sim_start:
            timestamp to start the simulation
        sim_end:
            timestamp to end the simulaiton
        sim_steps:
            number of samples to get from the simulation
        kwargs:
            Extra parameters for the Roadrunner simulation

        Returns
        -------
        result:
            simulation output

        """

        dsets: List[xr.Dataset] = []

        for sample in self.samples:

            sample = pd.DataFrame.from_dict(sample)
            dfs = []

            for _, row in sample.iterrows():
                for par_name, par in zip(row.index, row):
                    self.r.setValue(par_name, par)

                s = self.r.simulate(start=sim_start,
                                    end=sim_end,
                                    steps=sim_steps,
                                    **kwargs)
                df = pd.DataFrame(s, columns=s.colnames).set_index("time")
                dfs.append(df)

            dset = xr.concat([df.to_xarray() for df in dfs],
                             dim=pd.Index(np.arange(sample.shape[0]), name='sim'))

            dsets.append(dset)

        result = xr.concat(dsets, dim=pd.Index(list(self.pop_vars.values())[0],
                                               name=list(self.pop_vars.keys())[0]))

        return result

    def to_petab(self,
                 sim_dfs: xr.Dataset) -> None:
        """
        From dataframe to PETAB format

        Parameters
        ----------
        sim_dfs:
            dataframe input


        """

        measurement_ls: List[pd.DataFrame] = []
        condition_ls: List[Dict[str, Optional[str, float, int]]] = []
        parameter_ls: List[Dict[str, Optional[str, float, int]]] = []
        observable_ls: List[Dict[str, Optional[str, float, int]]] = []

        for j, gen in enumerate(sim_dfs['gender'].values):

            measurement_pop: List[pd.DataFrame] = []

            sim_df = sim_dfs.sel(gender=gen)

            condition_ls.append({
                'conditionId': gen,
                'conditionName': '',
                'kabs': f'kabs_{gen}'

            })

            for par in self.parameters_var:
                condition_ls[-1].update({par: f'{par}_{gen}'})

            for col in ['y_gut', 'y_cent', 'y_peri']:
                condition_ls[-1].update({col: self.compartment_starting_values[col]})

            for sim in sim_df['sim'].values:
                df_s = sim_df.isel(sim=sim).to_dataframe().reset_index()
                unique_measurement = []

                for col in ['y_gut', 'y_cent', 'y_peri']:
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
                            "simulationConditionId": gen,  # f"model{j}_data{sim}",
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

        parameters: List[str] = list(self.samples[0].keys())

        for par in parameters:
            if par in self.parameters_var:
                for gen in sim_dfs['gender'].values:
                    parameter_ls.append({
                        'parameterId': f'{par}_{gen}',
                        'parameterName': f'{par}_{gen}',
                        'parameterScale': 'log10',
                        'lowerBound': 0.01,
                        'upperBound': 100,
                        'nominalValue': 1,
                        'estimate': 1,
                        'parameterUnit': 'l/min'
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

        measurement_df.to_csv(self.model_path.parent / "measurements_multi_pk.tsv",
                              sep="\t", index=False)

        condition_df.to_csv(self.model_path.parent / "conditions_multi_pk.tsv",
                            sep="\t", index=False)

        parameter_df.to_csv(self.model_path.parent / "parameters_multi_pk.tsv",
                            sep='\t', index=False)

        observable_df.to_csv(self.model_path.parent / "observables_multi_pk.tsv",
                             sep='\t', index=False)


if __name__ == "__main__":
    seed = None  # 1234
    n_samples = 100

    # sampling from distribution
    parameter_names = ['kabs', 'CL']

    # men
    mu_male = np.log(np.array([0.1, 0.5]))  # mean in normal space
    cov_male = Covariance.from_diagonal([1, 1])
    dsn_male = BivariateLogNormal(mean=mu_male, cov=cov_male,
                                  parameter_names=parameter_names)
    samples_male = dsn_male.rvs(n_samples, seed=seed)
    console.rule("male", style="white")

    # women
    mu_female = np.log(np.array([0.01, 0.5]))  # mean in normal space
    # mu_female = np.log(np.array([3, 3]))  # mean in normal space
    cov_female = Covariance.from_diagonal([1, 1])
    dsn_female = BivariateLogNormal(mean=mu_female, cov=cov_female,
                                    parameter_names=parameter_names)
    samples_female = dsn_female.rvs(n_samples, seed=seed)
    console.rule("female", style="white")

    # plot distributions
    BivariateLogNormal.plot_distributions(
        dsns=[dsn_male, dsn_female],
        samples=[samples_male, samples_female],
    )
    plt.savefig(str(FIG_PATH) + '/00_dsn.png')
    plt.show()


    # simulation
    MODEL_PATH: Path = Path(__file__).parent / "simple_pk.xml"
    compartment_starting_values = {'y_gut': 1, 'y_cent': 0, 'y_peri': 0}
    ode_sim = ODESimulation(model_path=MODEL_PATH,
                            samples=[samples_male, samples_female],
                            pop_vars={'gender': ['male', 'female']},
                            compartment_starting_values=compartment_starting_values,
                            parameters_var='kabs')
    synth_dset: xr.Dataset = ode_sim.sim()
    console.print(synth_dset)

    # convert to PeTab problem
    ode_sim.to_petab(synth_dset)


    # 1. define distributions (multi-var log normal)
    # 2. calculate a PDF from that (PDF calculation) whatever analytical, complicated

    # 3. take samples
    # 4. do a kernel estimate of the samples => pdf (

