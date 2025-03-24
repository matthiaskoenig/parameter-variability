"""Factory to create the various experiments."""
from parameter_variability.bayes.pypesto.experiment import *
from pathlib import Path
from pydantic_yaml import parse_yaml_file_as, to_yaml_str
import yaml

if __name__ == "__main__":

    from pymetadata.console import console

    console.rule(style="white")
    console.print(PETabExperiment.model_json_schema())
    console.rule(style="white")

    # Define the true values of the parameters for distribution sampling
    true_par: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 1.0, "scale": 1})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 2.0, "scale": 1}))
    }

    true_sampling: dict[str, Sampling] = {
        'MALE': Sampling(
            n_samples=20,
            steps=100,
            parameters=[true_par['MALE']]
        ),
        'FEMALE': Sampling(
            n_samples=20,
            steps=100,
            parameters=[true_par['FEMALE']])
    }

    # Define parameters for each experiment
    pars_uninformative: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 0.0, "scale": 10})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 0.0, "scale": 10}))
    }

    pars_high_variance: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 0.0, "scale": 100})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 0.0, "scale": 100}))
    }

    pars_biased: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 10.0, "scale": 0.2})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 1.0, "scale": 0.2}))
    }

    exp_exact = PETabExperiment(
        id='exact',
        model='simple_chain',
        groups=[
            Group(
                id='MALE',
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[true_par['MALE']]
                )
            ),
            Group(
                id='FEMALE',
                sampling=true_sampling['FEMALE'],
                estimation=Estimation(
                    parameters=[true_par['FEMALE']]
                )
            )
        ]
    )

    # Set up PETabExperiment with parameters dicts
    exp_uninformative = PETabExperiment(
        id="uninformative",
        model="simple_chain",
        groups=[
            Group(
                id="MALE",
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[
                        pars_uninformative['MALE']
                    ],
                )
            ),
            Group(
                id="FEMALE",
                sampling=true_sampling['FEMALE'],
                estimation=Estimation(
                    parameters=[
                        pars_uninformative['FEMALE']
                    ],
                )
            ),
        ]
    )

    exp_high_variance = PETabExperiment(
        id="high_variance",
        model="simple_chain",
        groups=[
            Group(
                id="MALE",
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[
                        pars_high_variance['MALE']
                    ],
                )
            ),
            Group(
                id="FEMALE",
                sampling=true_sampling['FEMALE'],
                estimation=Estimation(
                    parameters=[
                        pars_high_variance['FEMALE']
                    ],
                )
            ),
        ]
    )

    exp_biased = PETabExperiment(
        id="biased",
        model="simple_chain",
        groups=[
            Group(
                id="MALE",
                sampling=true_sampling['MALE'],
                estimation=Estimation(
                    parameters=[
                        pars_biased['MALE']
                    ],
                )
            ),
            Group(
                id="FEMALE",
                sampling=true_sampling['FEMALE'],
                estimation=Estimation(
                    parameters=[
                        pars_biased['FEMALE']
                    ],
                )
            ),
        ]
    )



    # for n in [1, 2, 10, 20]:
    #     m_new = exp_uninformative.model_copy()
    #     m_new.groups[0].sampling.n_samples = n

    exps = PETabExperimentList(
        experiments=[exp_exact, exp_uninformative, exp_high_variance, exp_biased]
    )

    json_ = exps.model_dump_json(indent=2)
    console.print(json_)
    console.rule(style="white")

    yml = to_yaml_str(exps)
    console.print(yml)
    console.rule(style="white")

    exps_m = exps.model_dump(mode='json')

    # Dump PETabExperiments into YAML file
    with open(Path(__file__).parent / "xps.yaml", "w") as f:
        yaml.dump(exps_m, f, sort_keys=False, indent=2)
