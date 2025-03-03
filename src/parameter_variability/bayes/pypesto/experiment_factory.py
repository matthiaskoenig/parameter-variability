"""Factory to create the various experiments."""
from parameter_variability.bayes.pypesto.experiment import *
from pathlib import Path


if __name__ == "__main__":

    from pymetadata.console import console

    console.rule(style="white")
    console.print(PETabExperiment.model_json_schema())
    console.rule(style="white")

    true_par: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 1.0, "scale": 0.2})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
            type=DistributionType.LOGNORMAL,
            parameters={"loc": 10.0, "scale": 0.2}))
    }

    true_sampling: dict[str, Sampling] = {
        'MALE': Sampling(
            n_samples=10,
            steps=20,
            parameters=[true_par['MALE']]
        ),
        'FEMALE': Sampling(
            n_samples=10,
            steps=20,
            parameters=[true_par['FEMALE']])
    }

    pars_uninformative: dict[str, Parameter] = {
        'MALE': Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 1.0, "scale": 0.2})),
        'FEMALE': Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 10.0, "scale": 0.2}))
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
        id="high_variance",
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

    from pydantic_yaml import parse_yaml_file_as, to_yaml_str
    import yaml
    yml = to_yaml_str(exps)
    console.print(yml)
    console.rule(style="white")

    exps_m = exps.model_dump(mode='json')

    # This parses YAML as the MyModel type
    with open(Path(__file__).parent / "exp_test.yml", "w") as f:
        # yml = f.read()
        # exp_test = parse_yaml_raw_as(PETabExperimentList, yml)
        # json = exp_test.model_dump_json(indent=2)
        # console.print(json)
        # console.rule(style="white")
        # f.write(yml) # Works but not ordered
        yaml.dump(exps_m, f, sort_keys=False, indent=2)
