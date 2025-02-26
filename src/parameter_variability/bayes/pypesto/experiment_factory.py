"""Factory to create the various experiments."""
from parameter_variability.bayes.pypesto.experiment import *
from pathlib import Path


if __name__ == "__main__":

    from pymetadata.console import console

    console.rule(style="white")
    console.print(PETabExperiment.model_json_schema())
    console.rule(style="white")

    exp_uninformative = PETabExperiment(
        id="uninformative",
        model="simple_chain",
        groups=[
            Group(
                id="MALE",
                sampling=Sampling(
                    n_samples=10,
                    steps=20,
                    parameters=[
                        Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 1.0, "scale": 0.2})),
                    ],
                ),
                estimation=Estimation(
                    parameters=[
                        Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 1.0, "scale": 0.2})),
                    ],
                )
            ),
            Group(
                id="FEMALE",
                sampling=Sampling(
                    n_samples=10,
                    steps=20,
                    parameters=[
                        Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 10.0, "scale": 0.2})),
                    ],
                ),
                estimation=Estimation(
                    parameters=[
                        Parameter(id="k1", distribution=Distribution(
                            type=DistributionType.LOGNORMAL,
                            parameters={"loc": 10.0, "scale": 0.2})),
                    ],
                )
            ),
        ]
    )

    # for n in [1, 2, 10, 20]:
    #     m_new = exp_uninformative.model_copy()
    #     m_new.groups[0].sampling.n_samples = n


    json = exp_uninformative.model_dump_json(indent=2)
    console.print(json)
    console.rule(style="white")

    from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
    yml = to_yaml_str(exp_uninformative)
    console.print(yml)
    console.rule(style="white")

    # This parses YAML as the MyModel type
    with open(Path(__file__).parent / "exp_test.yml", "r") as f:
        yml = f.read()
        exp_test = parse_yaml_raw_as(PETabExperiment, yml)
        json = exp_test.model_dump_json(indent=2)
        console.print(json)
        console.rule(style="white")

