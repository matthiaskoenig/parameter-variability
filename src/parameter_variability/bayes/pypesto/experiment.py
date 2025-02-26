"""Factory to create the various experiments.

sample_size:
    n_samples:
        - 1
        - 5
        - 10
        - 20
    prior_same: True

unbalanced_samples:
    n_samples:
        k1_MALE: 5
        k1_FEMALE: 10

ode_timesteps:
    time:
        - 3
        - 10
        - 20


"""
from enum import Enum
from pathlib import Path

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ValidationError, validator

class DistributionType(str, Enum):
    """Type of prior."""

    NORMAL: str = "normal"
    LOGNORMAL: str = "lognormal"


class BaseModel(PydanticBaseModel):
    """Base model."""
    pass

class Distribution(BaseModel):
    """Parameter dictionary for the priors.

    FIXME: add some validation
    """
    parameters: dict[str, float]
    type: DistributionType = DistributionType.LOGNORMAL


class Parameter(BaseModel):
    """Priors used for sampling or estimation."""
    id: str
    distribution: Distribution


class Estimation(BaseModel):
    """Priors used for sampling or estimation."""
    parameters: list[Parameter]


class Sampling(BaseModel):
    """Priors used for sampling or estimation."""
    parameters: list[Parameter]
    n_samples: int = 10
    steps: int = 20
    tend: float = 100
    # FIXME: error settings, ...


class Group(BaseModel):
    """Group for stratification of model."""
    id: str
    sampling: Sampling
    estimation: Estimation


class PETabExperiment(BaseModel):
    """PETab experiment."""
    id: str
    model: str
    groups: list[Group]

class PETabExperimentList(BaseModel):
    """PETab experiment list."""
    experiments: list[PETabExperiment]


__all__ = [
    "DistributionType",
    "Distribution",
    "Parameter",
    "Sampling",
    "Estimation",
    "Group",
    "PETabExperiment",
    "PETabExperimentList",
]

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

    json = exp_uninformative.model_dump_json(indent=2)
    console.print(json)
    console.rule(style="white")

    from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
    yml = to_yaml_str(exp_uninformative)
    console.print(yml)
    console.rule(style="white")
