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
from typing import Optional, Union, List
import yaml

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

# TODO: finish Compartment integration
class Compartment(BaseModel):
    """Compartment settings for sampling"""
    id: str
    starting_value: float


class Estimation(BaseModel):
    """Priors used for sampling or estimation."""
    parameters: list[Parameter]

    def get_dsn_parameter(self, parameter: str) -> Optional[dict[str, float]]:
        for par in self.parameters:
            if par.id == parameter:
                return par.distribution.parameters
        return None


class Sampling(BaseModel):
    """Priors used for sampling or estimation."""
    parameters: list[Parameter]
    n_samples: int = 10
    steps: int = 20
    tend: float = 100
    # compartments: list[Compartment]
    # FIXME: error settings, ...

    def get_dsn_parameter(self, parameter: str) -> Optional[dict[str, float]]:
        for par in self.parameters:
            if par.id == parameter:
                return par.distribution.parameters
        return None


class Group(BaseModel):
    """Group for stratification of model."""
    id: str
    sampling: Sampling
    estimation: Estimation

    def get_parameter_list(self, type: str) -> list[Parameter]:
        strat: Union[Sampling, Estimation] = getattr(self, type)
        return strat.parameters


    def get_parameter(self, type: str, parameter: str, dsn_par: str) -> float:
        strat: Union[Sampling, Estimation] = getattr(self, type)
        dsn_parameters = strat.get_dsn_parameter(parameter)

        return dsn_parameters[dsn_par]


class PETabExperiment(BaseModel):
    """PETab experiment."""
    id: str
    model: str
    dosage: Optional[dict[str, float]] = None # ;   dosage = {"IVDOSE_icg": 10}
    skip_error_column: Optional[List[str]] = None
    groups: List[Group]

    @property
    def group_ids(self) -> list[str]:
        return [g.id for g in self.groups]

    def group_by_id(self, id: str) -> Optional[Group]:
        """Get group by id, return None if not found."""
        for g in self.groups:
            if g.id == id:
                return g
        return None

class PETabExperimentList(BaseModel):
    """PETab experiment list."""
    experiments: list[PETabExperiment]

    def to_yaml(self, path: Path):
        # json_ = self.model_dump_json(indent=2)
        # console.print(json_)
        # console.rule(style="white")
        # yml = to_yaml_str(self)
        # console.print(yml)
        # console.rule(style="white")

        # Dump PETabExperiments into YAML file
        with open(path, "w") as f:
            exps_m = self.model_dump(mode='json')
            yaml.dump(exps_m, f, sort_keys=False, indent=2)


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

    e_new: PETabExperiment = parse_yaml_raw_as(PETabExperiment, yml)
    console.print(e_new)
    console.print(e_new.group_ids)

