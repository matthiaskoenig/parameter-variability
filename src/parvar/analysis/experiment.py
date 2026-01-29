"""Factory to create the various PETab simulation experiments."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Any

import pandas as pd
import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
from pymetadata.console import console


class DistributionType(str, Enum):
    """Type of prior."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"


class BaseModel(PydanticBaseModel):
    """Base model."""

    pass


class Distribution(BaseModel):
    """Parameter dictionary for the priors."""

    parameters: dict[str, float]
    type: DistributionType = DistributionType.LOGNORMAL


class Noise(BaseModel):
    """Parameter dictionary for noise."""

    add_noise: bool
    cv: float
    type: DistributionType = DistributionType.NORMAL


class Parameter(BaseModel):
    """Priors used for sampling or estimation."""

    id: str
    distribution: Distribution


class Observable(BaseModel):
    """Observables settings for sampling"""

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
    noise: Noise
    observables: Optional[list[Observable]]

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
    prior_type: str
    dosage: Optional[dict[str, float]] = (
        None  # ;   dosage = {"IVDOSE_icg": 10}  # FIXME: rename to model_changes
    )
    # observables: FIXME;  # [Cve_icg], Afeces_icg, [LI__icg], [LI__icg_bi] | [Cve_icg]
    skip_error_column: Optional[List[str]] = None  # FIXME move to sampling
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

    @staticmethod
    def print_schema():
        """Print schema information."""
        console.rule("JSON schema", style="white", align="left")
        console.print(PETabExperiment.model_json_schema())

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def print_json(self):
        """Print JSON."""
        console.rule("JSON", style="white", align="left")
        console.print(self.to_json())

    def to_yaml(self) -> str:
        """Serialize to YAML."""
        return to_yaml_str(self)

    @staticmethod
    def from_yaml(yaml: str) -> PETabExperiment:
        return parse_yaml_raw_as(PETabExperiment, yaml)

    def print_yaml(self):
        """Print YAML."""
        console.rule("YAML", style="white", align="left")
        console.print(self.to_yaml())


class PETabExperimentList(BaseModel):
    """PETab experiment list."""

    experiments: list[PETabExperiment]

    def to_yaml_file(self, path: Path):
        """Dump PETabExperiments into YAML file."""
        with open(path, "w") as f:
            exps_m = self.model_dump(mode="json")
            yaml.dump(exps_m, f, sort_keys=False, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        """Serialization of list of experiments to DataFrame."""
        items = []
        for exp in self.experiments:
            d = {}
            d["id"] = exp.id
            d["model"] = exp.model
            d["n_groups"] = len(exp.groups)
            d["groups"] = [g.id for g in exp.groups]
            d["prior_type"] = exp.prior_type
            d["n"] = [g.sampling.n_samples for g in exp.groups]
            d["n_t"] = [g.sampling.steps for g in exp.groups]
            d["noise_cv"] = [g.sampling.noise.cv for g in exp.groups]

            parameters: List[Any] = []
            pars_per_group = [g.sampling.parameters for g in exp.groups]
            for pars in pars_per_group:
                par_ls = []
                for par in pars:
                    par_det = {
                        par.id: par.distribution.parameters,
                        "dsn_type": str(par.distribution.type),
                    }
                    par_ls.append(par_det)

                parameters.append(par_ls)
            d["parameters"] = parameters

            # d = exp.model_dump(mode='python')
            items.append(d)

        df = pd.DataFrame(data=items)
        cols_with_ls = ["groups", "n", "n_t", "noise_cv", "parameters"]
        #
        # for col in cols_with_ls:
        #     df[col] = df[col].apply(literal_eval)
        #
        df = df.explode(cols_with_ls)

        return df

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)


__all__ = [
    "DistributionType",
    "Distribution",
    "Parameter",
    "Sampling",
    "Estimation",
    "Group",
    "PETabExperiment",
    "PETabExperimentList",
    "Noise",
    "Observable",
]
