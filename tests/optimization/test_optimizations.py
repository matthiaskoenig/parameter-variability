"""Test optimization of PEtab problems."""

import pytest
from pathlib import Path

from parvar.experiments.factories import definitions_minimal
from parvar.experiments.petab_factory import (
    create_petabs_for_definitions,
    select_all_experiments,
)
from parvar.optimization.petab_optimization import optimize_experiments
from parvar.experiments.factories.simple_chain import (
    factory_data as factory_data_simple_chain,
)

optim = {
    "timepoints": {
        "timepoints": [5],
    },
}

testdata = [
    (definitions_minimal, factory_data_simple_chain, optim),
]


@pytest.mark.parametrize(
    "definitions, factory_data, optimizations",
    testdata,
    # ids=["simple_chain", "icg", "simple_pk"],
    ids=["simple_chain"],
)
def test_optimization(
    definitions: dict, factory_data: dict, optimizations: dict, tmp_path: Path
) -> None:
    """Test the optimizers."""
    create_petabs_for_definitions(
        results_path=tmp_path,
        definitions={k: v for (k, v) in definitions.items() if k == "timepoints"},
        factory_data=factory_data,
    )

    # select all problems to optimize
    yaml_paths: list[Path] = select_all_experiments(
        results_path=tmp_path,
    )
    yaml_paths = sorted(yaml_paths)

    optimize_experiments(
        results_dir=tmp_path,
        yaml_paths=yaml_paths,
        caching=False,
    )
