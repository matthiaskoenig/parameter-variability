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

# fewer starts and samples than an analysis uses: the test checks that the
# optimization and the sampling run through, not the quality of the estimates
test_settings = {
    "n_starts": 4,
    "n_samples": 500,
    "n_chains": 2,
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
        settings=test_settings,
    )
    # every experiment is optimized and sampled, an error is written to a file
    for yaml_path in yaml_paths:
        uid = yaml_path.parent.name
        assert not (tmp_path / f"{uid}_errors.tsv").exists(), (
            tmp_path / f"{uid}_errors.tsv"
        ).read_text()
        assert (tmp_path / f"{uid}_results.tsv").exists()
