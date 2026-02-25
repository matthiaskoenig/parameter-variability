"""Test optimization of PEtab problems."""

import pytest
from pathlib import Path
from typing import Callable

from parvar.experiments.petab_factory import create_petabs_for_definitions
from parvar.optimization.run_optimization import run_optimizations

from parvar.experiments.factories import definitions as definitions_icg
from parvar.experiments.factories import factory as factory_icg
from parvar.experiments.factories import optimizations as optimizations_icg

from parvar.experiments.factories import definitions as definitions_simple_chain
from parvar.experiments.factories import factory as factory_simple_chain
from parvar.experiments.factories import optimizations as optimizations_simple_chain

from parvar.experiments.factories import definitions as definitions_simple_pk
from parvar.experiments.factories import factory as factory_simple_pk
from parvar.experiments.factories import optimizations as optimizations_simple_pk

testdata = [
    (definitions_simple_chain, factory_simple_chain, optimizations_simple_chain),
    (definitions_icg, factory_icg, optimizations_icg),
    (definitions_simple_pk, factory_simple_pk, optimizations_simple_pk),
]


@pytest.mark.parametrize(
    "definitions, factory, optimizations",
    testdata,
    ids=["simple_chain", "icg", "simple_pk"],
)
def test_optimizations(
    definitions: dict, factory: Callable, optimizations: dict, tmp_path: Path
) -> None:
    create_petabs_for_definitions(
        definitions={k: v for (k, v) in definitions.items() if k == "timepoints"},
        factory=factory,
        results_path=tmp_path,
    )

    run_optimizations(
        optimizations={k: v for (k, v) in optimizations.items() if k == "timepoints"},
        results_path=tmp_path,
    )

    assert (tmp_path / "xps" / "timepoints" / "bayes_results.tsv").exists()
