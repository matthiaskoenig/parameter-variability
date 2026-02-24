import pytest
from pathlib import Path
from typing import Callable

from parvar.analysis.petab_factory import create_petabs_for_definitions
from parvar.analysis.run_optimization import run_optimizations

from parvar.factories.icg_body_flat import definitions as definitions_icg
from parvar.factories.icg_body_flat import factory as factory_icg
from parvar.factories.icg_body_flat import optimizations as optimizations_icg

from parvar.factories.simple_chain import definitions as definitions_simple_chain
from parvar.factories.simple_chain import factory as factory_simple_chain
from parvar.factories.simple_chain import optimizations as optimizations_simple_chain

from parvar.factories.simple_pk import definitions as definitions_simple_pk
from parvar.factories.simple_pk import factory as factory_simple_pk
from parvar.factories.simple_pk import optimizations as optimizations_simple_pk

testdata = [
    (definitions_simple_chain, factory_simple_chain, optimizations_simple_chain),
    (definitions_icg, factory_icg, optimizations_icg),
    (definitions_simple_pk, factory_simple_pk, optimizations_simple_pk),
]

@pytest.mark.parametrize(
    "definitions, factory, optimizations", testdata, ids=["simple_chain", "icg", "simple_pk"]
)

def test_optimizations(
    definitions: dict,
    factory: Callable,
    optimizations: dict,
    tmp_path: Path
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
