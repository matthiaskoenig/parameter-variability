import pytest
from pathlib import Path
from typing import Callable

from parvar.experiments.petab_factory import create_petabs_for_definitions

from parvar.experiments.factories import definitions as definitions_icg
from parvar.experiments.factories import factory as factory_icg

from parvar.experiments.factories import definitions as definitions_simple_chain
from parvar.experiments.factories import factory as factory_simple_chain

from parvar.experiments.factories import definitions as definitions_simple_pk
from parvar.experiments.factories import factory as factory_simple_pk

testdata = [
    (definitions_simple_chain, factory_simple_chain),
    (definitions_icg, factory_icg),
    (definitions_simple_pk, factory_simple_pk),
]


@pytest.mark.parametrize(
    "definitions, factory", testdata, ids=["simple_chain", "icg", "simple_pk"]
)
def test_factory(definitions: dict, factory: Callable, tmp_path: Path) -> None:
    """Test the factory."""
    create_petabs_for_definitions(
        definitions={k: v for (k, v) in definitions.items() if k == "timepoints"},
        factory=factory,
        results_path=tmp_path,
    )
    assert (tmp_path / "xps" / "timepoints").exists()
