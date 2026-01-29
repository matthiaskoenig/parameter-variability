import pytest
from pathlib import Path
from typing import Callable

from parvar.analysis.petab_factory import create_petabs_for_definitions
from parvar.factories.icg_body_flat import definitions as definitions_icg
from parvar.factories.icg_body_flat import factory as factory_icg
from parvar.factories.simple_chain import definitions as definitions_simple_chain
from parvar.factories.simple_chain import factory as factory_simple_chain
from parvar.factories.simple_pk import definitions as definitions_simple_pk
from parvar.factories.simple_pk import factory as factory_simple_pk


testdata = [
    (definitions_simple_chain, factory_simple_chain),
    (definitions_icg, factory_icg),
    (definitions_simple_pk, factory_simple_pk),
]


@pytest.mark.parametrize("definitions, factory", testdata, ids=["simple_chain", "icg"])
def test_factory(definitions: dict, factory: Callable, tmp_path: Path) -> None:
    """Test the factory."""
    create_petabs_for_definitions(
        definitions={k: v for (k, v) in definitions.items() if k == "timepoints"},
        factory=factory,
        results_path=tmp_path,
    )
    assert (tmp_path / "xps" / "timepoints").exists()
