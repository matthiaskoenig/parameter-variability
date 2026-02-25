import pytest
from pathlib import Path

from parvar.experiments.petab_factory import create_petabs_for_definitions

from parvar.experiments.factories.icg_body_flat import definitions as definitions_icg
from parvar.experiments.factories.icg_body_flat import factory_data as factory_data_icg

from parvar.experiments.factories.simple_chain import (
    definitions as definitions_simple_chain,
)
from parvar.experiments.factories.simple_chain import (
    factory_data as factory_data_simple_chain,
)

from parvar.experiments.factories.simple_pk import definitions as definitions_simple_pk
from parvar.experiments.factories.simple_pk import (
    factory_data as factory_data_simple_pk,
)

testdata = [
    (definitions_simple_chain, factory_data_simple_chain),
    (definitions_icg, factory_data_icg),
    (definitions_simple_pk, factory_data_simple_pk),
]


@pytest.mark.parametrize(
    "definitions, factory_data", testdata, ids=["simple_chain", "icg", "simple_pk"]
)
def test_factory(definitions: dict, factory_data: dict, tmp_path: Path) -> None:
    """Test the factory."""
    create_petabs_for_definitions(
        results_path=tmp_path,
        definitions={k: v for (k, v) in definitions.items() if k == "timepoints"},
        factory_data=factory_data,
    )
    assert (tmp_path / "xps" / "timepoints").exists()
