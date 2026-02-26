"""Create all experiments."""

from parvar import RESULTS_SIMPLE_PK, RESULTS_SIMPLE_CHAIN, RESULTS_ICG
from parvar.experiments.petab_factory import create_petabs_for_definitions

from parvar.experiments.factories import definitions_minimal
from parvar.experiments.factories.simple_pk import (
    factory_data as factory_data_simple_pk,
)
from parvar.experiments.factories.simple_chain import (
    factory_data as factory_data_simple_chain,
)
from parvar.experiments.factories.icg_body_flat import (
    factory_data as factory_data_icg_body_flat,
)


if __name__ == "__main__":
    for results_path, factory_data in [
        (RESULTS_SIMPLE_PK, factory_data_simple_pk),
        (RESULTS_SIMPLE_CHAIN, factory_data_simple_chain),
        (RESULTS_ICG, factory_data_icg_body_flat),
    ]:
        create_petabs_for_definitions(
            results_path=results_path,
            definitions=definitions_minimal,
            factory_data=factory_data,
        )
