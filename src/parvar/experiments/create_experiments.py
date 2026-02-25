"""Create all experiments."""

from parvar import RESULTS_SIMPLE_PK, RESULTS_SIMPLE_CHAIN, RESULTS_ICG
from parvar.experiments.petab_factory import create_petabs_for_definitions

from parvar.experiments.factories.simple_pk import definitions as definitions_simple_pk
from parvar.experiments.factories.simple_pk import (
    factory_data as factory_data_simple_pk,
)

from parvar.experiments.factories.simple_chain import (
    definitions as definitions_simple_chain,
)
from parvar.experiments.factories.simple_chain import (
    factory_data as factory_data_simple_chain,
)


from parvar.experiments.factories.icg_body_flat import (
    definitions as definitions_icg_body_flat,
)
from parvar.experiments.factories.simple_chain import (
    factory_data as factory_data_icg_body_flat,
)


if __name__ == "__main__":
    # simple pk
    create_petabs_for_definitions(
        results_path=RESULTS_SIMPLE_PK,
        definitions=definitions_simple_pk,
        factory_data=factory_data_simple_pk,
    )

    # simple chain
    create_petabs_for_definitions(
        results_path=RESULTS_SIMPLE_CHAIN,
        definitions=definitions_simple_chain,
        factory_data=factory_data_simple_chain,
    )

    # icg_body_flat
    create_petabs_for_definitions(
        results_path=RESULTS_ICG,
        definitions=definitions_icg_body_flat,
        factory_data=factory_data_icg_body_flat,
    )
