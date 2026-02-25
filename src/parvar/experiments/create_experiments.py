"""Create all experiments."""

from parvar import RESULTS_SIMPLE_PK
from parvar.experiments.petab_factory import create_petabs_for_definitions
from parvar.experiments.factories.simple_pk import definitions as definitions_simple_pk
from parvar.experiments.factories.simple_pk import (
    factory_data as factory_data_simple_pk,
)

# select subset
# definitions = {k:v for k,v in definitions if k=="timepoints"}

# simple pk
create_petabs_for_definitions(
    results_path=RESULTS_SIMPLE_PK,
    definitions=definitions_simple_pk,
    factory_data=factory_data_simple_pk,
)
