"""Creating all the models from scratch."""

from parvar.models.icg_example import example_icg
from parvar.models.simple_chain import create_simple_chain
from parvar.models.simple_chain_example import example_simple_chain
from parvar.models.simple_pk import create_simple_pk
from parvar.models.simple_pk_example import example_simple_pk


def run_all():
    """Create all models and run examples."""
    example_icg()

    create_simple_chain()
    example_simple_chain()

    create_simple_pk()
    example_simple_pk()


if __name__ == "__main__":
    run_all()
