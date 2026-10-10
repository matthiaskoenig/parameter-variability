"""Creating all the models from scratch."""

from pathlib import Path

from parvar.models.icg_example import example_icg
from parvar.models.simple_chain import create_simple_chain
from parvar.models.simple_chain_example import example_simple_chain
from parvar.models.simple_pk import create_simple_pk
from parvar.models.simple_pk_example import example_simple_pk


def run_all(output_dir: Path | None = None):
    """Create all models in `output_dir` and run examples.

    The models are written to the models of the package by default; the
    examples simulate those.
    """
    example_icg()

    create_simple_chain(output_dir=output_dir)
    example_simple_chain()

    create_simple_pk(output_dir=output_dir)
    example_simple_pk()


if __name__ == "__main__":
    run_all()
