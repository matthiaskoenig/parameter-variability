from pathlib import Path

from parvar.models.create_models import run_all


def test_create_models(tmp_path: Path) -> None:
    """Test creation of models.

    The models are written to `tmp_path`, not over the models of the package
    which the other tests read.
    """
    run_all(output_dir=tmp_path)
    for sid in ["simple_chain", "simple_pk"]:
        assert (tmp_path / f"{sid}.xml").exists()
        assert (tmp_path / f"{sid}.md").exists()
