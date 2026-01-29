from parvar.analysis.utils import uuid_alphanumeric


def test_uuid_generation() -> None:
    """Test UUID generation."""
    uuid: str = uuid_alphanumeric()
    assert uuid
