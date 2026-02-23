from parvar.analysis.utils import (
    uuid_alphanumeric,
    get_group_from_pid,
    get_parameter_from_pid
)


def test_uuid_generation() -> None:
    """Test UUID generation."""
    uuid: str = uuid_alphanumeric()
    assert uuid

def test_get_group_from_pid() -> None:
    """Test get_group_from_pid()."""
    group: str = get_group_from_pid('PBPK_PARAMETER_GROUP-1')
    assert group

def test_get_parameter_from_pid() -> None:
    """Test get_parameter_from_pid()."""
    param: str = get_parameter_from_pid('PBPK_PARAMETER_GROUP-1')
    assert param
