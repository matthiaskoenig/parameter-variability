"""Helper functions."""

import string
import uuid
import re

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase  # 0-9A-Za-z
BASE = len(ALPHABET)  # 62

_GROUP_PATTERN = re.compile(r'^[\x00-\x7f]+$')
_INVALID_UNDERSCORE = re.compile(r'_')

def _int_to_base62(n: int) -> str:
    """Encode a non-negative integer to base62 using ALPHABET."""
    if n == 0:
        return ALPHABET[0]
    chars = []
    while n > 0:
        n, r = divmod(n, BASE)
        chars.append(ALPHABET[r])
    return "".join(reversed(chars))


def uuid_alphanumeric(length: int = 20) -> str:
    """
    Generate an alphanumeric identifier of exactly `length` characters,
    using UUIDv4 as the entropy source.

    Args:
        length (int): Desired length (must be > 0).

    Returns:
        str: Alphanumeric identifier (0-9, A-Z, a-z) of length `length`.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")

    parts = []
    while sum(len(p) for p in parts) < length:
        # Use uuid4()'s 128-bit randomness; convert to an integer and base62-encode
        u_int = uuid.uuid4().int  # large integer from UUIDv4
        parts.append(_int_to_base62(u_int))

    return "".join(parts)[:length]

def get_group_from_pid(s: str) -> str | None:
    """
    Extract the last word from an underscore-separated string.

    - Words are separated by '_'.
    - The last word may contain any ASCII characters except '_'.
    - Returns None if there is no valid last word.
    """
    # Get the part after the last underscore (or the whole string if no underscore)
    last = s.rsplit('_', 1)[-1]

    # Must be non-empty, ASCII-only, and contain no underscore
    if last and _GROUP_PATTERN.match(last) and not _INVALID_UNDERSCORE.search(last):
        return last
    return None


def get_parameter_from_pid(s: str) -> str:
    """
    Remove the last word from an underscore-separated string.

    - Words are separated by '_'.
    - The last word may contain any ASCII characters except '_'.
    - If the last part is not a valid last word, the original string is returned.
    """
    # Find last underscore position
    idx = s.rfind('_')  # similar usage appears in typical underscore-trimming examples [web:11][web:14]

    # If there is no underscore, nothing to remove
    if idx == -1:
        return s

    last = s[idx + 1 :]

    # Validate last word
    if last and _GROUP_PATTERN.match(last) and not _INVALID_UNDERSCORE.search(last):
        return s[:idx]  # everything before the last underscore
    return s


if __name__ == "__main__":
    print(uuid_alphanumeric())
    print(get_group_from_pid('PBPK_PARAMETER_GROUP-1'))
    print(get_parameter_from_pid('PBPK_PARAMETER_GROUP-1'))
