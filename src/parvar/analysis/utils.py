"""Helper functions."""

import string
import uuid

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase  # 0-9A-Za-z
BASE = len(ALPHABET)  # 62


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


if __name__ == "__main__":
    print(uuid_alphanumeric())
