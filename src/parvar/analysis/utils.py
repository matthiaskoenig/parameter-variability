import pandas as pd
import ast


def extract_key_from_dict(s: pd.Series, key: str) -> pd.Series:
    """
    Given a Series of strings that look like dictionaries,
    return a Series with the value for `key` from each.
    """

    def parse_and_get(x):
        d = ast.literal_eval(x)

        if isinstance(d, dict):
            return d.get(key)
        else:
            return None

    return s.apply(parse_and_get)
