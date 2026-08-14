import re
from pathlib import Path

import json
import numpy as np
import pandas as pd
import ast

from pymetadata.console import console
from parvar import OPTIMIZATION_RUN, SERVER_RUN_NUMBER


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


def join_optimization_results(
    results_path: Path, xp_type: str, server: bool = True
) -> pd.DataFrame:
    """Join the experiment setup with the results.
    Outputs the result dataframe used in plots
    """

    console.print(f"Joining optimization results '{results_path.name}'...")
    # directories: Path = results_path / "xps" / xp_type
    # console.print(directories)

    # Optimization results
    if server:
        directories: Path = (
            results_path.parent
            / "server"
            / SERVER_RUN_NUMBER
            / results_path.name
            / "xps"
            / xp_type
        )
        optim_filenames = (directories / OPTIMIZATION_RUN).glob("*.tsv")
    else:
        directories: Path = results_path / "xps" / xp_type
        optim_filenames = (
            results_path.parent / OPTIMIZATION_RUN / results_path.name
        ).glob("*.tsv")

    # if file was saved before, just use cache
    if (directories / "definitions_results.parquet").is_file():
        console.print(f"Reading Parquet '{results_path.name}'...")
        return pd.read_parquet((directories / "definitions_results.parquet"))

    # 1st Simple model eveyrone know
    #   Textbook example
    # 2nd Used a real life model with real life data
    #   Pick an example from another lab

    df_ls = []
    for filename in optim_filenames:
        # console.print(filename)
        df = pd.read_csv(filename, sep="\t")
        df["values"] = df["values"].apply(lambda x: np.array(json.loads(x)))
        df_ls.append(df)

    df_bayes = pd.concat(df_ls)
    # df_bayes.drop(["Unnamed: 0"], axis=1, inplace=True)
    console.print(df_bayes.info())

    df_xp = pd.read_csv(directories / "definitions.tsv", sep="\t")

    df_join = df_xp.merge(df_bayes, on=["id", "group", "parameter"], how="inner")
    df_join["sample_loc"] = extract_key_from_dict(df_join["dsn_par"], "loc")
    df_join["sample_scale"] = extract_key_from_dict(df_join["dsn_par"], "scale")

    col_rename = {
        "mean": "bayes_sampler_mean",
        "median": "bayes_sampler_median",
        "n_samples": "bayes_sampler_n_samples",
        "values": "bayes_sampler_values",
    }

    df_join.rename(columns=col_rename, inplace=True)

    # Order of the columns for the result dataframe
    col_order = [
        "id",
        "model",
        "prior_type",
        "group",
        "parameter",
        "samples",
        "timepoints",
        "noise_cv",
        "sample_loc",
        "sample_scale",
        "optim_duration",
        "bayes_sampler_mean",
        "bayes_sampler_median",
        "bayes_sampler_n_samples",
        "bayes_sampler_values",
        "ess",
        "hdi_high",
        "hdi_low",
    ]

    df = df_join[col_order]
    df = df[df["prior_type"] != "no_prior"]  # remove no prior experiments

    df["bayes_sampler_values"] = df["bayes_sampler_values"].apply(
        lambda x: json.dumps(x.tolist())
    )

    df.to_csv(directories / "definitions_results.tsv", sep="\t", index=False)
    df.to_parquet(directories / "definitions_results.parquet")

    return df


def reference_df_filter(column: str, df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Get subset of dataframe for column."""
    reference_cp = reference.copy()
    reference_cp.pop(column)

    mask = pd.Series([True] * len(df), index=df.index)

    for col, val in reference_cp.items():
        mask &= df[col] == val

    return df[mask]


def point_bias(df: pd.DataFrame, array: np.array) -> np.array:
    return (df["sample_loc"].to_numpy() - array) / df["sample_loc"].to_numpy()


# Use in facetted_heatmap


def _sort_columns(columns):
    """
    Sort column values intelligently:
    - If all values are numeric (or can be converted to numeric), sort numerically ascending.
    - Otherwise, sort alphabetically ascending.

    Parameters
    ----------
    columns : pd.Index
        The column index to sort.

    Returns
    -------
    list
        Sorted list of column values.
    """
    col_list = list(columns)

    # Attempt to convert all column values to numeric
    numeric_values = []
    all_numeric = True
    for val in col_list:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            all_numeric = False
            break

    if all_numeric:
        # Sort numerically ascending: pair numeric values with originals, sort, extract
        sorted_pairs = sorted(zip(numeric_values, col_list), key=lambda x: x[0])
        return [pair[1] for pair in sorted_pairs]
    else:
        # Sort alphabetically ascending
        return sorted(col_list, key=str)


def _parse_string_to_array(s):
    """
    Parse a string representation of a (possibly nested) numerical array
    into a flat numpy float array.

    Handles formats like:
        - "[[1.0, 2.0], [3.0, 4.0]]"
        - "[1.0, 2.0, 3.0]"
        - "[[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]]"
        - "1.0" (single value)
        - Already parsed lists/arrays (passthrough)

    Parameters
    ----------
    s : str, list, np.ndarray, or numeric
        The value to parse.

    Returns
    -------
    np.ndarray
        1D float array (flattened).
    """
    # If already a numpy array, just flatten
    if isinstance(s, np.ndarray):
        return s.flatten().astype(float)

    # If already a list, convert and flatten
    if isinstance(s, list):
        return np.array(_flatten_nested(s), dtype=float)

    # If a single numeric value
    if isinstance(s, (int, float)):
        return np.array([float(s)])

    # If None or NaN-like
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.array([], dtype=float)

    # It's a string — try to parse it
    if isinstance(s, str):
        s = s.strip()

        # Handle empty strings
        if s == "" or s == "[]" or s == "[[]]":
            return np.array([], dtype=float)

        # Try ast.literal_eval first (handles Python-style lists)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (int, float)):
                return np.array([float(parsed)])
            elif isinstance(parsed, list):
                return np.array(_flatten_nested(parsed), dtype=float)
        except (ValueError, SyntaxError):
            pass

        # Try json.loads (handles JSON-style arrays)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (int, float)):
                return np.array([float(parsed)])
            elif isinstance(parsed, list):
                return np.array(_flatten_nested(parsed), dtype=float)
        except (ValueError, json.JSONDecodeError):
            pass

        # Last resort: extract all numbers via regex
        try:
            numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
            if numbers:
                return np.array([float(n) for n in numbers], dtype=float)
        except (ValueError, TypeError):
            pass

    # If nothing worked, return empty
    return np.array([], dtype=float)


def _flatten_nested(lst):
    """
    Recursively flatten a nested list of arbitrary depth.

    Parameters
    ----------
    lst : list
        Potentially nested list.

    Returns
    -------
    list
        Flat list of numeric values.
    """
    flat = []
    for item in lst:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat.extend(_flatten_nested(item))
        elif isinstance(item, (int, float)):
            flat.append(float(item))
        elif item is None:
            continue
        else:
            try:
                flat.append(float(item))
            except (ValueError, TypeError):
                continue
    return flat


def _parse_and_flatten(df, value_col):
    """
    Parse string-encoded arrays in the value column and flatten them.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with string-encoded array column.
    value_col : str
        Column name containing string representations of arrays.

    Returns
    -------
    pd.DataFrame
        Copy of df with value_col replaced by flattened numpy arrays.
    """
    df_parsed = df.copy()
    df_parsed[value_col] = df_parsed[value_col].apply(_parse_string_to_array)
    return df_parsed


def _aggregate_arrays(df, value_col, value_agg):
    """
    Aggregate array-valued column to scalar values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with array-valued column (already parsed and flattened).
    value_col : str
        Column containing flat numpy arrays.
    value_agg : str or callable
        Aggregation method.

    Returns
    -------
    pd.DataFrame
        Copy of df with an additional column '{value_col}__scalar' containing
        the aggregated scalar values.
    """
    agg_functions = {
        "mean": lambda arr: np.nanmean(arr),
        "median": lambda arr: np.nanmedian(arr),
        "sum": lambda arr: np.nansum(arr),
        "min": lambda arr: np.nanmin(arr),
        "max": lambda arr: np.nanmax(arr),
        "std": lambda arr: np.nanstd(arr),
        "var": lambda arr: np.nanvar(arr),
        "length": lambda arr: len(arr),
        "first": lambda arr: arr[0] if len(arr) > 0 else np.nan,
        "last": lambda arr: arr[-1] if len(arr) > 0 else np.nan,
        "rms": lambda arr: np.sqrt(np.nanmean(np.array(arr) ** 2)),
        "percentile_25": lambda arr: np.nanpercentile(arr, 25),
        "percentile_75": lambda arr: np.nanpercentile(arr, 75),
        "iqr": lambda arr: np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25),
        "range": lambda arr: np.nanmax(arr) - np.nanmin(arr),
    }

    if isinstance(value_agg, str):
        if value_agg not in agg_functions:
            raise ValueError(
                f"Unknown value_agg '{value_agg}'. "
                f"Available options: {list(agg_functions.keys())}"
            )
        agg_func = agg_functions[value_agg]
    elif callable(value_agg):
        agg_func = value_agg
    else:
        raise TypeError("value_agg must be a string or callable.")

    df_agg = df.copy()
    agg_col = f"{value_col}__scalar"

    df_agg[agg_col] = df_agg[value_col].apply(
        lambda arr: _safe_aggregate(arr, agg_func)
    )

    return df_agg


def _safe_aggregate(arr, agg_func):
    """
    Safely aggregate an array, handling edge cases.

    Parameters
    ----------
    arr : np.ndarray
        The flat array to aggregate.
    agg_func : callable
        Function that takes an array and returns a scalar.

    Returns
    -------
    float
        Aggregated scalar value, or np.nan if aggregation fails.
    """
    try:
        if arr is None:
            return np.nan
        if isinstance(arr, np.ndarray) and len(arr) == 0:
            return np.nan
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return np.nan
        return float(agg_func(arr))
    except (TypeError, ValueError, IndexError):
        return np.nan
