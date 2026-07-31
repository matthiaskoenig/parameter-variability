from pathlib import Path

import json
import numpy as np
import pandas as pd
import ast

from pymetadata.console import console
from parvar import OPTIMIZATION_RUN


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
    directories: Path = results_path / "xps" / xp_type
    # console.print(directories)

    # Optimization results
    if server:
        optim_filenames = (directories / OPTIMIZATION_RUN).glob("*.tsv")
    else:
        optim_filenames = (
            results_path.parent / OPTIMIZATION_RUN / results_path.name
        ).glob("*.tsv")

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
