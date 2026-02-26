from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from console import console

from parvar import RESULTS_SIMPLE_PK
from parvar.analysis.utils import extract_key_from_dict


@dataclass
class PyPestoAnalysis:
    results_path: Path
    xp_type: str

    def load_results_df(self, filename: str) -> pd.DataFrame:
        file_path = self.results_path / "xps" / self.xp_type / filename

        return pd.read_csv(file_path, sep="\t")

    def join_results_df(self, xp_results: str, bayes_results: str) -> pd.DataFrame:
        df_xp = self.load_results_df(xp_results)
        df_bayes = self.load_results_df(bayes_results)

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
            "bayes_sampler_mean",
            "bayes_sampler_median",
            "bayes_sampler_n_samples",
            "bayes_sampler_values",
        ]

        df = df_join[col_order]

        return df


if __name__ == "__main__":
    analysis = PyPestoAnalysis(
        results_path=RESULTS_SIMPLE_PK,
        xp_type="all",
    )

    results = analysis.join_results_df(
        xp_results="results.tsv", bayes_results="bayes_results.tsv"
    )

    results.to_csv(analysis.results_path / "xps" / "join.tsv", sep="\t", index=False)

    console.print(results)
