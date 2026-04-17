from pathlib import Path
from typing import Any

import pandas as pd
from pymetadata.console import console
from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.plots.bias_histogram_plot import bias_histogram
from parvar.analysis.plots.ess_violinplot import ess_violinplot  # , ess_violinplots,
from parvar.analysis.plots.grouped_boxplot import grouped_boxplot
from parvar.analysis.plots.runtime_boxplot import runtime_boxplot  # , runtime_boxplots
from parvar.analysis.utils import join_optimization_results

reference: dict[str, Any] = {
    "prior_type": "exact_prior",
    "timepoints": 5,
    "samples": 5,
    "noise_cv": 0.1,
}

optimization_run: str = "run_4"
xp_type: str = "all"


def get_server_results_path(
    results_path: Path, optimization_run: str = optimization_run
) -> Path:
    """Get location of the optimization results."""
    return results_path.parent / "server" / optimization_run / results_path.name


def preprocess_data(
    optimization_run: str = optimization_run, xp_type: str = xp_type
) -> None:
    """Preprocess the data once."""
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = get_server_results_path(results_path=r)
        _ = join_optimization_results(results_path=results_path, xp_type=xp_type)


def load_data(r, xp_type=xp_type) -> pd.DataFrame:
    """Load the results."""
    results_path = get_server_results_path(results_path=r)
    # df_path: Path = results_path / "xps" / xp_type / "definitions_results.tsv"
    # console.print(f"Loading data: {df_path}")
    # results: pd.DataFrame = pd.read_csv(df_path, sep="\t")

    df_path: Path = results_path / "xps" / xp_type / "definitions_results.parquet"
    console.print(f"Loading data: {df_path}")
    results: pd.DataFrame = pd.read_parquet(df_path)

    console.print(f"Data loaded: {df_path}", style="green")
    return results


if __name__ == "__main__":
    preprocess_data(optimization_run=optimization_run)

    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results: pd.DataFrame = load_data(r)
        results_path = get_server_results_path(r)

        plot_path = results_path / "xps" / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)

        for col in ["prior_type", "samples", "timepoints", "noise_cv"]:
            grouped_boxplot(
                results, reference=reference, column=col, save_path=plot_path
            )

            bias_histogram(
                results, reference=reference, column=col, save_path=plot_path
            )

            runtime_boxplot(results, column=col, save_path=plot_path)

            ess_violinplot(results, column=col, save_path=plot_path)

        # # 1. Reference plot
        # reference_plot(df=results, reference=reference, save_path=plot_path)
        #
        # # 2. Histogram plot
        # bias_histogram(df=results, save_path=plot_path)
        #
        # # 3. Runtime boxplot
        # runtime_boxplots(df=results, save_path=plot_path)
        #
        # # 4. ESS violin plot
        # ess_violinplots(df=results, save_path=plot_path)

        # grouped_boxplot(results, save_path=plot_path)
