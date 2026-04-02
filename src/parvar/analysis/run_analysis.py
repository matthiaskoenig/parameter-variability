from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.plots.bias_histogram_plot import bias_histogram
from parvar.analysis.plots.ess_violinplot import ess_violinplot  # , ess_violinplots,
from parvar.analysis.plots.grouped_boxplot import grouped_boxplot
from parvar.analysis.plots.runtime_boxplot import runtime_boxplot  # , runtime_boxplots
from parvar.analysis.utils import append_server_result, join_optimization_results


if __name__ == "__main__":
    reference = {
        "prior_type": "prior_biased_2",
        "timepoints": 9,
        "samples": 40,
        "noise_cv": 0.001,
    }

    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results_path = append_server_result(results_path=r, which="run_2")

        results = join_optimization_results(results_path=results_path, xp_type="all")
        plot_path = results_path / "xps" / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        for col in ["prior_type", "samples", "timepoints", "noise_cv"]:
            grouped_boxplot(results, column=col, save_path=plot_path)

            bias_histogram(results, column=col, save_path=plot_path)

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
