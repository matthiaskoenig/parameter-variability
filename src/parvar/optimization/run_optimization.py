"""Run example optimizations with model."""

from pathlib import Path

from pymetadata.console import console

from parvar import RESULTS_DIR, RESULTS_SIMPLE_PK, RESULTS_SIMPLE_CHAIN, RESULTS_ICG
from parvar.experiments.petab_factory import select_all_experiments
from parvar.optimization.petab_optimization import optimize_experiments


if __name__ == "__main__":
    optimization_run: str = "2025-03-23_v3"

    for results_path in [
        RESULTS_SIMPLE_PK,
        RESULTS_SIMPLE_CHAIN,
        RESULTS_ICG,
    ]:
        console.rule(results_path.name, align="left", style="white")

        # select problems to optimize
        # yaml_paths: list[Path] = select_experiments(
        #     results_path=results_path,
        #     definitions=definitions_subset,
        # )
        # console.print(f"YAML paths: {len(yaml_paths)}")
        # console.rule()

        opt_results_dir = RESULTS_DIR / optimization_run / results_path.name
        opt_results_dir.mkdir(parents=True, exist_ok=True)

        # select all problems to optimize
        yaml_paths: list[Path] = select_all_experiments(
            results_path=results_path,
        )
        yaml_paths = sorted(yaml_paths)
        console.print(f"YAML paths: {len(yaml_paths)}")

        # optimize
        optimize_experiments(
            results_dir=opt_results_dir, yaml_paths=yaml_paths, caching=True
        )
