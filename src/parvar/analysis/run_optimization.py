from itertools import product
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pymetadata.console import console

from parvar import RESULTS_ICG
from parvar.analysis.petab_optimization import PyPestoSampler


def xps_selector(
    results_dir: Path, xp_type: str, conditions: Optional[Dict[str, list]]
) -> list[str]:
    """Select the xps that match the desired conditions."""
    df = pd.read_csv(results_dir / f"xps_{xp_type}" / "results.tsv", sep="\t")

    if not conditions:  # empty dict -> no filtering
        return df

    if "n_t" in conditions:
        conditions["n_t"] = [t - 1 for t in conditions["n_t"]]

    combinations = list(product(*(conditions[col] for col in conditions)))

    matching_indices = set()

    for comb in combinations:
        comb_dict = dict(zip(conditions.keys(), comb))
        mask = pd.Series(True, index=df.index)
        for col, val in comb_dict.items():
            mask &= df[col].eq(val)
        matching_indices.update(df[mask].index)

    df_res = df.loc[list(matching_indices)].sort_index()

    if df_res.empty:
        raise console.print(
            "No XPs were selected. Check if conditions are correct", style="warning"
        )

    return df_res["id"].unique().tolist()


def optimize_petab_xp(yaml_file: Path) -> list[dict]:
    """Optimize single petab problem using PyPesto."""
    pypesto_sampler = PyPestoSampler(yaml_file=yaml_file)
    pypesto_sampler.load_problem()
    pypesto_sampler.optimizer()
    pypesto_sampler.bayesian_sampler(n_samples=1000)
    pypesto_sampler.results_hdi()
    # pypesto_sampler.results_median()

    results = []
    results_petab = pypesto_sampler.results_dict()
    for pid, stats in results_petab.items():
        results.append(
            {
                "xp": yaml_file.parent.name,
                "pid": pid,
                **stats,
            }
        )

    return results


def optimize_petab_xps(results_dir: Path, exp_type: str, xp_ids: list[str]):
    """Optimize the given PEtab problems."""

    xp_path = results_dir / f"xps_{exp_type}"
    yaml_files: list[Path] = []
    for xp in xp_path.iterdir():
        if xp.is_dir() and xp.name in xp_ids:
            for yaml_file in xp.glob("**/petab.yaml"):
                yaml_files.append(yaml_file)

    yaml_files = sorted(yaml_files)

    infos = []
    for yaml_file in yaml_files:
        console.rule(yaml_file.name, style="white", align="left")
        results: list[dict] = optimize_petab_xp(yaml_file)
        infos.extend(results)

    df = pd.DataFrame(infos)
    df.to_csv(
        RESULTS_ICG / f"xps_{exp_type}" / "bayes_results.tsv", sep="\t", index=False
    )
    console.print(df)
    return df
