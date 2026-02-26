# from pathlib import Path
#
# import pandas as pd
#
#
# def collect_optimization_results(yaml_paths: list[Path]) -> pd.DataFrame:
#     """Collect the optimization results after the optimization finished."""
#
#     # FIXME: this is after the optimization in the analysis
#
#     yaml_files = sorted(yaml_files)
#
#     infos = []
#     for yaml_file in yaml_files:
#         console.rule(yaml_file.name, style="white", align="left")
#         # results: list[dict] = optimize_petab_xp(yaml_file)
#         # read result file
#         infos.extend(results)
#
#     df = pd.DataFrame(infos)
#     df.to_csv(
#         results_path / "xps" / xp_type / "bayes_results.tsv", sep="\t", index=False
#     )
#     console.print(df)
#     return df
