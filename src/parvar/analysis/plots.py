# def visualize_timepoints_samples():
#     """Visualize the results."""
#
#     for xp_key in ["n", "Nt"]:
#
#         # data processing
#         df = pd.read_csv(RESULTS_ICG / f"xps_{xp_key}.tsv", sep="\t")
#         df[xp_key] = df.xp.str.split("_").str[-1]
#         df[xp_key] = df[xp_key].astype(int)
#         df["category"] = df.pid.str.split("_").str[-1]
#         df["parameters"] = df.pid.str.split("_").str[:-1].str.join("_")  # FIXME: Category names edge cases
#         df = df.sort_values(by=xp_key)
#         console.print(df)
#
#         # visulazation
#         f, axs = plt.subplots(nrows=len(df['parameters'].unique()),
#                               dpi=300, layout="constrained",
#                               figsize=(6, 4 * len(df['parameters'].unique())))
#
#         # plot the mean
#         for cat in df['category'].unique():
#             for ax, par in zip(axs, df['parameters'].unique()):
#                 ax.axhline(y=pars_true_icg[f"{par}_{cat}"].distribution.parameters['loc'],
#                            label=f"{par} (exact)", linestyle="--", color=colors[cat])
#
#                 df_cat = df[(df['category'] == cat) & (df['parameters'] == par)]
#                 ax.errorbar(
#                     x=df_cat[xp_key], y=df_cat["median"],
#                     yerr=df_cat["std"],
#                     # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
#                     label=cat,
#                     marker="o",
#                     color=colors[cat],
#                     # linestyle="",
#                     markeredgecolor="black",
#                 )
#
#                 ax.set_xlabel(xp_key)
#                 ax.set_ylabel(f"Parameter {par}")
#                 ax.legend()
#         f.savefig(RESULTS_ICG / f"xps_{xp_key}.png", bbox_inches="tight")
#
#         plt.show()
#
# def visualize_priors():
#     """Visualize the different priors."""
#
#     # data processing
#     df = pd.read_csv(RESULTS_ICG / f"xps_prior.tsv", sep="\t")
#     df["category"] = df.pid.str.split("_").str[-1]
#     df["prior"] = df.xp.str.split("_").str[-1]
#     df["parameters"] = df.pid.str.split("_").str[:-1].str.join("_") # FIXME: Category names edge cases
#
#     # visualization
#     from matplotlib import pyplot as plt
#     f, axs = plt.subplots(nrows=len(df['parameters'].unique()),
#                           dpi=300, layout="constrained",
#                           figsize=(6, 4*len(df['parameters'].unique())))
#
#     # plot the mean
#     for cat in df['category'].unique():
#         for ax, par in zip(axs, df['parameters'].unique()):
#             ax.axhline(y=pars_true_icg[f"{par}_{cat}"].distribution.parameters['loc'],
#                        label=f"{par} (exact)", linestyle="--", color=colors[cat])
#             for k, prior in enumerate(["exact", "biased"]):  # ["exact", "biased", "noprior"]
#                 # TODO: plot x parameters in different subplots
#                 df_cat = df[(df['prior'] == prior) &
#                             (df['category'] == cat) &
#                             (df['parameters'] == par)]
#                 console.print(df_cat)
#                 ax.errorbar(
#                     x=prior, y=df_cat["median"],
#                     yerr=df_cat["std"],
#                     # yerr=[df_cat["hdi_low"], df_cat["hdi_high"]],
#                     label=f"{par}_{cat}",
#                     marker="o",
#                     color=colors[cat],
#                     # linestyle="",
#                     markeredgecolor="black",
#                 )
#
#             # FIXME: boxplot
#             # ax.boxplot(df)
#
#
#
#             ax.set_xlabel("n (samples)")
#             ax.set_ylabel(f"Parameter {par}")
#             ax.legend()
#     plt.show()
#     f.savefig(RESULTS_ICG / f"xps_prior.png", bbox_inches="tight")
