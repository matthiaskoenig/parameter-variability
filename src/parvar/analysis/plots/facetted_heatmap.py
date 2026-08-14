import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from parvar import RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG
from parvar.analysis.utils import (
    point_bias,
    join_optimization_results,
    _sort_columns,
    _parse_and_flatten,
    _aggregate_arrays,
)


def facetted_heatmap(
    df,
    y_col,
    h_facet_col,
    v_cats,
    value_col,
    value_agg="mean",
    cmap="viridis",
    figsize=None,
    vmin=None,
    vmax=None,
    annot=True,
    fmt=".2f",
    title=None,
    cell_agg="mean",
    row_height=4,
    col_width=5,
):
    """
    Create a faceted heatmap with horizontal facets and alternating vertical facets.
    Handles a value column that contains string-encoded nested float arrays
    (e.g., "[[1.0, 2.0], [3.0, 4.0]]") by parsing, flattening, and aggregating.

    Three-stage pipeline:
    1. Parsing & Flattening (string → flat float array)
    2. Array-level aggregation (value_agg): flat array → scalar
    3. Cell-level aggregation (cell_agg): multiple scalars per cell → one value

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    y_col : str
        Column name for the shared y-axis categorical variable.
    h_facet_col : str
        Column name for horizontal faceting (defines columns of the subplot grid).
    v_cats : list of str
        List of column names (any length >= 1) that alternate as vertical facets
        (rows of subplot grid). Each becomes the x-axis in its respective row.
    value_col : str
        Column name for the numerical variable. If bias, each entry is a
        string representation of a float array, e.g., "[[1.0, 2.0], [3.0, 4.0]]".
        Else the array is assumed to be numerical.
    value_agg : str or callable, optional
        How to aggregate each flattened array into a single scalar value.
        If str, must be one of: 'mean', 'median', 'sum', 'min', 'max', 'std',
        'var', 'length', 'first', 'last', 'rms', 'percentile_25', 'percentile_75',
        'iqr', 'range'.
        If callable, should accept an array-like and return a scalar.
        Default is 'mean'.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated from row_height/col_width.
    vmin : float, optional
        Minimum value for color scale. If None, uses data min after aggregation.
    vmax : float, optional
        Maximum value for color scale. If None, uses data max after aggregation.
    annot : bool, optional
        Whether to annotate cells with numerical values. Default is True.
    fmt : str, optional
        Format string for annotations. Default is '.2f'.
    title : str, optional
        Overall figure title. Default is None.
    cell_agg : str or callable, optional
        Aggregation function for pivot_table (cell-level aggregation when multiple
        rows map to the same heatmap cell). Default is 'mean'.
    row_height : float, optional
        Height of each subplot row in inches. Default is 4.
    col_width : float, optional
        Width of each subplot column in inches. Default is 5.

    Returns
    -------
    fig, axes : matplotlib Figure and 2D array of Axes
    """
    assert len(v_cats) >= 1, "v_cats must contain at least 1 column name."
    if value_col == "bias":
        # --- Stage 1: Parse strings and flatten to flat float arrays ---
        df_parsed = _parse_and_flatten(df, "bayes_sampler_values")

        # --- Stage 2: Array-level aggregation (flat array → scalar) ---
        df_agg = _aggregate_arrays(df_parsed, "bayes_sampler_values", value_agg)

        # --- Proceed with the scalar aggregated column ---
        agg_col = "bias"
        df_agg[agg_col] = point_bias(df_agg, df_agg["bayes_sampler_values__scalar"])

    else:
        # No parsing needed — value_col already contains scalar floats
        df_agg = df.copy()
        agg_col = value_col

    h_facet_values = sorted(df_agg[h_facet_col].unique())
    n_cols = len(h_facet_values)
    n_rows = len(v_cats)

    if figsize is None:
        figsize = (col_width * n_cols + 2, row_height * n_rows)

    # Determine shared color scale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always 2D
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    im = None  # to keep reference for colorbar

    for row_idx, v_cat in enumerate(v_cats):
        for col_idx, h_val in enumerate(h_facet_values):
            ax = axes[row_idx, col_idx]

            # Filter data for this horizontal facet
            subset = df_agg[df_agg[h_facet_col] == h_val]

            # if vmin is None:
            #     vmin = subset[agg_col].min()
            # if vmax is None:
            #     vmax = subset[agg_col].max()

            # Pivot: y_col as index, current v_cat as columns
            pivot = subset.pivot_table(
                index=y_col, columns=v_cat, values=agg_col, aggfunc=cell_agg
            )

            # Sort for consistency
            pivot = pivot.sort_index()
            pivot = pivot[_sort_columns(pivot.columns)]

            facet_data = pivot.values[~np.isnan(pivot.values)]
            if len(facet_data) > 0:
                facet_vmin = facet_data.min()
                facet_vmax = facet_data.max()
            else:
                facet_vmin, facet_vmax = 0, 1

            # Plot heatmap
            data_matrix = pivot.values
            im = ax.imshow(
                data_matrix,
                aspect="auto",
                cmap=cmap,
                vmin=facet_vmin,
                vmax=facet_vmax,
                origin="lower",
            )

            # Set tick labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)

            # X-axis label
            ax.set_xlabel(v_cat, fontsize=9)

            # Y-axis label (only on leftmost column)
            if col_idx == 0:
                ax.set_ylabel(y_col)
            else:
                ax.set_yticklabels([])

            # Title for top row subplots (horizontal facet labels)
            if row_idx == 0:
                ax.set_title(f"{h_facet_col} = {h_val}", fontsize=11, fontweight="bold")

            # Row label on the right side
            # if col_idx == n_cols - 1:
            #     ax_right = ax.twinx()
            #     ax_right.set_yticks([])
            #     ax_right.set_ylabel(
            #         f"X: {v_cat}",
            #         rotation=270,
            #         labelpad=18,
            #         fontsize=10,
            #         fontstyle="italic",
            #     )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)

            # Add min/max labels to individual colorbars
            # cbar.ax.set_ylabel(
            #     f'[{facet_vmin:.1f}, {facet_vmax:.1f}]',
            #     fontsize=7, rotation=270, labelpad=12
            # )

            # Annotate cells
            if annot:
                for i in range(data_matrix.shape[0]):
                    for j in range(data_matrix.shape[1]):
                        val = data_matrix[i, j]
                        if not np.isnan(val):
                            norm_val = (
                                (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                            )
                            text_color = "white" if norm_val < 0.5 else "black"
                            ax.text(
                                j,
                                i,
                                f"{val:{fmt}}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=7,
                            )

    # Add shared colorbar
    # agg_label = value_agg if isinstance(value_agg, str) else value_agg.__name__
    # fig.subplots_adjust(right=0.88)
    # cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label(f"{value_col} ({agg_label})")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig, axes


# --- Example usage ---
if __name__ == "__main__":
    for r in [RESULTS_SIMPLE_CHAIN, RESULTS_SIMPLE_PK, RESULTS_ICG]:
        results = join_optimization_results(results_path=r, xp_type="all", server=True)
        fig, axes = facetted_heatmap(
            results,
            y_col="noise_cv",
            h_facet_col="prior_type",
            v_cats=["timepoints", "samples", "group"],
            value_col="bias",
            value_agg="mean",
            cell_agg="mean",
            cmap="RdYlGn_r",
            title="Mixed String Format Example",
            row_height=3,
            col_width=4,
        )

        plt.show()
