"""
Overview
--------
This module provides functions and classes for generating and analyzing
heatmaps of various statistical quantities derived from spectroscopic data,
as processed by the picca_bookkeeper pipeline.

Key Components:
- `HeatmapAnalysis`: Inherits from `ReadDeltas` (picca_bookkeeper/read_deltas.py)
                     and serves as the main data structure for analysis.
- `Plots`: Contains static methods to create heatmaps for variance, flux,
                    residuals, and other statistics, with options to save
                    outputs and customize visualization.

Main Usage:
- Used to visualize statistical summaries (e.g., variance, flux, residuals)
  across wavelength grids, typically for QA or scientific interpretation of
  pipeline outputs.
- Methods are designed to process the output arrays from `ReadDeltas`, which
  is central to data I/O in this repo.

Integration:
- Relies on `ReadDeltas` for data loading and preprocessing.
- Uses hints from `picca_bookkeeper/hints.py` for type annotations.
- Designed to be called by higher-level scripts or Jupyter notebooks
  that orchestrate analysis and plotting.

Related Modules:
- `picca_bookkeeper/read_deltas.py`: Data reading and basic statistics.
- `picca_bookkeeper/hints.py`: Type definitions for plotting.
- Other analysis scripts and plotting modules may use this file for
  visualizations.

To use this code, ensure you have processed data via `ReadDeltas` and pass the
resulting objects to the static methods in `Plots` for visualization and
further analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from picca_bookkeeper.read_deltas import ReadDeltas

if TYPE_CHECKING:
    from typing import Dict, Optional, Tuple, Type

    from picca_bookkeeper.hints import Axes, Figure


class HeatmapAnalysis(ReadDeltas):
    """
    Thin wrapper around `ReadDeltas` used for heatmap-based analysis.

    Inherits from `ReadDeltas` and serves as a semantic placeholder to
    indicate that this analysis focuses on generating 2D statistical
    visualizations (heatmaps) from the processed delta field data.

    This class doesn't introduce new functionality but helps organize
    plotting logic under the context of statistical heatmap summaries.
    """
    pass


class Plots:
    """
    Collection of static plotting utilities to visualize statistical summaries
    (e.g., variance, residuals) as heatmaps. Works with `ReadDeltas` or
    `HeatmapAnalysis` objects.

    Each method is a standalone plotting function with options for:
    - matplotlib figure / axes injection
    - saving plots to file
    - saving computed heatmap data for reuse

    Typical Usage:
    >>> Plots.var_lss(analysis_obj, save_plot=True, output_prefix="output/plot")
    """

    @staticmethod
    def var_lss(
        analysis: Type[ReadDeltas],
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a 2D heatmap of the quantity:
        VAR - η ⋅ σ_pip^2 (residual between observed variance and pipeline
                           prediction).

        Parameters
        ----------
        analysis : ReadDeltas
            The data container holding delta arrays, weights, lambda grid, etc.
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure object to plot into.
            Required if `ax` is also provided.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot into.
        y_min : float, optional
            Minimum value on y-axis; used to limit heatmap range.
        y_max : float, optional
            Maximum value on y-axis; used to limit heatmap range.
        add_mean_line : bool, default=False
            Whether to overlay the mean y-value at each x-bin.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask applied to select pixels/forests.
        output_prefix : str or Path, optional
            Path prefix for saved output files (e.g., "plots/var_lss").
        save_data : bool, default=False
            Whether to save histogram data (.npz).
        save_plot : bool, default=False
            Whether to save the rendered heatmap as an image (.png).
        save_dict : dict, optional
            Extra data to include in the saved .npz output.

        Raises
        ------
        ValueError
            If saving is requested but `output_prefix` is not provided.

        Notes
        -----
        The 2D histogram is computed using λ and the residual variance statistic
        across all forests. Used for QA or evaluating model misfits.
        """
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")

        if save_data:
            data_dict = {}

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        n_forests = len(analysis.deltas2_arr)

        heatmap, xedges, yedges = np.histogram2d(
            np.concatenate([analysis.lambda_grid[mask[i]]
                           for i in range(n_forests)]),
            (
                1 / analysis.weights_arr[mask]
                - (analysis.eta * analysis.var_pipe_arr)[mask]
            ),
            bins=776,
        )

        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        if save_data:
            data_dict["histogram-heatmap"] = heatmap
            data_dict["histogram-extent"] = extent

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        im = ax.imshow(
            heatmap.T,
            extent=extent,
            **{
                **dict(origin="lower", aspect="auto"),
                **imshow_kwargs,
            },
        )

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$VAR-\eta \cdot \sigma_{\rm pip}^2$")
        ax.set_title(
            r"Heatmap for values of $VAR - \eta \cdot \sigma_{\rm pip}^2$")

        fig.colorbar(im, cax=cax, orientation="vertical")

        if add_mean_line:
            ycenters = 0.5 * (yedges[1:] + yedges[:-1])
            averages = []
            for i in range(len(heatmap)):
                try:
                    averages.append(np.average(ycenters, weights=heatmap[i]))
                except ZeroDivisionError:
                    averages.append(np.nan)

            ax.plot(
                0.5 * (xedges[:-1] + xedges[1:]),
                averages,
                **{
                    **dict(color="white", lw=1),
                    **mean_line_kwargs,
                },
            )

            if save_data:
                data_dict["mean_line"] = (
                    0.5 * (xedges[:-1] + xedges[1:]), averages)

        plt.tight_layout()

        if output_prefix is not None:
            if save_plot:
                output_prefix = Path(output_prefix)
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + "-var_lss.png"),
                    dpi=300,
                )

            if save_data:
                output_prefix = Path(output_prefix)
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + "-var_lss.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

    @staticmethod
    def heatmap_stat(
        analysis: Type[ReadDeltas],
        x_values: np.ndarray,
        y_stat: np.ndarray,
        bins: Tuple[int, int] | Tuple[np.ndarray, int] = (200, 777),
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Arguments
        ---------
            bins (array): Which bins to use for the x-axis heatmap.
                        (Default: (200, 777))
            use_weights (bool): Use weights instead of counts for the heatmap.
                        (Default: False)
        """
        """
        Generic 2D heatmap plotting method for arbitrary (x, y) statistics.

        Main Arguments
        ----------
        bins : tuple, default=(200, 777)
            Binning strategy. Can be (n_xbins, n_ybins) or (x_edges, n_ybins).
        use_weights : bool, default=False
            Whether to weight the histogram by `analysis.weights_arr`.
            Use weights instead of counts for the heatmap.

        Additional Parameters
        ----------
        analysis : ReadDeltas
            Data container used for weights and masking.
        x_values : np.ndarray
            Values along the x-axis (e.g., λ or z or another variable).
        y_stat : np.ndarray
            Values to plot along the y-axis (e.g., variance, residuals, etc.)
        fig : matplotlib.figure.Figure, optional
            Figure object for custom plotting.
        ax : matplotlib.axes.Axes, optional
            Axes object for custom plotting.
        y_min : float, optional
            Minimum y-axis value (overrides auto-scaling).
        y_max : float, optional
            Maximum y-axis value (overrides auto-scaling).
        add_mean_line : bool, default=False
            Whether to overlay the average y-value per x-bin.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask for selecting subset of data.
        output_prefix : str or Path, optional
            If saving, this prefix is used for file outputs.
        save_data : bool, default=False
            Whether to save histogram and mean line data (.npz).
        save_plot : bool, default=False
            Whether to save the plotted heatmap (.png).
        save_dict : dict, optional
            Extra data to include when saving to .npz.

        Raises
        ------
        ValueError
            If saving is requested but `output_prefix` is not provided.

        Notes
        -----
        This method is flexible and allows plotting any derived statistic
        versus x-axis values. Used for visualizing correlations, residuals, etc.
        The y-axis is symmetrically centered around the weighted mean.
        """
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")

        if save_data:
            data_dict = {}

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fiug should be provided at the same time")

        n_forests = len(analysis.deltas2_arr)
        n_bins = bins[1]
        assert n_bins % 2  # This is to be able to define y_edges around mean properly

        y_center = np.average(y_stat, weights=analysis.weights_arr[mask])
        if y_min is None:
            y_min = min(
                np.percentile(y_stat, 0.1),
                y_center -
                np.sqrt(np.cov(y_stat, aweights=analysis.weights_arr[mask])),
            )
        if y_max is None:
            y_max = max(
                np.percentile(y_stat, 99.9),
                y_center +
                np.sqrt(np.cov(y_stat, aweights=analysis.weights_arr[mask])),
            )
        y_delta = (y_max - y_min) / (n_bins + 1)

        y_edges = np.concatenate(
            (
                np.flip(np.arange(y_center - y_delta / 2, y_min, -y_delta)),
                np.arange(y_center + y_delta / 2, y_max, y_delta),
            )
        )

        if use_weights:
            weights = analysis.weights_arr[mask]
        else:
            weights = None
        heatmap, xedges, yedges = np.histogram2d(  # type: ignore
            x_values,
            y_stat,
            bins=(bins[0], y_edges),
            weights=weights,
            normed=False,
        )

        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        if save_data:
            data_dict["histogram-heatmap"] = heatmap
            data_dict["histogram-extent"] = extent

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        im = ax.imshow(
            heatmap.T,
            extent=extent,
            **{
                **dict(origin="lower", aspect="auto"),
                **imshow_kwargs,
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")

        if add_mean_line:
            ycenters = 0.5 * (yedges[1:] + yedges[:-1])
            averages = []
            for i in range(len(heatmap)):
                try:
                    averages.append(np.average(ycenters, weights=heatmap[i]))
                except ZeroDivisionError:
                    averages.append(np.nan)

            ax.plot(
                0.5 * (xedges[:-1] + xedges[1:]),
                averages,
                **{
                    **dict(color="white", lw=1),
                    **mean_line_kwargs,
                },
            )

            if save_data:
                data_dict["mean_line"] = (
                    0.5 * (xedges[:-1] + xedges[1:]), averages)

        if output_prefix is not None:
            if save_plot:
                output_prefix = Path(output_prefix)
                plt.tight_layout()
                plt.savefig(
                    str(output_prefix) + ".png",
                    dpi=300,
                )

            if save_data:
                np.savez(
                    str(output_prefix) + ".npz",
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

    @staticmethod
    def flux(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap of log10(flux) as a function of observed wavelength.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-flux"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        Plots.heatmap_stat(
            analysis,
            x_values=np.concatenate(
                [
                    analysis.lambda_grid[mask[i]]
                    for i in range(len(analysis.deltas2_arr))
                ]
            ),
            y_stat=np.log10(analysis.flux_arr[mask]),
            use_weights=use_weights,
            bins=(analysis.lambda_edges, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            mask=mask if mask is not None else None,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Flux")
        ax.set_title(r"Heatmap for values of flux")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def flux_var(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap of log10(flux variance) as a function of observed
        wavelength.

        This visualizes the inverse inverse variance of the flux array.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-flux_var"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        Plots.heatmap_stat(
            analysis,
            x_values=np.concatenate(
                [
                    analysis.lambda_grid[mask[i]]
                    for i in range(len(analysis.deltas2_arr))
                ]
            ),
            y_stat=np.log10(1 / (analysis.flux_ivar_arr[mask])),
            use_weights=use_weights,
            bins=(analysis.lambda_edges, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\log(\sigma_{\rm pip})$")
        ax.set_title(r"Heatmap for values of $\sigma_{\rm pip}$")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def flux_var_over_flux(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap of log10(flux variance / flux) vs wavelength.

        This can be used as a diagnostic of noise relative to signal strength.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None

        Notes
        -----
        The plotted quantity is:
            log10(1 / (ivar × flux)) = log10(variance / flux)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-flux_var"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        y_stat = np.log10(
            1 / (analysis.flux_ivar_arr[mask] * analysis.flux_arr[mask]))
        # y_stat=1/(analysis.flux_ivar_arr[mask]*analysis.flux_arr[mask])
        Plots.heatmap_stat(
            analysis,
            x_values=np.concatenate(
                [
                    analysis.lambda_grid[mask[i]]
                    for i in range(len(analysis.deltas2_arr))
                ]
            ),
            y_stat=y_stat,  # np.log10(1 / analysis.flux_ivar_arr[mask]),
            use_weights=use_weights,
            bins=(analysis.lambda_edges, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\log(\sigma_{\rm pip}/Flux)")
        ax.set_title(r"Heatmap for values of $\sigma_{\rm pip}$/Flux")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def pipe_var(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap of log10(pipe-estimated flux variance) vs wavelength.

        Uses `var_pipe_arr` from the pipeline to generate the heatmap.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-pipe_var"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        Plots.heatmap_stat(
            analysis,
            x_values=np.concatenate(
                [
                    analysis.lambda_grid[mask[i]]
                    for i in range(len(analysis.deltas2_arr))
                ]
            ),
            y_stat=np.log10(analysis.var_pipe_arr[mask]),
            use_weights=use_weights,
            bins=(analysis.lambda_edges, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\log(\tilde{\sigma}_{\rm pip})$")
        ax.set_title(r"Heatmap for values of $\tilde{\sigma}_{\rm pip}$")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def flux_var_rf(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap of log10(flux variance) vs rest-frame wavelength.

        Useful for checking how noise properties vary in the quasar rest frame.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
            Note: x-axis is "lambda_rf_arr," rather than lambda grid.
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
            Note: x-axis is "lambda_rf_arr," rather than lambda grid.
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-flux-_var_rf"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        Plots.heatmap_stat(
            analysis,
            x_values=analysis.lambda_rf_arr[mask],
            y_stat=np.log10(1 / analysis.flux_ivar_arr[mask]),
            use_weights=use_weights,
            bins=(200, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\log(\sigma_{\rm pip})$")
        ax.set_title(r"Heatmap for values of $\sigma_{\rm pip}$")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def lambda_vs_lambda_rf(
        analysis: Type[ReadDeltas],
        use_weights: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        add_mean_line: bool = False,
        imshow_kwargs: Dict = dict(),
        mean_line_kwargs: Dict = dict(),
        mask: Optional[np.ndarray] = None,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot a heatmap comparing observed wavelength vs rest-frame
        wavelength.

        Helps verify redshift conversion and assess completeness across
        wavelength.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object containing flux and wavelength information.
            Note: x-axis is "rest-frame wavelength."
        use_weights : bool, optional
            Whether to weight each sample by its pipeline weight.
        fig : matplotlib.figure.Figure, optional
            Figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Must be provided with `fig` if not None.
            Note: x-axis is "rest-frame wavelength."
        y_min : float, optional
            Minimum y-axis value for the plot.
        y_max : float, optional
            Maximum y-axis value for the plot.
        add_mean_line : bool, optional
            Whether to overlay a mean line on the heatmap.
        imshow_kwargs : dict, optional
            Additional kwargs passed to `imshow()` for customization.
            For example:
                imshow_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        mean_line_kwargs : dict, optional
            Additional kwargs for `plot()` when drawing the mean line.
            For example:
                mean_line_kwargs = {"color": "black",
                                    "linestyle": "--",
                                    "linewidth": 1.5,
                                    "label": "Mean"}
        mask : np.ndarray, optional
            Boolean mask array for selecting valid data points.
        output_prefix : str or Path, optional
            Prefix for saving outputs (plot or data).
        save_data : bool, optional
            Whether to save the processed data array.
        save_plot : bool, optional
            Whether to save the resulting plot as a PNG.
        save_dict : dict, optional
            Extra metadata to save alongside the data if `save_data` is True.

        Returns
        -------
        None

        Notes
        -----
        X-axis is observed wavelength (`lambda_grid`),
        Y-axis is rest-frame wavelength (`lambda_rf_arr`).
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix).parent / (
                Path(output_prefix).name + "-lambda_vs_lambda_rf"
            )

        if mask is None:
            if hasattr(analysis, "msk"):
                mask = analysis.msk
            else:
                mask = np.full_like(analysis.deltas2_arr, True, dtype=bool)

        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fig should be provided at the same time")

        Plots.heatmap_stat(
            analysis,
            x_values=np.concatenate(
                [
                    analysis.lambda_grid[mask[i]]
                    for i in range(len(analysis.deltas2_arr))
                ]
            ),
            y_stat=analysis.lambda_rf_arr[mask],
            use_weights=use_weights,
            bins=(analysis.lambda_edges, 777),
            fig=fig,
            ax=ax,
            y_min=y_min,
            y_max=y_max,
            add_mean_line=add_mean_line,
            imshow_kwargs=imshow_kwargs,
            mean_line_kwargs=mean_line_kwargs,
            output_prefix=output_prefix,
            save_data=save_data,
            save_dict=save_dict,
        )

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\lambda_{\rm RF}$")

        if save_plot:
            plt.tight_layout()
            plt.savefig(
                str(output_prefix) + ".png",
                dpi=300,
            )

    @staticmethod
    def var_residual(
        analysis: Type[ReadDeltas],
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
    ) -> None:
        """
        Placeholder for future implementation: plot residual variance.

        Currently does nothing.

        Parameters
        ----------
        analysis : Type[ReadDeltas]
            Data object (not used yet).
        fig : matplotlib.figure.Figure, optional
            Figure object (not used yet).
        ax : matplotlib.axes.Axes, optional
            Axes object (not used yet).

        Returns
        -------
        None
        """
        pass
