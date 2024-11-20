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
    pass


class Plots:
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
            np.concatenate([analysis.lambda_grid[mask[i]] for i in range(n_forests)]),
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
        ax.set_title(r"Heatmap for values of $VAR - \eta \cdot \sigma_{\rm pip}^2$")

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
                data_dict["mean_line"] = (0.5 * (xedges[:-1] + xedges[1:]), averages)

        plt.tight_layout()

        if output_prefix is not None:
            if save_plot:
                output_prefix = Path(output_prefix)
                plt.savefig(
                    output_prefix.parent / (output_prefix.name + "-var_lss.png"),
                    dpi=300,
                )

            if save_data:
                output_prefix = Path(output_prefix)
                np.savez(
                    output_prefix.parent / (output_prefix.name + "-var_lss.npz"),
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
            bins (array): Which bins to use for the x-axis heatmap. (Default: (200, 777))
            use_weights (bool): Use weights instead of counts for the heatmap. (Default: False)
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
                y_center - np.sqrt(np.cov(y_stat, aweights=analysis.weights_arr[mask])),
            )
        if y_max is None:
            y_max = max(
                np.percentile(y_stat, 99.9),
                y_center + np.sqrt(np.cov(y_stat, aweights=analysis.weights_arr[mask])),
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
                data_dict["mean_line"] = (0.5 * (xedges[:-1] + xedges[1:]), averages)

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

        y_stat = np.log10(1 / (analysis.flux_ivar_arr[mask] * analysis.flux_arr[mask]))
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
        pass
