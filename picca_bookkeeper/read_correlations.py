"""
This module provides tools for reading, processing, and plotting correlation
function data in the picca_bookkeeper package.

It defines the CorrelationPlots class, which includes static methods for
loading correlation data (including auto- and cross-correlations) from FITS files,
computing statistics, and generating visualizations.

The correlation data is typically organized by region, absorber type, and tracer,
and may be auto-correlation or cross-correlation, as specified by parameters.

Interaction with other codes:

 - Relies on the Bookkeeper object to locate relevant correlation files
   within the repository.
 - Uses the Wedge class to bin and process the correlation data.
 - Designed to work with other modules in picca_bookkeeper that manage
   data paths and metadata.
 - Typical usage is to call CorrelationPlots methods with a Bookkeeper and
   appropriate parameters to extract and visualize correlation results for
   cosmological analyses.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vega.plots.wedges import Wedge

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Dict, Optional, Tuple

    from picca_bookkeeper.hints import Axes, Figure


class CorrelationPlots:
    @staticmethod
    def cf(
        bookkeeper: Optional[Bookkeeper],
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        correlation_file: Path | str = "",
        mumin: float = 0,
        mumax: float = 1,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot correlation function as a wedge.

        Parameters
        ----------
        bookkeeper : Optional[Bookkeeper]
            Bookkeeper object used to locate the correlation file / collect data.
        region : str, default="lya"
            Region identifier for the first dataset.
        region2 : Optional[str], default=None
            Region identifier for the second dataset (if cross-correlations).
        absorber : str, default="lya"
            Absorber type for the first dataset.
        absorber2 : Optional[str], default=None
            Absorber type for the second dataset (if cross-correlations).
        correlation_file : Path or str, optional
            Provide path to a correlation file (overrides Bookkeeper object).
        mumin : float, default=0
            Minimum mu value for the wedge.
        mumax : float, default=1
            Maximum mu value for the wedge.
        ax : matplotlib Axes, optional
            Axes to plot on. Created if None.
        r_factor : int, default=2
            Exponenential factor applied to distance r for visual scaling.
        plot_kwargs : dict, optional
            Extra arguments passed to `ax.errorbar()`.
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, default=False
            If True, skip plotting and return computed arrays.
        output_prefix : Path or str, optional
            File structure / prefix used for saving outputs.
        save_data : bool, default=False
            Whether to save the computed values to a .npz file.
        save_plot : bool, default=False
            Whether to save the figure to as a .png file.
        save_dict : dict, optional
            Extra metadata to store in the output .npz file.

        Returns
        -------
        r : ndarray
            Radial distances (bin centers).
        values : ndarray
            Scaled correlation values r^r_factor ξ(r).
        errors : ndarray
            Errors on correlation values.
        nb : ndarray
            Number of contributing pairs per bin.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if isinstance(bookkeeper, Bookkeeper):
            correlation_file = bookkeeper.paths.exp_cf_fname(
                absorber, region, absorber2, region2
            )

        with fitsio.FITS(correlation_file) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            wedge = Wedge(
                rp=(cor_header["RPMIN"],
                    cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0),
                    cor_header["RTMAX"], cor_header["NT"]),
                r=(cor_header.get("RTMIN", 0),
                   cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
            )
        data_wedge = wedge(da, co)

        # Computing nb wedge manually
        axis_sum = wedge.weights.sum(axis=0)
        w = axis_sum > 0
        nb_weights = np.copy(wedge.weights)
        nb_weights[:, w] /= wedge.weights.sum(axis=0)[w]
        nb_wedge = data_wedge[0], nb_weights.dot(nb)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return (
                data_wedge[0],
                r_coef * data_wedge[1],
                r_coef * np.sqrt(np.diag(data_wedge[2])),
                nb_wedge[1],
            )

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.errorbar(
            data_wedge[0],
            r_coef * data_wedge[1],
            yerr=r_coef * np.sqrt(np.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * data_wedge[1]
            data_dict["errors"] = r_coef * np.sqrt(np.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = nb_wedge[1]
            # data_dict['wedge_data'] = data_wedge

        plt.tight_layout()

        if output_prefix is not None:
            if save_plot:
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + "-plot_cf.png"),
                )
            if save_data:
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + "-plot_cf.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

        return (
            data_wedge[0],
            r_coef * data_wedge[1],
            r_coef * np.sqrt(np.diag(data_wedge[2])),
            nb_wedge[1],
        )

    @staticmethod
    def rp(
        bookkeeper: Optional[Bookkeeper],
        region: str = "lya",
        auto: Optional[bool] = False,
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        correlation_file: Path | str = "",
        rtmin: float = 0,
        rtmax: float = 4,
        rebin: Optional[int] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
        tracer: str = "qso",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot correlation function in a wedge, averaged over a transverse bin.

        Parameters
        ----------
        bookkeeper : Optional[Bookkeeper]
            Bookkeeper object used to locate the correlation file / collect data.
        region : str, default="lya"
            Region identifier for the first dataset.
        auto : bool, default=False
            Whether to plot auto-correlation (True) or cross-correlation (False).
        region2 : Optional[str], default=None
            Region identifier for the second dataset (if cross-correlations).
        absorber : str, default="lya"
            Absorber type for the first dataset.
        absorber2 : Optional[str], default=None
            Absorber type for the second dataset, if applicable.
        correlation_file : Path or str, optional
            Provide path to a correlation file (overrides Bookkeeper object).
        rtmin : float, default=0
            Minimum transverse distance in Mpc/h.
        rtmax : float, default=4
            Maximum transverse distance in Mpc/h.
        rebin : Optional[int], optional
            If specified, average over `rebin` number of bins.
        ax : matplotlib Axes, optional
            Axes to plot on. Created if None.
        plot_kwargs : dict, optional
            Extra arguments passed to `ax.errorbar()`.
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, default=False
            If True, skip plotting and return computed arrays.
        output_prefix : Path or str, optional
            File prefix used for saving outputs.
        save_data : bool, default=False
            Whether to save the computed values to a .npz file.
        save_plot : bool, default=False
            Whether to save the figure to as a .png.
        save_dict : dict, optional
            Extra metadata to store in the output .npz file.
        tracer : str, default="qso"
            Tracer type used in cross-correlations.

        Returns
        -------
        rp : ndarray
            Line-of-sight distances (r_parallel).
        data : ndarray
            Averaged correlation values over r_perp.
        error : ndarray
            Corresponding errors on coorelation xi(r_parallel).
        nb : ndarray
            Number of contributing pairs per r_parallel bin.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if isinstance(bookkeeper, Bookkeeper):
            if auto:
                correlation_file = bookkeeper.paths.exp_cf_fname(
                    absorber, region, absorber2, region2
                )
                name = "cf_rp"
            else:
                correlation_file = bookkeeper.paths.exp_xcf_fname(
                    absorber, region, tracer
                )
                name = "xcf_rp"

        with fitsio.FITS(correlation_file) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()
            nrp, nrt = cor_header["NP"], cor_header["NT"]

        mat = da.reshape(nrp, nrt)
        errmat = np.sqrt(np.diag(co)).reshape(nrp, nrt)
        nmat = nb.reshape(nrp, nrt)

        weights = np.copy(1 / errmat**2)  # 1 / errmat**2

        rt = rt.reshape(nrp, nrt)
        rp = rp.reshape(nrp, nrt)

        w = (rt >= rtmin) & (rt <= rtmax)
        weights[~w] = 0

        if rebin is not None:
            if nrp % rebin != 0:
                raise ValueError("NP should be divisible by rebin, NP: ", nrt)

            mat = mat.reshape(nrp // rebin, -1)
            errmat = errmat.reshape(nrp // rebin, -1)
            nmat = nmat.reshape(nrp // rebin, -1)
            rp = rp.reshape(nrp // rebin, -1)
            weights = weights.reshape(nrp // rebin, -1)

        rp = np.average(rp, weights=weights, axis=1)
        data = np.average(mat, weights=weights, axis=1)
        error = np.sqrt(np.average(errmat**2, weights=weights**2, axis=1))
        nb = np.sum(nmat, axis=1)

        if just_return_values:
            return (rp, data, error, nb)

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.errorbar(
            rp,
            data,
            yerr=error,
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_title(
            rf"${rtmin} < r_\perp < {rtmax} \, \, [\mathrm{{Mpc \, h^{{-1}}}}]$"
        )
        ax.set_xlabel(r"$r_\parallel  \, [\mathrm{{Mpc \, h^{{-1}}  }}]$")
        ax.set_ylabel(r"$\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$")
        if save_data:
            data_dict["rp"] = rp
            data_dict["data"] = data
            data_dict["errors"] = error
            data_dict["nb"] = nb
            # data_dict['wedge_data'] = data_wedge

        plt.tight_layout()

        if output_prefix is not None:
            if save_plot:
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + f"-plot_{name}.png"),
                )
            if save_data:
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + f"-plot_{name}.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

        return (rp, data, error, nb)

    @staticmethod
    def compare_cfs(
        bkp1: Bookkeeper,
        bkp2: Optional[Bookkeeper] = None,
        mumin: float = 0,
        mumax: float = 1,
        region: str = "lya",
        region2: Optional[str] = None,
        correlation_file: Path | str = "",
        correlation_file2: Path | str = "",
        label: Optional[str] = None,
        label2: Optional[str] = None,
    ) -> None:
        """
        Compare two correlation functions (CFs) by plotting them and
        their differences.

        Parameters
        ----------
        bkp1 : Bookkeeper
            First Bookkeeper instance to compare.
        bkp2 : Bookkeeper, optional
            Second Bookkeeper instance to compare.
            If None, a single CF is plotted.
        mumin : float, optional
            Minimum value of the μ wedge.
            Default is 0.
        mumax : float, optional
            Maximum value of the μ wedge.
            Default is 1.
        region : str, optional
            First region to use for correlation.
            Default is "lya".
        region2 : str, optional
            Optional second region for the cross-correlation.
        correlation_file : str or Path, optional
            File path for the first correlation function.
            If empty, inferred from Bookkeeper.
        correlation_file2 : str or Path, optional
            File path for the second correlation function.
            If empty, inferred from Bookkeeper.
        label : str, optional
            Label for the first correlation function in the plot.
        label2 : str, optional
            Label for the second correlation function in the plot.

        Notes
        -----
        Creates a three-panel plot:
            1. Both CFs
            2. Their difference
            3. Ratio of their errors
        """
        fig, axs = plt.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
        )

        if label is None and isinstance(bkp1, Bookkeeper):
            label = bkp1.label
        if label2 is None and isinstance(bkp2, Bookkeeper):
            label2 = bkp2.label

        ax = axs[0]
        a = CorrelationPlots.cf(
            bkp1,
            region=region,
            region2=region2,
            r_factor=2,
            ax=ax,
            correlation_file=correlation_file,
            mumin=mumin,
            mumax=mumax,
            plot_kwargs=dict(label=label),
        )
        b = CorrelationPlots.cf(
            bkp2,
            region=region,
            region2=region2,
            correlation_file=correlation_file2,
            r_factor=2,
            ax=ax,
            mumin=mumin,
            mumax=mumax,
            plot_kwargs=dict(label=label2),
        )
        ax.grid(visible=True)
        #    ax.set_ylim(-1, 1)
        ax.legend()

        ax = axs[1]
        ax.plot(a[0], b[1] - a[1], c="k")
        ax.set_xlabel(axs[0].get_xlabel())
        ax.set_ylabel(f"{label2}-{label}")
        ax.grid(visible=True)
        ax.set_xlabel(None)

        ax = axs[2]
        ax.plot(a[0], b[2] / a[2], c="k")
        ax.set_xlabel(axs[0].get_xlabel())
        ax.set_ylabel(f"errors{label2}/{label}")
        ax.grid(visible=True)

        # axs[1].set_ylim(0.85, 1.15)
        axs[0].set_xlabel(None)

    @staticmethod
    def compare_xcfs(
        bkp1: Bookkeeper,
        bkp2: Optional[Bookkeeper] = None,
        mumin: float = 0,
        mumax: float = 1,
        region: str = "lya",
        correlation_file: str = "",
        correlation_file2: str = "",
        label: Optional[str] = None,
        label2: Optional[str] = None,
    ) -> None:
        """
        Compare two cross-correlation functions (XCFs) visually and
        quantitatively.

        Parameters
        ----------
        bkp1 : Bookkeeper
            First Bookkeeper instance to compare.
        bkp2 : Bookkeeper, optional
            Second Bookkeeper instance to compare.
            If None, a single XCF is plotted.
        mumin : float, optional
            Minimum μ value for wedge selection.
            Default is 0.
        mumax : float, optional
            Maximum μ value for wedge selection.
            Default is 1.
        region : str, optional
            Region to use for cross-correlation.
            Default is "lya".
        correlation_file : str, optional
            Path to first XCF file.
            If empty, inferred from Bookkeeper.
        correlation_file2 : str, optional
            Path to second XCF file.
            If empty, inferred from Bookkeeper.
        label : str, optional
            Label for the first XCF.
        label2 : str, optional
            Label for the second XCF.

        Notes
        -----
        Generates three subplots:
            1. Overlaid XCFs
            2. Difference between them
            3. Error ratio
        """
        fig, axs = plt.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
        )

        if label is None and isinstance(bkp1, Bookkeeper):
            label = bkp1.label
        if label2 is None and isinstance(bkp2, Bookkeeper):
            label2 = bkp2.label

        ax = axs[0]
        a = CorrelationPlots.xcf(
            bkp1,
            region=region,
            r_factor=2,
            ax=ax,
            correlation_file=correlation_file,
            mumin=mumin,
            mumax=mumax,
            plot_kwargs=dict(label=label),
        )
        b = CorrelationPlots.xcf(
            bkp2,
            region=region,
            r_factor=2,
            ax=ax,
            correlation_file=correlation_file2,
            mumin=mumin,
            mumax=mumax,
            plot_kwargs=dict(label=label2),
        )
        ax.grid(visible=True)
        ax.set_ylim(-1, 1)
        ax.legend()

        ax = axs[1]
        ax.plot(a[0], b[1] - a[1], c="k")
        ax.set_xlabel(axs[0].get_xlabel())
        ax.set_ylabel(f"{label2}-{label}")
        ax.grid(visible=True)
        ax.set_xlabel(None)

        ax = axs[2]
        ax.plot(a[0], b[2] / a[2], c="k")
        ax.set_xlabel(axs[0].get_xlabel())
        ax.set_ylabel(f"errors{label2}/{label}")
        ax.grid(visible=True)

        # axs[1].set_ylim(0.85, 1.15)
        axs[0].set_xlabel(None)

    @staticmethod
    def xcf(
        bookkeeper: Optional[Bookkeeper],
        region: str = "lya",
        absorber: str = "lya",
        mumin: float = 0,
        mumax: float = 1,
        abs_mu: bool = True,
        correlation_file: Path | str = "",
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
        tracer: str = "qso",
    ) -> Tuple[np.ndarray, ...]:
        """
        Plot a cross-correlation function (XCF) in a given μ wedge.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Object that provides correlation file paths and collects data.
        region : str, optional
            Region used for cross-correlation.
            Default is "lya".
        absorber : str, optional
            Absorber species. Default is "lya".
        mumin : float, optional
            Minimum μ value for the wedge.
            Default is 0.
        mumax : float, optional
            Maximum μ value for the wedge.
            Default is 1.
        abs_mu : bool, optional
            Whether to take absolute value of μ in wedge selection.
            Default is True.
        correlation_file : str or Path, optional
            Path to the correlation file.
            If not provided, it is inferred from `bookkeeper`.
        ax : matplotlib.axes.Axes, optional
            Axis to draw the plot on. If None, a new one is created.
        r_factor : int, optional
            Exponent factor applied to distance r when scaling correlation.
            Default is 2.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to 'errorbar().'
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, optional
            If True, only returns data arrays without plotting.
            Default is False.
        output_prefix : str or Path, optional
            Prefix for saving data and plots. Required if `save_data`
            or `save_plot` is True. (Default: None, plot not saved)
        save_data : bool, optional
            Whether to save the data to .npz file under the output_prefix file
            structure. Default is False.
        save_plot : bool, optional
            Whether to save the plot to .png file.
            Default is False.
        save_dict : dict, optional
            Additional metadata to include when saving data in the npz file
            if save_data option is True. Default is empty dict.
        tracer : str, optional
            Tracer type used in XCF. Default is "qso".

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple containing:
                - r bin centers,
                - scaled correlation values,
                - scaled errors,
                - effective number of pairs (nb)

        Raises
        ------
        ValueError
            If saving is requested but no `output_prefix` is provided.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if isinstance(bookkeeper, Bookkeeper):
            correlation_file = bookkeeper.paths.exp_xcf_fname(
                absorber, region, tracer)

        with fitsio.FITS(correlation_file) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            wedge = Wedge(
                rp=(cor_header["RPMIN"],
                    cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0),
                    cor_header["RTMAX"], cor_header["NT"]),
                r=(cor_header.get("RTMIN", 0),
                   cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
                abs_mu=abs_mu,
            )
        data_wedge = wedge(da, co)

        # Computing nb wedge manually
        axis_sum = wedge.weights.sum(axis=0)
        w = axis_sum > 0
        nb_weights = np.copy(wedge.weights)
        nb_weights[:, w] /= wedge.weights.sum(axis=0)[w]
        nb_wedge = data_wedge[0], nb_weights.dot(nb)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return (
                data_wedge[0],
                r_coef * data_wedge[1],
                r_coef * np.sqrt(np.diag(data_wedge[2])),
                nb_wedge[1],
            )

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.errorbar(
            data_wedge[0],
            r_coef * data_wedge[1],
            yerr=r_coef * np.sqrt(np.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * data_wedge[1]
            data_dict["errors"] = r_coef * np.sqrt(np.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = nb_wedge[1]
            # data_dict['wedge_data'] = data_wedge

        plt.tight_layout()

        if output_prefix is not None:
            if save_plot:
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + "-plot_xcf.png"),
                )
            if save_data:
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + "-plot_xcf.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

        return (
            data_wedge[0],
            r_coef * data_wedge[1],
            r_coef * np.sqrt(np.diag(data_wedge[2])),
            nb_wedge[1],
        )

    # @staticmethod
    # def multiple_cf(
    #     bookkeepers: List[Bookkeeper],
    #     region: str,
    #     labels: List[str] = None,
    #     region2: str = None,
    #     absorber: str = "lya",
    #     absorber2: str = None,
    #     correlation_file: Path | str = None,
    #     mumin: float = 0,
    #     mumax: float = 1,
    #     ax: Optional[Axes] = None,
    #     r_factor: int = 2,
    #     plot_kwargs: Dict = dict(),
    #     just_return_values: bool = False,
    #     output_prefix: Path | str = None,
    #     save_data: bool = False,
    #     save_plot: bool = False,
    #     save_dict: Dict = None,
    # ):
    #     """
    #     Plotting correlation function in defined wedge.

    #     Arguments:
    #     ----------
    #     bookkeepers: Bookkeepers object to use for collecting data.
    #     region: region to compute correlations from.
    #     labels: Labels to use for the different bookkeeper realizations.
    #     region2: region to compute correlations from if multiple regions have been used.
    #     mumin: wedge min value.
    #     mumax: wedge max value.
    #     ax: axis where to draw the plot, if None, it'll be created.
    #     r_factor: exponential factor to apply to distance r.
    #     output_prefix: Save the plot under this file structure (Default: None, plot not saved)
    #     save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
    #     save_plot: Save the plot into a png file. (Default: False).
    #     save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
    #     """
    #     if (save_data or save_plot) and output_prefix is None:
    #         raise ValueError("Set output_prefix in order to save data.")
    #     if save_data:
    #         data_dict = {}
    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     for bookkeeper, label in zip(bookkeepers, labels):
    #         values = CorrelationPlots.plot_cf(
    #             bookkeeper=bookkeeper,
    #             region=region,
    #             region2=region2,
    #             mumin=mumin,
    #             mumax=mumax,
    #             plot_kwargs={**plot_kwargs, **dict(label=label)},
    #             ax=ax,
    #             r_factor=r_factor,
    #         )

    #         if save_data:
    #             data_dict[f"{label}_r"] = values[0]
    #             data_dict[f"{label}_values"] = values[1]
    #             data_dict[f"{label}_errors"] = values[2]
    #             data_dict[f"{label}_nb"] = values[3]

    #     ax.legend()
    #     plt.tight_layout()
    #     if save_plot:
    #         plt.savefig(
    #             output_prefix.parent / (output_prefix.name + "-multiple_cf.png"),
    #         )
    #     if save_data:
    #         np.savez(
    #             output_prefix.parent / (output_prefix.name + "-multiple_cf.npz"),
    #             **{**save_dict, **data_dict},
    #         )

    @staticmethod
    def cf_errorbarsize(
        bookkeeper: Bookkeeper,
        region: str,
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        correlation_file: Optional[Path | str] = None,
        mumin: float = 0,
        mumax: float = 1,
        r_factor: int = 2,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot the size of the error bars in a correlation function (CF) for a
        given μ wedge.

        Parameters
        ----------
        bookkeeper : Bookkeeper
            Object providing file paths for correlation data.
        region : str
            Primary region used in the CF.
        region2 : str, optional
            Second region for cross-correlation (if any).
        absorber : str, optional
            Primary absorber. Default is "lya".
        absorber2 : str, optional
            Secondary absorber (if any).
        correlation_file : str or Path, optional
            Provide path to the correlation file.
            If None, inferred from bookkeeper.
        mumin : float, optional
            Minimum μ value for wedge selection.
            Default is 0.
        mumax : float, optional
            Maximum μ value for wedge selection.
            Default is 1.
        r_factor : int, optional
            Exponent factor applied to distance r.
            Default is 2.
        ax : matplotlib.axes.Axes, optional
            Axis to draw the plot on. If None, a new one is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to 'plot().'
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, optional
            If True, returns computed arrays without plotting.
            Default is False.
        output_prefix : str or Path, optional
            Prefix used for saving files if `save_plot` or `save_data` is True.
        save_data : bool, optional
            Whether to save the data to .npz file under the output_prefix
            file structure. Default is False.
        save_plot : bool, optional
            Whether to save the plot to .png file.
            Default is False.
        save_dict : dict, optional
            Additional metadata to include when saving data in the npz file
            if save_data option is True. Default is empty dict.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - r bin centers,
            - scaled CF errors,
            - effective number of pairs (nb)

        Raises
        ------
        ValueError
            If saving is requested but no `output_prefix` is provided.
        """

        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(
            bookkeeper.paths.exp_cf_fname(absorber, region, absorber2, region2)
        ) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            wedge = Wedge(
                rp=(cor_header["RPMIN"],
                    cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0),
                    cor_header["RTMAX"], cor_header["NT"]),
                r=(cor_header.get("RTMIN", 0),
                   cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
            )
        data_wedge = wedge(da, co)

        # Computing nb wedge manually
        axis_sum = wedge.weights.sum(axis=0)
        w = axis_sum > 0
        nb_weights = np.copy(wedge.weights)
        nb_weights[:, w] /= wedge.weights.sum(axis=0)[w]
        nb_wedge = data_wedge[0], nb_weights.dot(nb)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return data_wedge[0], r_coef * np.sqrt(np.diag(data_wedge[2])), nb_wedge[1]

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            data_wedge[0],
            r_coef * np.sqrt(np.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \sigma_\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))
        plt.tight_layout()

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * np.sqrt(np.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = nb_wedge[1]
            # data_dict['wedge_data'] = data_wedge

        if output_prefix is not None:
            if save_plot:
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + "-plot_cf.png"),
                )
            if save_data:
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + "-plot_cf.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

        return data_wedge[0], r_coef * np.sqrt(np.diag(data_wedge[2])), nb_wedge[1]

    @staticmethod
    def xcf_errorbarsize(
        bookkeeper: Bookkeeper,
        region: str,
        absorber: str = "lya",
        correlation_file: Optional[Path | str] = None,
        mumin: float = 0,
        mumax: float = 1,
        abs_mu: bool = True,
        r_factor: int = 2,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
        tracer: str = "qso",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute and optionally plot error bars for a cross-correlation
        function (XCF) within a specified μ wedge.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Object that provides correlation file paths and collects data.
        region : str, optional
            Region used for cross-correlation.
            Default is "lya".
        absorber : str, optional
            Absorber species. Default is "lya".
        mumin : float, optional
            Minimum μ value for the wedge.
            Default is 0.
        mumax : float, optional
            Maximum μ value for the wedge.
            Default is 1.
        abs_mu : bool, optional
            Whether to take absolute value of μ in wedge selection.
            Default is True.
        correlation_file : str or Path, optional
            Path to the correlation file.
        r_factor : int, optional
            Exponent factor applied to distance r.
            Default is 2.
        ax : matplotlib.axes.Axes, optional
            Axis to draw the plot on. If None, a new one is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to 'plot().'
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, optional
            If True, skip plotting and return computed arrays only.
            Default is False.
        output_prefix : str or Path, optional
            Prefix used for saving files if `save_plot` or `save_data` is True.
        save_data : bool, optional
            Whether to save the data to .npz file under the output_prefix
            file structure. Default is False.
        save_plot : bool, optional
            Whether to save the plot to .png file.
            Default is False.
        save_dict : dict, optional
            Additional metadata to include when saving data in the npz file
            if save_data option is True. Default is empty dict.
        tracer : str, optional
            Tracer label (default is "qso").

        Returns
        -------
        r : np.ndarray
            Array of radial distances.
        errors : np.ndarray
            Scaled standard deviation of the XCF.
        nb : np.ndarray
            Weighted number of bins contributing to each r.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(
            bookkeeper.paths.exp_xcf_fname(absorber, region, tracer)
        ) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            wedge = Wedge(
                rp=(cor_header["RPMIN"],
                    cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0),
                    cor_header["RTMAX"], cor_header["NT"]),
                r=(cor_header.get("RTMIN", 0),
                   cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
                abs_mu=abs_mu,
            )
        data_wedge = wedge(da, co)

        # Computing nb wedge manually
        axis_sum = wedge.weights.sum(axis=0)
        w = axis_sum > 0
        nb_weights = np.copy(wedge.weights)
        nb_weights[:, w] /= wedge.weights.sum(axis=0)[w]
        nb_wedge = data_wedge[0], nb_weights.dot(nb)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return data_wedge[0], r_coef * np.sqrt(np.diag(data_wedge[2])), nb_wedge[1]

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            data_wedge[0],
            r_coef * np.sqrt(np.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \sigma_\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))
        plt.tight_layout()

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * np.sqrt(np.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = nb_wedge[1]
            # data_dict['wedge_data'] = data_wedge

        if output_prefix is not None:
            if save_plot:
                plt.savefig(
                    output_prefix.parent /
                    (output_prefix.name + "-plot_cf.png"),
                )
            if save_data:
                np.savez(
                    output_prefix.parent /
                        (output_prefix.name + "-plot_cf.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data or save_plot:
            raise ValueError("Set output_prefix in order to save data.")

        return data_wedge[0], r_coef * np.sqrt(np.diag(data_wedge[2])), nb_wedge[1]

    # @staticmethod
    # def multiple_cf_errorbarsize(
    #     bookkeepers: List[Bookkeeper],
    #     region: str,
    #     labels: List[str] = None,
    #     region2: str = None,
    #     absorber: str = "lya",
    #     absorber2: str = None,
    #     correlation_file: Path | str = None,
    #     mumin: float = 0,
    #     mumax: float = 1,
    #     ax: Optional[Axes] = None,
    #     r_factor: int = 2,
    #     plot_kwargs: Dict = dict(),
    #     just_return_values: bool = False,
    #     output_prefix: Path | str = None,
    #     save_data: bool = False,
    #     save_plot: bool = False,
    #     save_dict: Dict = None,
    # ):
    #     """
    #     Plotting correlation function errors in defined wedge.

    #     Arguments:
    #     ----------

    #     bookkeepers: Bookkeepers object to use for collecting data.
    #     region: region to compute correlations from.
    #     labels: Labels to use for the different bookkeeper realizations.
    #     region2: region to compute correlations from if multiple regions have been used.
    #     mumin: wedge min value.
    #     mumax: wedge max value.
    #     ax: axis where to draw the plot, if None, it'll be created.
    #     r_factor: exponential factor to apply to distance r.
    #     output_prefix: Save the plot under this file structure (Default: None, plot not saved)
    #     save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
    #     save_plot: Save the plot into a png file. (Default: False).
    #     save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
    #     """
    #     if (save_data or save_plot) and output_prefix is None:
    #         raise ValueError("Set output_prefix in order to save data.")
    #     if save_data:
    #         data_dict = {}
    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     for bookkeeper, label in zip(bookkeepers, labels):
    #         values = CorrelationPlots.plot_cf_errorbarsize(
    #             bookkeeper=bookkeeper,
    #             region=region,
    #             region2=region2,
    #             mumin=mumin,
    #             mumax=mumax,
    #             ax=ax,
    #             r_factor=r_factor,
    #             plot_kwargs={**plot_kwargs, **dict(label=label)},
    #         )

    #         if save_data:
    #             data_dict[f"{label}_r"] = values[0]
    #             data_dict[f"{label}_errors"] = values[1]

    #     ax.legend()
    #     plt.tight_layout()
    #     if save_plot:
    #         plt.savefig(
    #             output_prefix.parent / (output_prefix.name + "-multiple_cf.png"),
    #         )
    #     if save_data:
    #         np.savez(
    #             output_prefix.parent / (output_prefix.name + "-multiple_cf.npz"),
    #             **{**save_dict, **data_dict},
    #         )

    @staticmethod
    def cf_map(
        bookkeeper: Bookkeeper,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        correlation_file: Optional[Path | str] = None,
        r_factor: int = 2,
        vmin: float = -0.04,
        vmax: float = 0.04,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[Tuple[float, ...], np.ndarray, np.ndarray, np.ndarray]:
        """
        bookkeeper: bookkeeper object to use for collect data.
        region: region to use.
        region2: if set use different second region.
        absorber: absorber to use.
        absorber2: if set use different second absorber.
        correlation_file: Provide path to correlation file alternatively.
        r_factor: exponential factor to apply to distance r.
        vmin: min third axis for the colormap.
        vmax: max third axis for the colormap.
        fig: matplotlib figure to make the plot in.
        ax: axis where to draw the plot, if None, it'll be created.
        plot_kwargs: extra kwargs to sent to the plotting function.
        output_prefix: Save the plot under this file structure
                        (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix
                        file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data
                        option is True. (Default: Empty dict)
        """

        """
        Generate a 2D correlation function heatmap from the provided
        correlation file.

        Parameters
        ----------
        bookkeeper : Optional[Bookkeeper]
            Bookkeeper object used to locate the correlation file / collect data.
        region : str, default="lya"
            Region identifier for the first dataset.
        region2 : Optional[str], default=None
            Region identifier for the second dataset (if cross-correlations).
        absorber : str, default="lya"
            Absorber type for the first dataset.
        absorber2 : Optional[str], default=None
            Absorber type for the second dataset, if applicable.
        correlation_file : Path or str, optional
            Provide path to a correlation file
            (overrides Bookkeeper object).
        r_factor : int, optional
            Exponent factor to scale the correlation values by radius
            (default is 2).
        vmin : float, optional
            Minimum value for color map scaling
            (default is -0.04).
        vmax : float, optional
            Maximum value for color map scaling
            (default is 0.04).
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure to use.
        ax : matplotlib.axes.Axes, optional
            Axis object to draw the heatmap. If None, a new figure is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to 'imshow().'
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, optional
            If True, skip plotting and return computed arrays only.
            Default is False.
        output_prefix : str or Path, optional
            Prefix used for saving files if `save_plot` or `save_data` is True.
        save_data : bool, optional
            Whether to save the data to .npz file under the output_prefix
            file structure. Default is False.
        save_plot : bool, optional
            Whether to save the plot to .png file.
            Default is False.
        save_dict : dict, optional
            Additional metadata to include when saving data in the npz file if
            save_data option is True. Default is empty dict.

        Returns
        -------
        extent : tuple
            Plot extent as (rt_min, rt_max, rp_min, rp_max).
        mat : np.ndarray
            Scaled correlation matrix.
        errmat : np.ndarray
            Matrix of correlation uncertainties.
        nmat : np.ndarray
            Matrix of contributing bin counts.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(
            bookkeeper.paths.exp_cf_fname(absorber, region, absorber2, region2)
        ) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()

        extent: tuple[float, ...]
        extent = (
            cor_header.get("RTMIN", 0),
            cor_header["RTMAX"],
            cor_header["RPMIN"],
            cor_header["RPMAX"],
        )
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header["NP"], cor_header["NT"]
        r = r.reshape(nrp, nrt)
        mat = da.reshape(nrp, nrt) * r**r_factor
        errmat = np.sqrt(np.diag(co)).reshape(nrp, nrt) * r**r_factor
        nmat = nb.reshape(nrp, nrt)

        if just_return_values:
            return extent, mat, errmat, nmat

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = dict(extent=extent, mat=mat, errmat=errmat, nmat=nmat)
        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fiug should be provided at the same time")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        im = ax.imshow(
            mat,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="seismic",
            **{
                **dict(vmin=vmin, vmax=vmax),
                **plot_kwargs,
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(r"$r_{\perp} \, [h^{-1} \, \rm{Mpc}]$")
        ax.set_ylabel(r"$r_{\parallel} \, [h^{-1} \, \rm{Mpc}]$")
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r"$r\xi(r_{\parallel,r_{\perp}})$")

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

        return extent, mat, errmat, nmat

    @staticmethod
    def xcf_map(
        bookkeeper: Bookkeeper,
        region: str = "lya",
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        correlation_file: Optional[Path | str] = None,
        r_factor: int = 2,
        vmin: float = -0.4,
        vmax: float = 0.4,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
        tracer: str = "qso",
    ) -> Tuple[Tuple[float, ...], np.ndarray, np.ndarray, np.ndarray]:
        """
        Plot a cross-correlation function heatmap using bookkeeper-managed
        correlation data.

        Parameters
        ----------
        bookkeeper : Optional[Bookkeeper]
            Bookkeeper object used to locate the correlation file / collect data.
        region : str, default="lya"
            Region identifier for the first dataset.
        absorber : str, default="lya"
            Absorber type for the first dataset.
        absorber2 : Optional[str], default=None
            Absorber type for the second dataset, if applicable.
        correlation_file : Path or str, optional
            Provide path to a correlation file
            (overrides Bookkeeper object).
        r_factor : int, optional
            Exponent factor to scale the correlation values by radius
            (default is 2).
        vmin : float, optional
            Minimum value for color map scaling
            (default is -0.04).
        vmax : float, optional
            Maximum value for color map scaling
            (default is 0.04).
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure to use.
        ax : matplotlib.axes.Axes, optional
            Axis object to draw the heatmap. If None, a new figure is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to 'imshow().'
            For example:
                plot_kwargs = {"cmap": "coolwarm",
                                 "vmin": -0.05,
                                 "vmax": 0.05,
                                 "aspect": "auto"}
        just_return_values : bool, optional
            If True, skip plotting and return computed arrays only.
            Default is False.
        output_prefix : str or Path, optional
            Prefix used for saving files if `save_plot` or `save_data` is True.
        save_data : bool, optional
            Whether to save the data to .npz file under the output_prefix
            file structure. Default is False.
        save_plot : bool, optional
            Whether to save the plot to .png file.
            Default is False.
        save_dict : dict, optional
            Additional metadata to include when saving data in the npz file if
            save_data option is True. Default is empty dict.
        tracer : str, optional
            Tracer label to distinguish data set (default is "qso").

        Returns
        -------
        extent : tuple
            Plot extent as (rt_min, rt_max, rp_min, rp_max).
        mat : np.ndarray
            Scaled cross-correlation matrix.
        errmat : np.ndarray
            Matrix of uncertainties.
        nmat : np.ndarray
            Matrix of contributing bin counts.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(
            bookkeeper.paths.exp_xcf_fname(absorber, region, tracer)
        ) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()

        extent: tuple[float, ...]
        extent = (
            cor_header.get("RTMIN", 0),
            cor_header["RTMAX"],
            cor_header["RPMIN"],
            cor_header["RPMAX"],
        )
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header["NP"], cor_header["NT"]
        r = r.reshape(nrp, nrt)
        mat = da.reshape(nrp, nrt) * r**r_factor
        errmat = np.sqrt(np.diag(co)).reshape(nrp, nrt) * r**r_factor
        nmat = nb.reshape(nrp, nrt)

        if just_return_values:
            return extent, mat, errmat, nmat

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = dict(extent=extent, mat=mat, errmat=errmat, nmat=nmat)
        if ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            raise ValueError("ax and fiug should be provided at the same time")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        im = ax.imshow(
            mat,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="seismic",
            **{
                **dict(vmin=vmin, vmax=vmax),
                **plot_kwargs,
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(r"$r_{\perp} \, [h^{-1} \, \rm{Mpc}]$")
        ax.set_ylabel(r"$r_{\parallel} \, [h^{-1} \, \rm{Mpc}]$")
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r"$r\xi(r_{\parallel,r_{\perp}})$")

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

        return extent, mat, errmat, nmat
