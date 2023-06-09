import fitsio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import picca.wedgize
import scipy as sp
from pathlib import Path
from picca_bookkeeper.bookkeeper import Bookkeeper
from typing import *
import numpy as np


class CorrelationPlots:
    @staticmethod
    def cf(
        bookkeeper: Bookkeeper,
        region: str = "lya",
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
        mumin: float = 0,
        mumax: float = 1,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        errorbar_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting correlation function in defined wedge.

        Arguments:
        ----------

        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        region2: region to compute correlations from if multiple regions have been used.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_cf_fname(absorber, region, absorber2, region2)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            w = picca.wedgize.wedge(
                mumin=mumin, 
                mumax=mumax,
                rpmax=cor_header["RPMAX"],
                rpmin=cor_header["RPMIN"],
                nrp=cor_header["NP"],
                rtmax=cor_header["RTMAX"],
                rtmin=cor_header.get("RTMIN", 0),
                nrt=cor_header["NT"],
            )
        data_wedge = w.wedge(da, co)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return (
                data_wedge[0],
                r_coef * data_wedge[1],
                r_coef * sp.sqrt(sp.diag(data_wedge[2])),
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
            yerr=r_coef * sp.sqrt(sp.diag(data_wedge[2])),
            **errorbar_kwargs,
        )
        ax.grid()
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * data_wedge[1]
            data_dict["errors"] = r_coef * sp.sqrt(sp.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)
            # data_dict['wedge_data'] = data_wedge

        plt.tight_layout()
        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-plot_cf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_cf.npz"),
                **{**save_dict, **data_dict},
            )

        return (
            data_wedge[0],
            r_coef * data_wedge[1],
            r_coef * sp.sqrt(sp.diag(data_wedge[2])),
        )

    @staticmethod
    def xcf(
        bookkeeper: Bookkeeper,
        region: str= "lya",
        absorber: str = "lya",
        mumin: float = 0,
        mumax: float = 1,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        errorbar_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting correlation function in defined wedge.

        Arguments:
        ----------

        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        region2: region to compute correlations from if multiple regions have been used.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_xcf_fname(absorber, region)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            w = picca.wedgize.wedge(
                mumin=mumin,
                mumax=mumax,
                rpmin=cor_header["RPMAX"],
                rpmax=cor_header["RPMIN"],
                nrp=cor_header["NP"],
                rtmin=cor_header["RTMAX"],
                rtmax=cor_header.get("RTMIN", 0),
                nrt=cor_header["NT"],
                absoluteMu=True,
            )
        data_wedge = w.wedge(da, co)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return (
                data_wedge[0],
                r_coef * data_wedge[1],
                r_coef * sp.sqrt(sp.diag(data_wedge[2])),
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
            yerr=r_coef * sp.sqrt(sp.diag(data_wedge[2])),
            **errorbar_kwargs,
        )
        ax.grid()
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * data_wedge[1]
            data_dict["errors"] = r_coef * sp.sqrt(sp.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)
            # data_dict['wedge_data'] = data_wedge

        plt.tight_layout()
        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-plot_xcf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_xcf.npz"),
                **{**save_dict, **data_dict},
            )

        return (
            data_wedge[0],
            r_coef * data_wedge[1],
            r_coef * sp.sqrt(sp.diag(data_wedge[2])),
        )

    @staticmethod
    def multiple_cf(
        bookkeepers: List[Bookkeeper],
        region: str,
        labels: List[str] = None,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
        mumin: float = 0,
        mumax: float = 1,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        errorbar_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = None,
    ):
        """
        Plotting correlation function in defined wedge.

        Arguments:
        ----------

        bookkeepers: Bookkeepers object to use for collecting data.
        region: region to compute correlations from.
        labels: Labels to use for the different bookkeeper realizations.
        region2: region to compute correlations from if multiple regions have been used.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        for bookkeeper, label in zip(bookkeepers, labels):
            values = CorrelationPlots.plot_cf(
                bookkeeper=bookkeeper,
                region=region,
                region2=region2,
                mumin=mumin,
                mumax=mumax,
                errorbar_kwargs={**errorbar_kwargs, **dict(label=label)},
                ax=ax,
                r_factor=r_factor,
            )

            if save_data:
                data_dict[f"{label}_r"] = values[0]
                data_dict[f"{label}_values"] = values[1]
                data_dict[f"{label}_errors"] = values[2]
                data_dict[f"{label}_nb"] = values[3]

        ax.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-multiple_cf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-multiple_cf.npz"),
                **{**save_dict, **data_dict},
            )

    @staticmethod
    def cf_errorbarsize(
        bookkeeper,
        region,
        region2=None,
        absorber: str = "lya",
        absorber2: str = None,
        mumin=0,
        mumax=1,
        r_factor=2,
        ax=None,
        plot_kwargs=dict(),
        just_return_values=False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting correlation errorbar in defined wedge.

        Arguments:
        ----------
        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        region2: region to compute correlations from if multiple regions have been used.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_cf_fname(absorber, region, absorber2, region2)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            w = picca.wedgize.wedge(
                mumin=mumin, 
                mumax=mumax,
                rpmax=cor_header["RPMAX"],
                rpmin=cor_header["RPMIN"],
                nrp=cor_header["NP"],
                rtmax=cor_header["RTMAX"],
                rtmin=cor_header.get("RTMIN", 0),
                nrt=cor_header["NT"],
            )
        data_wedge = w.wedge(da, co)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return data_wedge[0], r_coef * sp.sqrt(sp.diag(data_wedge[2]))

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            data_wedge[0],
            r_coef * sp.sqrt(sp.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid()
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \sigma_\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))
        plt.tight_layout()

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * sp.sqrt(sp.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)
            # data_dict['wedge_data'] = data_wedge

        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-plot_cf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_cf.npz"),
                **{**save_dict, **data_dict},
            )

        return data_wedge[0], r_coef * sp.sqrt(sp.diag(data_wedge[2]))

    @staticmethod
    def xcf_errorbarsize(
        bookkeeper: Bookkeeper,
        region: str,
        absorber: str = "lya",
        mumin: float = 0,
        mumax: float = 1,
        r_factor: int = 2,
        ax: matplotlib.axes.Axes = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting cross-correlation errorbars in defined wedge.

        Arguments:
        ----------
        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_xcf_fname(absorber, region)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()
            w = picca.wedgize.wedge(
                mumin=mumin,
                mumax=mumax,
                rpmin=cor_header["RPMAX"],
                rpmax=cor_header["RPMIN"],
                nrp=cor_header["NP"],
                rtmin=cor_header["RTMAX"],
                rtmax=cor_header.get("RTMIN", 0),
                nrt=cor_header["NT"],
                absoluteMu=True,
            )
        data_wedge = w.wedge(da, co)

        r_coef = data_wedge[0] ** r_factor

        if just_return_values:
            return data_wedge[0], r_coef * sp.sqrt(sp.diag(data_wedge[2]))

        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            data_wedge[0],
            r_coef * sp.sqrt(sp.diag(data_wedge[2])),
            **plot_kwargs,
        )
        ax.grid()
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \sigma_\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))
        plt.tight_layout()

        if save_data:
            data_dict["r"] = data_wedge[0]
            data_dict["values"] = r_coef * sp.sqrt(sp.diag(data_wedge[2]))
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)
            # data_dict['wedge_data'] = data_wedge

        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-plot_cf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_cf.npz"),
                **{**save_dict, **data_dict},
            )

        return data_wedge[0], r_coef * sp.sqrt(sp.diag(data_wedge[2]))

    @staticmethod
    def multiple_cf_errorbarsize(
        bookkeepers: List[Bookkeeper],
        region: str,
        labels: List[str] = None,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
        mumin: float = 0,
        mumax: float = 1,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = None,
    ):
        """
        Plotting correlation function errors in defined wedge.

        Arguments:
        ----------

        bookkeepers: Bookkeepers object to use for collecting data.
        region: region to compute correlations from.
        labels: Labels to use for the different bookkeeper realizations.
        region2: region to compute correlations from if multiple regions have been used.
        mumin: wedge min value.
        mumax: wedge max value.
        ax: axis where to draw the plot, if None, it'll be created.
        r_factor: exponential factor to apply to distance r.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        for bookkeeper, label in zip(bookkeepers, labels):
            values = CorrelationPlots.plot_cf_errorbarsize(
                bookkeeper=bookkeeper,
                region=region,
                region2=region2,
                mumin=mumin,
                mumax=mumax,
                ax=ax,
                r_factor=r_factor,
                plot_kwargs={**plot_kwargs, **dict(label=label)},
            )

            if save_data:
                data_dict[f"{label}_r"] = values[0]
                data_dict[f"{label}_errors"] = values[1]

        ax.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(
                output_prefix.parent / (output_prefix.name + "-multiple_cf.png"),
            )
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-multiple_cf.npz"),
                **{**save_dict, **data_dict},
            )

    @staticmethod
    def cf_map(
        bookkeeper: Bookkeeper,
        region: str='lya',
        region2: str=None,
        absorber: str = "lya",
        absorber2: str = None,
        r_factor: int = 2,
        vmin=-0.04,
        vmax=0.04,
        fig=None,
        ax: matplotlib.axes.Axes = None,
        imshow_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
        plot_bar_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting correlation heatmap.

        Arguments:
        ----------
        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        region2: region to compute correlations from if multiple regions have been used.
        r_factor: exponential factor to apply to distance r.
        ax: axis where to draw the plot, if None, it'll be created.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_cf_fname(absorber, region, absorber2, region2)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()
        
        extent = [cor_header.get('RTMIN', 0),cor_header['RTMAX'],cor_header['RPMIN'],cor_header['RPMAX']]
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header['NP'], cor_header['NT']
        r = r.reshape(nrp, nrt)
        mat = da.reshape(nrp, nrt)*r**r_factor

        if just_return_values:
            return extent, mat
        
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = dict(extent=extent, mat=mat)
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
                **imshow_kwargs,
                **dict(vmin=-0.04, vmax=0.04),
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(r'$r_{\perp} \, [h^{-1} \, \rm{Mpc}]$')
        ax.set_ylabel(r'$r_{\parallel} \, [h^{-1} \, \rm{Mpc}]$')
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'$r\xi(r_{\parallel,r_{\perp}})$')
        
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
        
    @staticmethod
    def xcf_map(
        bookkeeper: Bookkeeper,
        region: str='lya',
        absorber: str = "lya",
        absorber2: str = None,
        r_factor: int = 2,
        vmin=-0.04,
        vmax=0.04,
        fig=None,
        ax: matplotlib.axes.Axes = None,
        imshow_kwargs: Dict = dict(),
        plot_kwargs: Dict = dict(),
        plot_bar_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ):
        """
        Plotting correlation heatmap.

        Arguments:
        ----------
        bookkeeper: Bookkeeper object to use for collect data.
        region: region to compute correlations from.
        r_factor: exponential factor to apply to distance r.
        ax: axis where to draw the plot, if None, it'll be created.
        output_prefix: Save the plot under this file structure (Default: None, plot not saved)
        save_data: Save the data into a npz file under the output_prefix file structure. (Default: False).
        save_plot: Save the plot into a png file. (Default: False).
        save_dict: Extra information to save in the npz file if save_data option is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        with fitsio.FITS(bookkeeper.paths.exp_xcf_fname(absorber, region)) as ffile:
            try:
                da = ffile["COR"]["DA"][:]
            except ValueError:
                da = ffile["COR"]["DA_BLIND"][:]
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()
        
        extent = [cor_header.get('RTMIN', 0),cor_header['RTMAX'],cor_header['RPMIN'],cor_header['RPMAX']]
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header['NP'], cor_header['NT']
        r = r.reshape(nrp, nrt)
        mat = da.reshape(nrp, nrt)*r**r_factor

        if just_return_values:
            return extent, mat
        
        if (save_data or save_plot) and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = dict(extent=extent, mat=mat)
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
                **imshow_kwargs,
                **dict(vmin=-0.4, vmax=0.4),
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(r'$r_{\perp} \, [h^{-1} \, \rm{Mpc}]$')
        ax.set_ylabel(r'$r_{\parallel} \, [h^{-1} \, \rm{Mpc}]$')
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'$r\xi(r_{\parallel,r_{\perp}})$')
        
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
        