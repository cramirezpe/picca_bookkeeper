import logging
from pathlib import Path
from typing import *

import fitsio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vega.plots.wedges import Wedge

from picca_bookkeeper.bookkeeper import Bookkeeper

logger = logging.getLogger(__name__)

class ReadFits:
    def __init__(
        self,
        bookkeeper: Union[Bookkeeper, Path, str] = None,
        fit_file: Union[Path, str] = None,
        label: str = None,
    ):
        """
        Args:
            bookkeeper: Bookkeeper object to read fit information from. Could also
                be the path to a bookkeeper configuration file.
            fit_file: Vega output file (if no bookkeeper provided).
            label: Label to identify the fit run.
        """
        if bookkeeper is None:
            if fit_file is None:
                raise ValueError("Either bookkeeper or fit_file should be provided")    
        elif isinstance(bookkeeper, Bookkeeper):
            self.bookkeeper = bookkeeper
        else:
            self.bookkeeper = Bookkeeper(bookkeeper)

        if fit_file is not None:
            if Path(fit_file).is_file():
                self.fit_file = Path(fit_file)
            else:
                raise FileNotFoundError(
                    f"File does not exist, {str(fit_file)}"
                )
        else:
            self.fit_file = self.bookkeeper.paths.fit_out_fname()

        self.label = label

        self.read_fit()

    def __str__(self):
        if self.label is None:
            return self.fit_file.parents[1].name    
        else:
            return self.label

    def read_fit(self) -> None:
        """
        Read relevant information from output and store it in variables.
        """
        with fitsio.FITS(self.fit_file) as hdul:
            self.names = hdul["BESTFIT"]["names"].read()
            self.values = hdul["BESTFIT"]["values"].read()
            self.errors = hdul["BESTFIT"]["errors"].read()

            self.chi2 = hdul["BESTFIT"].read_header()["FVAL"]
            self.nparams = hdul["BESTFIT"].read_header()["NAXIS2"]
            
            self.ndata = 0
            for x in "lyaxlya", "lyaxlyb", "qsoxlya", "qsoxlyb":
                label = f"{x}_MASK"
                if label in hdul["MODEL"].get_colnames():
                    self.ndata += hdul["MODEL"][f"{x}_MASK"].read().sum()
            
            self.pvalue = 1 - sp.stats.chi2.cdf(self.chi2, self.ndata - self.nparams)

    @staticmethod
    def table_from_fit_data(
        fits: List[Self],
        params: List[str] = ["ap", "at", "bias_LYA", "beta_LYA"],
        params_names: List[str] = None,
        ap_baseline: float = 0.985,
        at_baseline: float = 0.918,
        precision: int = 3,
    ) -> pd.DataFrame:
        if params_names is None:
            params_names = params
        else:
            params_names = params_names

        header = ["name"]
        header += params_names
        header += ["fit", "pvalue"]

        rows = []

        for fit in fits:
            row = []
            row.append(fit.label)

            for param in params:
                if param in fit.names:
                    idx = np.argmax(fit.names == param)
                    if param == "ap":
                        row.append(
                            rf"{fit.values[idx]-ap_baseline:+.{precision}f} "
                            rf"± {fit.errors[idx]:.{precision}f}"
                        )
                    elif param == "at":
                        row.append(
                            rf"{fit.values[idx]-at_baseline:+.{precision}f} "
                            rf"± {fit.errors[idx]:.{precision}f}"
                        )
                    else:
                        row.append(
                            rf"{fit.values[idx]:.{precision}f} "
                            rf"± {fit.errors[idx]:.{precision}f}"
                        )
                else:
                    row.append("")
            
            row.append(
                f"{fit.chi2:.{precision}f}/({fit.ndata}-{fit.nparams})"
            )
            row.append(
                f"{fit.pvalue:.{precision}f}"
            )

            rows.append(row)

        df = pd.DataFrame(data=rows)
        df.columns = header
        df = df.sort_values("pvalue")

        return df

        

class FitPlots:
    @staticmethod
    def cf_model(
        bookkeeper: Bookkeeper = None,
        fit_file: Union[Path, str] = "",
        correlation_file: Union[Path, str] = "",
        region: str = "lya",
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
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot fit correlation function model.

        Args:
            bookkeeper: Bookkeeper object ot use for collecting data.
            fit_file: Vega fit output file (Needed if no bookkeeper provided).
            correlation_file: Correlation output fits file (Needed if no bookkeeper
                provided).
            region: region to use.
            region2: if set use different second region.
            absorber: absorber to use.
            absorber2: if set use different second absorber.
            mumin: wedge min value.
            mumax: wedge max value.
            ax: axis where to draw the plot, if None, it'll be created.
            r_factor: exponential factor to apply to distance r.
            plot_kwargs: extra kwargs to sent to the plotting function.
            output_prefix: Save the plot under this file structure (Default: None,
                plot not saved)
            save_data: Save the data into a npz file under the output_prefix file
                structure. (Default: False).
            save_dict: Extra information to save in the npz file if save_data option
                is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if not (fit_file == "" and correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not Path(fit_file).is_file():
            fit_file = bookkeeper.paths.fit_out_fname()

        if not Path(correlation_file).is_file():
            correlation_file = bookkeeper.paths.exp_cf_fname(
                absorber, region, absorber2, region2
            )

        with fitsio.FITS(correlation_file) as ffile:
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()

            N_p = cor_header["NP"]
            N_t = cor_header["NT"]

            wedge = Wedge(
                rp=(cor_header["RPMIN"], cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0), cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
            )

        with fitsio.FITS(fit_file) as ffile:
            model = np.trim_zeros(
                ffile["MODEL"][f"{region}x{region2}_MODEL"].read(),
                "b",
            )

        if model.size != co.shape[0]:
            # What we do here is to use NT from data as the valid
            # NT for the model. Then we reshape model to remove
            # values above rp_data_max.
            logger.warning(
                "Model size is different to data size, "
                "assuming NT is the same for both"
            )

            if model.size % N_t != 0:
                raise ValueError("Unable to set NP for model.")

            model_np = model.size // N_t
            # this reshapes into data size
            model = model.reshape(model_np, N_t)[:N_p, :].reshape(-1)

        model_wedge = wedge(model, co)

        r_coef = model_wedge[0] ** r_factor

        if just_return_values:
            return (
                model_wedge[0],
                r_coef * model_wedge[1],
            )

        if save_data and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            model_wedge[0],
            r_coef * model_wedge[1],
            **plot_kwargs,
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
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)

        plt.tight_layout()
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_cf_model.npz"),
                **{**save_dict, **data_dict},
            )

        return (model_wedge[0], r_coef * model_wedge[1])

    @staticmethod
    def xcf_model(
        bookkeeper: Bookkeeper = None,
        fit_file: Union[Path, str] = "",
        correlation_file: Union[Path, str] = "",
        region: str = "lya",
        absorber: str = "lya",
        mumin: float = 0,
        mumax: float = 1,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot fit correlation function model.

        Args:
            bookkeeper: Bookkeeper object ot use for collecting data.
            fit_file: Vega fit output file (Needed if no bookkeeper provided).
            correlation_file: Correlation output fits file (Needed if no bookkeeper
                provided).
            region: region to use.
            absorber: absorber to use.
            mumin: wedge min value.
            mumax: wedge max value.
            ax: axis where to draw the plot, if None, it'll be created.
            r_factor: exponential factor to apply to distance r.
            plot_kwargs: extra kwargs to sent to the plotting function.
            output_prefix: Save the plot under this file structure (Default: None,
                plot not saved)
            save_data: Save the data into a npz file under the output_prefix file
                structure. (Default: False).
            save_dict: Extra information to save in the npz file if save_data option
                is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if not (fit_file == "" and correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not Path(fit_file).is_file():
            fit_file = bookkeeper.paths.fit_out_fname()

        if not Path(correlation_file).is_file():
            correlation_file = bookkeeper.paths.exp_xcf_fname(absorber, region)

        with fitsio.FITS(correlation_file) as ffile:
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()

            N_p = cor_header["NP"]
            N_t = cor_header["NT"]

            wedge = Wedge(
                rp=(cor_header["RPMIN"], cor_header["RPMAX"], cor_header["NP"]),
                rt=(cor_header.get("RTMIN", 0), cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
                abs_mu=True,
            )

        with fitsio.FITS(fit_file) as ffile:
            model = np.trim_zeros(
                ffile["MODEL"][f"qsox{region}_MODEL"].read(),
                "b",
            )

        if model.size != co.shape[0]:
            # What we do here is to use NT from data as the valid
            # NT for the model. Then we reshape model to remove
            # values above rp_data_max.
            logger.warning(
                "Model size is different to data size, "
                "assuming NT is the same for both"
            )

            if model.size % N_t != 0:
                raise ValueError("Unable to set NP for model.")
            elif (model.size // N_t) % 2 != 0:
                raise ValueError("Unable to set NP for model. Due to odd number of NP.")

            model_np = model.size // N_t
            remove_idx = (model_np - N_p) // 2
            # this reshapes into data size
            model = model.reshape(model_np, N_t)[remove_idx:-remove_idx, :].reshape(-1)

        model_wedge = wedge(model, co)

        r_coef = model_wedge[0] ** r_factor

        if just_return_values:
            return (
                model_wedge[0],
                r_coef * model_wedge[1],
            )

        if save_data and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            model_wedge[0],
            r_coef * model_wedge[1],
            **plot_kwargs,
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
            data_dict["r_factor"] = r_factor
            data_dict["nb"] = w.W.dot(nb)

        plt.tight_layout()
        if save_data:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_xcf_model.npz"),
                **{**save_dict, **data_dict},
            )

        return (model_wedge[0], r_coef * model_wedge[1])

    @staticmethod
    def cf_model_map(
        bookkeeper: Bookkeeper = None,
        fit_file: Union[Path, str] = "",
        correlation_file: Union[Path, str] = "",
        region: str = "lya",
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        vmin: float = -0.04,
        vmax: float = 0.04,
        fig: matplotlib.figure.Figure = None, 
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot fit correlation function model.

        Args:
            bookkeeper: Bookkeeper object ot use for collecting data.
            fit_file: Vega fit output file (Needed if no bookkeeper provided).
            correlation_file: Correlation output fits file (Needed if no bookkeeper
                provided).
            region: region to use.
            region2: if set use different second region.
            absorber: absorber to use.
            absorber2: if set use different second absorber.
            ax: axis where to draw the plot, if None, it'll be created.
            r_factor: exponential factor to apply to distance r.
            vmin: min third axis (colormap).
            vmax: max third axis (colormap).
            plot_kwargs: extra kwargs to sent to the plotting function.
            output_prefix: Save the plot under this file structure (Default: None,
                plot not saved)
            save_data: Save the data into a npz file under the output_prefix file
                structure. (Default: False).
            save_plot: Save plot into file.
            save_dict: Extra information to save in the npz file if save_data option
                is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if not (fit_file == "" and correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not Path(fit_file).is_file():
            fit_file = bookkeeper.paths.fit_out_fname()

        if not Path(correlation_file).is_file():
            correlation_file = bookkeeper.paths.exp_cf_fname(
                absorber, region, absorber2, region2
            )

        with fitsio.FITS(correlation_file) as ffile:
            co = ffile["COR"]["CO"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()

            N_p = cor_header["NP"]
            N_t = cor_header["NT"]

        with fitsio.FITS(fit_file) as ffile:
            model = np.trim_zeros(
                ffile["MODEL"][f"{region}x{region2}_MODEL"].read(),
                "b",
            )

        if model.size != co.shape[0]:
            # What we do here is to use NT from data as the valid
            # NT for the model. Then we reshape model to remove
            # values above rp_data_max.
            logger.warning(
                "Model size is different to data size, "
                "assuming NT is the same for both"
            )

            if model.size % N_t != 0:
                raise ValueError("Unable to set NP for model.")

            model_np = model.size // N_t
            # this reshapes into data size
            model = model.reshape(model_np, N_t)[:N_p, :].reshape(-1)

        extent = [
            cor_header.get("RTMIN", 0),
            cor_header["RTMAX"],
            cor_header["RPMIN"],
            cor_header["RPMAX"],
        ]
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header["NP"], cor_header["NT"]
        r = r.reshape(nrp, nrt)

        mat = model.reshape(nrp, nrt)*r**r_factor

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
                **dict(vmin=vmin, vmax=vmax),
                **plot_kwargs,
            },
        )

        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel(r"$r_{\perp} \, [h^{-1} \, \rm{Mpc}]$")
        ax.set_ylabel(r"$r_{\parallel} \, [h^{-1} \, \rm{Mpc}]$")
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r"$r\xi(r_{\parallel,r_{\perp}})$")

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

        return extent, mat

    @staticmethod
    def xcf_model_map(
        bookkeeper: Bookkeeper = None,
        fit_file: Union[Path, str] = "",
        correlation_file: Union[Path, str] = "",
        region: str = "lya",
        absorber: str = "lya",
        ax: matplotlib.axes._axes.Axes = None,
        r_factor: int = 2,
        vmin: float = -0.4,
        vmax: float = 0.4,
        fig: matplotlib.figure.Figure = None, 
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Union[Path, str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot fit correlation function model.

        Args:
            bookkeeper: Bookkeeper object ot use for collecting data.
            fit_file: Vega fit output file (Needed if no bookkeeper provided).
            correlation_file: Correlation output fits file (Needed if no bookkeeper
                provided).
            region: region to use.
            absorber: absorber to use.
            ax: axis where to draw the plot, if None, it'll be created.
            r_factor: exponential factor to apply to distance r.
            vmin: min third axis for the colormap.
            vmax: max third axis for the colormap.
            plot_kwargs: extra kwargs to sent to the plotting function.
            output_prefix: Save the plot under this file structure (Default: None,
                plot not saved)
            save_data: Save the data into a npz file under the output_prefix file
                structure. (Default: False).
            save_plot: Save the plot into a png file.
            save_dict: Extra information to save in the npz file if save_data option
                is True. (Default: Empty dict)
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if not (fit_file == "" and correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not Path(fit_file).is_file():
            fit_file = bookkeeper.paths.fit_out_fname()

        if not Path(correlation_file).is_file():
            correlation_file = bookkeeper.paths.exp_xcf_fname(absorber, region)

        with fitsio.FITS(correlation_file) as ffile:
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]

            cor_header = ffile["COR"].read_header()

            N_p = cor_header["NP"]
            N_t = cor_header["NT"]

        with fitsio.FITS(fit_file) as ffile:
            model = np.trim_zeros(
                ffile["MODEL"][f"qsox{region}_MODEL"].read(),
                "b",
            )

        if model.size != co.shape[0]:
            # What we do here is to use NT from data as the valid
            # NT for the model. Then we reshape model to remove
            # values above rp_data_max.
            logger.warning(
                "Model size is different to data size, "
                "assuming NT is the same for both"
            )

            if model.size % N_t != 0:
                raise ValueError("Unable to set NP for model.")
            elif (model.size // N_t) % 2 != 0:
                raise ValueError("Unable to set NP for model. Due to odd number of NP.")

            model_np = model.size // N_t
            remove_idx = (model_np - N_p) // 2
            # this reshapes into data size
            model = model.reshape(model_np, N_t)[remove_idx:-remove_idx, :].reshape(-1)

        extent = [
            cor_header.get("RTMIN", 0),
            cor_header["RTMAX"],
            cor_header["RPMIN"],
            cor_header["RPMAX"],
        ]
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header["NP"], cor_header["NT"]
        r = r.reshape(nrp, nrt)

        mat = model.reshape(nrp, nrt)*r**r_factor

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
                **dict(vmin=vmin, vmax=vmax),
                **plot_kwargs,
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

        return extent, mat

    @staticmethod
    def plot_errorbars_from_fit(
        readfits: List[ReadFits],
        param: str,
        param_name: str = None,
        ax: matplotlib.axes._axes.Axes = None,
    ) -> List[matplotlib.container.Container]:
        """
        Args:
            readfits: List of readfits objects to show in the plot.
            param: Param to plot.
            param_name: Name of the param to show.
            ax: Axis where to plot. If not provided, it will be created.

        Returns: 
            List of plot handles to make legend from it.
        """
        if param_name is None:
            param_name = param

        if ax is None:
            fig, ax = plt.subplots()

        handles = []

        for i, fit in enumerate(readfits):
            idx = np.argmax(fit.names == param)
            value = fit.values[idx]
            error = fit.errors[idx]

            handles.append(
                ax.errorbar(
                    value, i, xerr=error, yerr=0, label=fit.label, marker="o"
                )
            )

        ax.set_xlabel(param_name)
        ax.set_yticks([])
        ax.grid()

        return handles

    @staticmethod
    def plot_p_value_from_fit(
        readfits: List[ReadFits],
        ax: matplotlib.axes._axes.Axes = None,
    ) -> List[matplotlib.container.Container]:
        """
        Args:
            readfits: List of readfits objects to show in the plot.
            ax: Axis where to plot. If not provided, it will be created.

        Returns: 
            List of plot handles to make legend from it.
        """
        if ax is None:
            fig, ax = plt.subplots()

        handles = []
        for i, fit in enumerate(readfits):
            handles.append(
                ax.errorbar(
                    fit.pvalue / 2,
                    i,
                    xerr=fit.pvalue / 2,
                    yerr=0,
                    label=fit.label,
                )
            )
    
        ax.set_xlabel("p-value")
        ax.set_yticks([])
        ax.grid()

        return handles