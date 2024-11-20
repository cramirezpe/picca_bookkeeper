from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import getdist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from getdist import MCSamples, plots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vega.parameters.param_utils import build_names
from vega.plots.wedges import Wedge

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Type

    import matplotlib
    from picca_bookkeeper.hints import Axes, Figure

logger = logging.getLogger(__name__)


def make_chain(names: List[int], mean: List[float], cov: np.ndarray) -> MCSamples:
    labels = build_names(names)
    gaussian_samples = np.random.multivariate_normal(mean, cov, size=1000000)
    samples = MCSamples(
        samples=gaussian_samples, names=names, labels=[labels[name] for name in labels]
    )
    return samples


class ReadFits:
    def __init__(
        self,
        bookkeeper: Optional[Bookkeeper | Path | str] = None,
        fit_file: Optional[Path | str] = None,
        label: Optional[str] = None,
        colour: Optional[str] = None,
        ap_baseline: Optional[float] = None,
        at_baseline: Optional[float] = None,
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
                raise FileNotFoundError(f"File does not exist, {str(fit_file)}")
        else:
            self.fit_file = self.bookkeeper.paths.fit_out_fname()

        self.label = label
        self.colour = colour

        self.ap_baseline = ap_baseline
        self.at_baseline = at_baseline

        self.read_fit()

    def __str__(self) -> str:
        if self.label is None:
            return self.fit_file.parents[1].name
        else:
            return self.label

    def read_fit(self) -> None:
        """
        Read relevant information from output and store it in variables.
        """
        with fitsio.FITS(self.fit_file) as hdul:
            self.values = dict()
            self.errors = dict()
            self.covs = dict()
            self.names = []
            for name, value, error, cov in zip(
                hdul["BESTFIT"]["names"].read(),
                hdul["BESTFIT"]["values"].read(),
                hdul["BESTFIT"]["errors"].read(),
                hdul["BESTFIT"]["covariance"].read(),
            ):
                self.names.append(name)
                self.values[name] = value
                self.errors[name] = error
                self.covs[name] = cov

            self.chi2 = hdul["BESTFIT"].read_header()["FVAL"]
            self.nparams = hdul["BESTFIT"].read_header()["NAXIS2"]

            self.ndata = 0
            columns = hdul["MODEL"].get_colnames()
            for column in hdul["MODEL"].get_colnames():
                if "MODEL" not in column and column[-4:] == "MASK":
                    self.ndata += hdul["MODEL"][column].read().sum()

            self.pvalue = 1 - sp.stats.chi2.cdf(self.chi2, self.ndata - self.nparams)

            self.model_header = hdul["MODEL"].read_header()

        # If fit performed in alpha/phi, translate result into ap/at.
        if ("ap" not in self.names or "at" not in self.names) and (
            "alpha" in self.names and "phi" in self.names
        ):
            self.names.append("ap")
            self.names.append("at")
            self.values["ap"] = self.values["alpha"] / np.sqrt(self.values["phi"])
            self.values["at"] = self.values["alpha"] * np.sqrt(self.values["phi"])

            self.errors["ap"] = np.sqrt(
                (self.errors["alpha"] / np.sqrt(self.values["phi"])) ** 2
                + (
                    self.errors["phi"]
                    * self.values["alpha"]
                    / (2 * self.values["phi"] ** (1.5))
                )
                ** 2
            )
            self.errors["at"] = np.sqrt(
                (self.errors["alpha"] * np.sqrt(self.values["phi"])) ** 2
                + (
                    self.errors["phi"]
                    * self.values["alpha"]
                    / np.sqrt(self.values["phi"])
                )
                ** 2
            )

        if "ap" in self.names and self.ap_baseline is not None:
            self.values["ap"] -= self.ap_baseline

        if "at" in self.names and self.at_baseline is not None:
            self.values["at"] -= self.at_baseline

    def compute_chain(self) -> None:
        res = {}
        res["chisq"] = self.chi2
        res["mean"] = list(self.values.values())
        res["cov"] = list(self.covs.values())

        res["pars"] = {
            name: {"val": self.values[name], "err": self.errors[name]}
            for name in self.names
        }

        self.chain = make_chain(res["pars"].keys(), res["mean"], res["cov"])

        
        
        
        
        
############################################################################
        
#     @staticmethod
#     import pandas as pd
#     from typing import List, Optional, Type

#     def table_from_fit_data(
#         fits: List[Type[ReadFits]],
#         params: List[str] = ["ap", "at", "bias_LYA", "beta_LYA"],
#         params_names: Optional[List[str]] = None,
#         precision: int = 3,
#         float_presentation: str = "f",
#     ) -> pd.DataFrame:
#         if params_names is None:
#             params_names = params
#         else:
#             params_names = params_names

#         header = ["name"]
#         header += params_names
#         header += ["fit", "pvalue"]

#         rows = []

#         if not fits:
#             return pd.DataFrame(columns=header)

#         # Determine baseline values
#         baseline_fit = fits[0]
#         baseline_values = {param: baseline_fit.values.get(param, 0) for param in params}

#         for fit in fits:
#             row = []
#             row.append(fit.label)

#             for param in params:
#                 if param in fit.values.keys():
#                     if param in ("ap", "at") and fit is not baseline_fit:
#                         # Compute the difference from the baseline value
#                         value = fit.values[param] - baseline_values.get(param, 0)
#                     else:
#                         # Use the baseline value itself
#                         value = fit.values[param]

#                     row.append(
#                         rf"{value:.{precision}{float_presentation}} "
#                         rf"± {fit.errors.get(param, 0):.{precision}{float_presentation}}"
#                     )
#                 else:
#                     row.append("")

#             row.append(
#                 f"{fit.chi2:.{precision}{float_presentation}}/({fit.ndata}-{fit.nparams})"
#             )
#             row.append(f"{fit.pvalue:.{precision}{float_presentation}}")

#             rows.append(row)

#         df = pd.DataFrame(data=rows)
#         df.columns = header
#         # df = df.sort_values("pvalue")

#         return df

        
        
##############################################################################
        
        
        
        
    @staticmethod
    def table_from_fit_data(
        fits: List[Type[ReadFits]],
        params: List[str] = ["ap", "at", "bias_LYA", "beta_LYA"],
        params_names: Optional[List[str]] = None,
        precision: int = 3,
        float_presentation: str = "f",
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
                if param in fit.values.keys():
                    if param == "ap":
                        row.append(
                            rf"{fit.values[param]:+.{precision}{float_presentation}} "
                            rf"± {fit.errors[param]:.{precision}{float_presentation}}"
                        )
                    elif param == "at":
                        row.append(
                            rf"{fit.values[param]:+.{precision}{float_presentation}} "
                            rf"± {fit.errors[param]:.{precision}{float_presentation}}"
                        )
                    else:
                        row.append(
                            rf"{fit.values[param]:.{precision}{float_presentation}} "
                            rf"± {fit.errors[param]:.{precision}{float_presentation}}"
                        )
                else:
                    row.append("")

            row.append(
                f"{fit.chi2:.{precision}{float_presentation}}/({fit.ndata}-{fit.nparams})"
            )
            row.append(f"{fit.pvalue:.{precision}{float_presentation}}")

            rows.append(row)

        df = pd.DataFrame(data=rows)
        df.columns = header
        # df = df.sort_values("pvalue")

        return df

##############################################################################################
    

class FitPlots:
    @staticmethod
    def cf_model(
        bookkeeper: Optional[Bookkeeper] = None,
        fit_file: Path | str = "",
        correlation_file: Path | str = "",
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        mumin: float = 0,
        mumax: float = 1,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
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

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if (fit_file == "" != correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not isinstance(bookkeeper, Bookkeeper):
            if not Path(fit_file).is_file() or not Path(correlation_file).is_file():
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if isinstance(bookkeeper, Bookkeeper):
            fit_file = bookkeeper.paths.fit_out_fname()
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
                r=(cor_header.get("RTMIN", 0), cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
            )

        with fitsio.FITS(fit_file) as ffile:
            colnames = ffile["MODEL"].get_colnames()
            if f"{region}x{region2}_MODEL" in colnames:
                field = f"{region}x{region2}_MODEL"
            elif f"{absorber}{region}x{absorber2}{region2}_MODEL" in colnames:
                field = f"{absorber}{region}x{absorber2}{region2}_MODEL"
            else:
                raise ValueError(
                    f"Unable to find compatible card for:\n"
                    f"\tregion:{region}\n\tabsorber:{absorber}"
                    f"\tregion2:{region2}\n\tabosrber2:{absorber2}",
                    colnames,
                )

            model = np.trim_zeros(
                ffile["MODEL"][field].read(),
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
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = model_wedge[0]
            data_dict["values"] = r_coef * model_wedge[1]
            data_dict["r_factor"] = r_factor

        plt.tight_layout()
        if save_data and output_prefix is not None:
            np.savez(
                output_prefix.parent / (output_prefix.name + "-plot_cf_model.npz"),
                **{**save_dict, **data_dict},
            )

        return (model_wedge[0], r_coef * model_wedge[1])

    @staticmethod
    def xcf_model(
        bookkeeper: Optional[Bookkeeper] = None,
        fit_file: Path | str = "",
        correlation_file: Path | str = "",
        region: str = "lya",
        absorber: str = "lya",
        mumin: float = 0,
        mumax: float = 1,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
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
            if (fit_file == "" != correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not isinstance(bookkeeper, Bookkeeper):
            if not Path(fit_file).is_file() or not Path(correlation_file).is_file():
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if isinstance(bookkeeper, Bookkeeper):
            fit_file = bookkeeper.paths.fit_out_fname()
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
                r=(cor_header.get("RTMIN", 0), cor_header["RTMAX"], cor_header["NT"]),
                mu=(mumin, mumax),
                abs_mu=True,
            )

        with fitsio.FITS(fit_file) as ffile:
            colnames = ffile["MODEL"].get_colnames()
            for name in (f"qsox{region}_MODEL", f"qsox{absorber}{region}_MODEL", f"{region}xqso_MODEL", f"{absorber}{region}xqso_MODEL"):
                if name in colnames:
                    field = name
                    break
            else:
                raise ValueError(
                    f"Unable to find compatible card for:\n"
                    f"\tregion:{region}\n\tabsorber:{absorber}",
                    colnames,
                )

            model = np.trim_zeros(
                ffile["MODEL"][field].read(),
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
        ax.grid(visible=True)
        ax.set_xlabel(r"$r \, [\mathrm{Mpc \, h^{-1}}]$")
        ax.set_ylabel(
            r"$r^{0} \xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$".format(r_factor)
        )
        ax.set_title("{0} < $\mu$ < {1}".format(mumin, mumax))

        if save_data:
            data_dict["r"] = model_wedge[0]
            data_dict["values"] = r_coef * model_wedge[1]
            data_dict["r_factor"] = r_factor

        plt.tight_layout()

        if output_prefix is not None:
            if save_data:
                np.savez(
                    output_prefix.parent / (output_prefix.name + "-plot_xcf_model.npz"),
                    **{**save_dict, **data_dict},
                )
        elif save_data:
            raise ValueError("Set output_prefix in order to save data.")

        return (model_wedge[0], r_coef * model_wedge[1])

    @staticmethod
    def cf_model_map(
        bookkeeper: Optional[Bookkeeper] = None,
        fit_file: Path | str = "",
        correlation_file: Path | str = "",
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        vmin: float = -0.04,
        vmax: float = 0.04,
        fig: Optional[Figure] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[Tuple[float, ...], np.ndarray]:
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

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if (fit_file == "" != correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not isinstance(bookkeeper, Bookkeeper):
            if not Path(fit_file).is_file() or not Path(correlation_file).is_file():
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if isinstance(bookkeeper, Bookkeeper):
            fit_file = bookkeeper.paths.fit_out_fname()
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
            colnames = ffile["MODEL"].get_colnames()
            if f"{region}x{region2}_MODEL" in colnames:
                field = f"{region}x{region2}_MODEL"
            elif f"{absorber}{region}x{absorber2}{region2}_MODEL" in colnames:
                field = f"{absorber}{region}x{absorber2}{region2}_MODEL"
            else:
                raise ValueError(
                    f"Unable to find compatible card for:\n"
                    f"\tregion:{region}\n\tabsorber:{absorber}"
                    f"\tregion2:{region2}\n\tabosrber2:{absorber2}",
                    colnames,
                )

            model = np.trim_zeros(
                ffile["MODEL"][field].read(),
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

        extent: tuple[float, ...]
        extent = (
            cor_header.get("RTMIN", 0.0),
            cor_header["RTMAX"],
            cor_header["RPMIN"],
            cor_header["RPMAX"],
        )
        r = np.sqrt(rp**2 + rt**2)
        nrp, nrt = cor_header["NP"], cor_header["NT"]
        r = r.reshape(nrp, nrt)

        mat = model.reshape(nrp, nrt) * r**r_factor

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

        return extent, mat

    @staticmethod
    def rp_model(
        bookkeeper: Optional[Bookkeeper] = None,
        fit_file: Path | str = "",
        correlation_file: Path | str = "",
        auto: Optional[bool] = False,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        rtmin: float = 0,
        rtmax: float = 1,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
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
            auto. Use auto-correlation
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

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if (fit_file == "" != correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not isinstance(bookkeeper, Bookkeeper):
            if not Path(fit_file).is_file() or not Path(correlation_file).is_file():
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if isinstance(bookkeeper, Bookkeeper):
            fit_file = bookkeeper.paths.fit_out_fname()
            if auto:
                correlation_file = bookkeeper.paths.exp_cf_fname(
                    absorber, region, absorber2, region2
                )
                name = "cf_rp"
            else:
                correlation_file = bookkeeper.paths.exp_xcf_fname(
                    absorber,
                    region,
                )
                name = "xcf_rp"

        with fitsio.FITS(correlation_file) as ffile:
            co = ffile["COR"]["CO"][:]
            nb = ffile["COR"]["NB"][:]

            cor_header = ffile["COR"].read_header()

            N_p = cor_header["NP"]
            N_t = cor_header["NT"]
            rp = ffile["COR"]["RP"][:]
            rt = ffile["COR"]["RT"][:]
            co = ffile["COR"]["CO"][:]

            cor_header = ffile["COR"].read_header()

        weights = 1 / np.diag(co).reshape(N_p, N_t)
        rt = rt.reshape(N_p, N_t)
        rp = rp.reshape(N_p, N_t)

        w = (rt >= rtmin) & (rt <= rtmax)
        weights[~w] = 0

        with fitsio.FITS(fit_file) as ffile:
            colnames = ffile["MODEL"].get_colnames()
            if auto:
                if f"{region}x{region2}_MODEL" in colnames:
                    field = f"{region}x{region2}_MODEL"
                elif f"{absorber}{region}x{absorber2}{region2}_MODEL" in colnames:
                    field = f"{absorber}{region}x{absorber2}{region2}_MODEL"
                else:
                    raise ValueError(
                        f"Unable to find compatible card for:\n"
                        f"\tregion:{region}\n\tabsorber:{absorber}"
                        f"\tregion2:{region2}\n\absorber2:{absorber2}",
                        colnames,
                    )
            else:
                for name in (f"qsox{region}_MODEL", f"qsox{absorber}{region}_MODEL", f"{region}xqso_MODEL", f"{absorber}{region}xqso_MODEL"):
                    if name in colnames:
                        field = name
                        break
                else:
                    raise ValueError(
                        f"Unable to find compatible card for:\n"
                        f"\tregion:{region}\n\tabsorber:{absorber}"
                        f"\tregion2:{region2}\n\absorber2:{absorber2}",
                        colnames,
                    )

            model = np.trim_zeros(
                ffile["MODEL"][field].read(),
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
            if auto:
                # this reshapes into data size
                model = model.reshape(model_np, N_t)[:N_p, :].reshape(-1)

            else:
                remove_idx = (model_np - N_p) // 2
                # this reshapes into data size
                model = model.reshape(model_np, N_t)[remove_idx:-remove_idx, :].reshape(
                    -1
                )

        mat = model.reshape(N_p, N_t)
        data = np.average(mat, weights=weights, axis=1)
        rp = np.average(rp, weights=weights, axis=1)

        if just_return_values:
            return (
                rp,
                data,
            )

        if save_data and output_prefix is None:
            raise ValueError("Set output_prefix in order to save data.")
        if save_data:
            data_dict = {}
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            rp,
            data,
            **plot_kwargs,
        )
        ax.grid(visible=True)
        ax.set_xlabel(r"$r_\parallel  \, [\mathrm{{Mpc \, h^{{-1}}  }}]$")
        ax.set_ylabel(r"$\xi(r) \, [\mathrm{{Mpc \, h^{{-1}}  }}]$")
        ax.set_title(
            rf"${rtmin} < r_\perp < {rtmax} \, \, [\mathrm{{Mpc \, h^{{-1}}}}]$"
        )

        if save_data:
            data_dict["rp"] = rp
            data_dict["data"] = data

        plt.tight_layout()
        if save_data and output_prefix is not None:
            np.savez(
                output_prefix.parent / (output_prefix.name + f"-plot_{name}_model.npz"),
                **{**save_dict, **data_dict},
            )

        return (rp, data)

    @staticmethod
    def xcf_model_map(
        bookkeeper: Optional[Bookkeeper] = None,
        fit_file: Path | str = "",
        correlation_file: Path | str = "",
        region: str = "lya",
        absorber: str = "lya",
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        vmin: float = -0.4,
        vmax: float = 0.4,
        fig: Optional[Figure] = None,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[Tuple[float, ...], np.ndarray]:
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
            if (fit_file == "" != correlation_file == ""):
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if not isinstance(bookkeeper, Bookkeeper):
            if not Path(fit_file).is_file() or not Path(correlation_file).is_file():
                raise ValueError(
                    "Should provide fit_file and correlation_file at the same"
                    "time or use a bookkeeper"
                )

        if isinstance(bookkeeper, Bookkeeper):
            fit_file = bookkeeper.paths.fit_out_fname()
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
            colnames = ffile["MODEL"].get_colnames()
            for name in (f"qsox{region}_MODEL", f"qsox{absorber}{region}_MODEL", f"{region}xqso_MODEL", f"{absorber}{region}xqso_MODEL"):
                if name in colnames:
                    field = name
                    break
            else:
                raise ValueError(
                    f"Unable to find compatible card for:\n"
                    f"\tregion:{region}\n\tabsorber:{absorber}",
                    colnames,
                )

            model = np.trim_zeros(
                ffile["MODEL"][field].read(),
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

        mat = model.reshape(nrp, nrt) * r**r_factor

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

        return extent, mat

    
####################################################################################



#     @staticmethod
#     def plot_errorbars_from_fit(
#         readfits: List[ReadFits],
#         param: str,
#         param_name: Optional[str] = None,
#         ax: Optional[Axes] = None,
#         plot_kwargs: Dict = dict(),
#         reference: Optional[ReadFits] = None,
#     ) -> List[matplotlib.container.Container | Any]:
#         """
#         Args:
#             readfits: List of readfits objects to show in the plot.
#             param: Param to plot.
#             param_name: Name of the param to show.
#             ax: Axis where to plot. If not provided, it will be created.
#             reference: Add reference as shaded line.

#         Returns:
#             List of plot handles to make legend from it.
#         """
#         if param_name is None:
#             param_name = param

#         if ax is None:
#             fig, ax = plt.subplots()

#         handles = []

#         # Baseline setup
#         baseline_value = None
#         baseline_error = None
        
#         if param in ("ap", "at") and readfits:
#             # Set the baseline using the first fit in the list
#             first_fit = readfits[0]
#             baseline_value = first_fit.values.get(param, None)
#             baseline_error = first_fit.errors.get(param, 0)
            
#             if baseline_value is None:
#                 baseline_value = first_fit.model_header.get(param, None)
#             if baseline_value is None and reference is not None:
#                 baseline_value = reference.values.get(param, None)
#             if baseline_value is None:
#                 baseline_value = 0
#             baseline_error = 0  # Assuming no baseline error for simplicity

#         if reference is not None and param in reference.values.keys():
#             value = reference.values.get(param, None)
#             error = reference.errors.get(param, 0)
#             # Centering reference value at 0
#             reference_centered_value = value - baseline_value
            
#             ax.axvspan(
#                 reference_centered_value - error,
#                 reference_centered_value + error,
#                 alpha=0.2,
#                 color="gray",
#             )
#             if param in ("ap", "at"):
#                 ax.axvspan(
#                     reference_centered_value - error,
#                     reference_centered_value - error / 3,
#                     alpha=0.2,
#                     color="red",
#                 )
#                 ax.axvspan(
#                     reference_centered_value + error / 3,
#                     reference_centered_value + error,
#                     alpha=0.2,
#                     color="red",
#                 )
#             handles.append(
#                 ax.axvline(
#                     reference_centered_value,
#                     color="black",
#                     ls="--",
#                     lw=0.6,
#                     alpha=1,
#                     label=reference.label,
#                 )
#             )

#         for i, fit in enumerate(readfits):
#             value = fit.values.get(param, None)
#             error = fit.errors.get(param, 0)

#             if value is None:
#                 value = fit.model_header.get(param, None)
#                 if value is None and reference is not None:
#                     value = reference.values.get(param, None)
#                 error = 0

#             if param in ("ap", "at") and baseline_value is not None:
#                 value -= baseline_value  # Adjust value to be the difference from baseline
#                 # Error remains the same

#             if fit.colour is not None:
#                 plot_kwargs = {**plot_kwargs, **dict(color=fit.colour)}

#             handles.append(
#                 ax.errorbar(
#                     value,
#                     i,
#                     xerr=error,
#                     yerr=0,
#                     label=fit.label,
#                     marker="o",
#                     **plot_kwargs,
#                 )
#             )

#         ax.set_xlabel(param_name)
#         ax.set_yticks([])
#         ax.grid(visible=True)

#         return handles







####################################################################################



    @staticmethod
    def plot_errorbars_from_fit(
        readfits: List[ReadFits],
        param: str,
        param_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        reference: Optional[ReadFits] = None,
    ) -> List[matplotlib.container.Container | Any]:
        """
        Args:
            readfits: List of readfits objects to show in the plot.
            param: Param to plot.
            param_name: Name of the param to show.
            ax: Axis where to plot. If not provided, it will be created.
            reference: Add reference as shaded line.

        Returns:
            List of plot handles to make legend from it.
        """
        if param_name is None:
            param_name = param

        if ax is None:
            fig, ax = plt.subplots()

        handles = []

        if reference is not None and param in reference.values.keys():
            value = reference.values.get(param, None)
            error = reference.errors.get(param, 0)
            ax.axvspan(
                value - error,
                value + error,
                alpha=0.2,
                color="gray",
            )
            if param in ("ap", "at"):
                ax.axvspan(
                    value - error,
                    value - error / 3,
                    alpha=0.2,
                    color="red",
                )
                ax.axvspan(
                    value + error / 3,
                    value + error,
                    alpha=0.2,
                    color="red",
                )
            handles.append(
                ax.axvline(
                    value,
                    color="black",
                    ls="--",
                    lw=0.6,
                    alpha=1,
                    label=reference.label,
                )
            )

        for i, fit in enumerate(readfits):
            value = fit.values.get(param, None)
            error = fit.errors.get(param, 0)

            if value is None:
                value = fit.model_header.get(param, None)

                if value is None and reference is not None:
                    value = reference.values.get(param, None)
                error = 0
                

            if fit.colour is not None:
                plot_kwargs = {**plot_kwargs, **dict(color=fit.colour)}

            handles.append(
                ax.errorbar(
                    value,
                    i,
                    xerr=error,
                    yerr=0,
                    label=fit.label,
                    marker="o",
                    **plot_kwargs,
                )
            )

        ax.set_xlabel(param_name)
        ax.set_yticks([])
        ax.grid(visible=True)

        return handles



####################################################################################

    @staticmethod
    def plot_p_value_from_fit(
        readfits: List[ReadFits],
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        reference: Optional[ReadFits] = None,
    ) -> List[matplotlib.container.Container | Any]:
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

        if reference is not None:
            handles.append(
                ax.axvline(
                    reference.pvalue,
                    color="black",
                    ls="--",
                    lw=0.6,
                    alpha=1,
                    label=reference.label,
                )
            )

        for i, fit in enumerate(readfits):
            if fit.colour is not None:
                plot_kwargs = {**plot_kwargs, **dict(color=fit.colour)}

            handles.append(
                ax.errorbar(
                    fit.pvalue / 2,
                    i,
                    xerr=fit.pvalue / 2,
                    yerr=0,
                    label=fit.label,
                    **plot_kwargs,
                )
            )

        ax.set_xlabel("p-value")
        ax.set_yticks([])
        ax.grid(visible=True)

        return handles

    @staticmethod
    def plot_triangle(
        param_names: List[str],
        readfits: Optional[List[ReadFits]] = None,
        chains: Optional[List[MCSamples]] = None,
        labels: Optional[List[str | None]] = None,
        colours: Optional[List[str | None]] = None,
        g: Optional[getdist.plots.GetDistPlotter] = None,
        plot_kwargs: Dict = dict(),
    ) -> None:
        if readfits is None and chains is None:
            raise ValueError("Either provide chains or ReadFits objects.")

        if readfits is not None:
            chains = []
            for fit in readfits:
                if hasattr(fit, "chain"):
                    chains.append(fit.chain)
                else:
                    raise AttributeError(
                        "Compute chain for all fits by using the method "
                        "compute_chain before running the triangle plot."
                    )

            if labels is None:
                labels = [fit.label for fit in readfits]
            if colours is None:
                colours = [fit.colour for fit in readfits]

        if g is None:
            g = getdist.plots.getSubplotPlotter()

        g.triangle_plot(
            chains,
            param_names,
            legend_labels=labels,
            contour_colors=colours,
            contours_ls=["solid", "dashed", "dotted"],
            **plot_kwargs,
        )
