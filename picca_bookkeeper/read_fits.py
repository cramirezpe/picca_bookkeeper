"""
read_fits.py

This module provides classes and functions for reading, processing, and 
visualizing results from FITS files produced by Vega and managed via the 
picca_bookkeeper.

Core Functionality:
-------------------
    - `ReadFits`: Class for extracting fit parameters, errors, covariance matrices, 
      and statistical summaries (e.g., chi2, p-values) from FITS output files. 
      Supports both direct file input and integration with a `Bookkeeper` 
      instance to manage file paths and configuration.
    - Handles translation of fit results between different parameterizations 
      (e.g., from 'alpha' and 'phi,' to 'ap' and 'at'), and supports blinded analyses.
    - Provides methods for computing MCMC chains from fit results (`compute_chain`), 
      and for constructing summary tables across multiple fits (`table_from_fit_data`).

Visualization Utilities:
------------------------
- `FitPlots`: Collection of static methods for visualizing fit results, including:
    - Correlation function and cross-correlation model plots (`cf_model`, `xcf_model`).
    - 2D map visualizations of correlation functions (`cf_model_map`, `xcf_model_map`).
    - Projected correlation function plots (`rp_model`).
    - Error bar plots for fit parameters (`plot_errorbars_from_fit`), 
      including support for blinded / unblinded display.
    - p-value plots (`plot_p_value_from_fit`).
    - Triangle plots for parameter covariances using GetDist (`plot_triangle`).

Dependencies and Interactions:
------------------------------
- Uses picca_bookkeeper's `Bookkeeper` class for file management and configuration.
- Integrates with Vega's plotting and parameter utilities 
  (e.g., `vega.plots.wedges.Wedge`, `vega.parameters.param_utils.build_names`).

Usage:
------
Typical usage involves creating a `ReadFits` object by providing either a 
`Bookkeeper` instance (preferred for automated path management) or a direct 
FITS file path. Visualization and summary functions are available via the 
`FitPlots` class, and can operate on lists of `ReadFits` instances for 
comparative analysis.

Example:
--------
    from picca_bookkeeper.bookkeeper import Bookkeeper
    from picca_bookkeeper.read_fits import ReadFits, FitPlots

    bk = Bookkeeper("config.yml")
    rf = ReadFits(bookkeeper=bk)
    rf.compute_chain()
    df = ReadFits.table_from_fit_data([rf])
    FitPlots.cf_model(bookkeeper=bk)
    FitPlots.plot_triangle(["ap", "at"], readfits=[rf])

Notes:
------
    - All plotting methods accept arguments for customizing output and saving 
      figures / data.
    - Blinding options are supported for analyses where baseline parameter 
      values should remain hidden.
    - Error handling ensures operation when files or data are missing.

"""
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
from getdist import MCSamples
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
    """
    Create a Gaussian posterior sample from a multivariate normal distribution.

    Arguments:
    ----------
        names: List of parameter name strings (used as GetDist parameter names).
        mean: List of mean values for the parameters.
        cov: Covariance matrix corresponding to the parameters.

    Returns:
    ----------
        MCSamples: A GetDist MCSamples object containing 1,000,000 samples
                   drawn from the specified Gaussian distribution, with labeled
                   parameter names for plotting or analysis.

    Notes:
    ----------
        The labels are constructed using `build_names` to provide LaTeX-style
        labels for use in corner plots and tables.
    """
    labels = build_names(names)
    gaussian_samples = np.random.multivariate_normal(mean, cov, size=1000000)
    samples = MCSamples(
        samples=gaussian_samples, names=names, labels=[labels[name] for name in labels]
    )
    return samples


class ReadFits:
    """
    Class to read, interpret, and post-process fit results from Vega FITS output files.

    Supports both direct input of FITS files and automated path resolution via a
    `Bookkeeper` instance. Extracts best-fit parameter values, errors, covariance
    matrices, and statistical metadata such as chi^2 and p-values.

    Also handles translation of parameterizations (e.g., from 'alpha' and 'phi' to
    'ap' and 'at'), and supports optional blinding of parameters.

    Attributes:
    ----------
        fit_file (Path): Path to the Vega output FITS file.
        label (str): Identifier for the fit (used in plotting and summaries).
        colour (str): Optional plotting color.
        ap_baseline (float): Value to subtract from ap (used when blinded=True).
        at_baseline (float): Value to subtract from at (used when blinded=True).
        values (Dict): Dictionary of best-fit parameter values.
        errors (Dict): Dictionary of parameter errors.
        covs (Dict): Dictionary of parameter covariance rows.
        names (List[str]): List of parameter names.
        chi2 (float): Fit χ² value.
        nparams (int): Number of fit parameters.
        ndata (int): Number of unmasked data points.
        pvalue (float): Fit p-value.
        model_header (Header): FITS header from the MODEL extension.
    """
    def __init__(
        self,
        bookkeeper: Optional[Bookkeeper | Path | str] = None,
        fit_file: Optional[Path | str] = None,
        label: Optional[str] = None,
        colour: Optional[str] = None,
        ap_baseline: Optional[float] = None,
        at_baseline: Optional[float] = None,
        blinded: bool = True,  # handle blinded(true) / unblined(false), default = blinded
    ):
        """
        Initialize a ReadFits object and load fit results from file.

        Arguments:
        ----------
            fit_file: Path to the Vega FITS file with fit results 
                        (if no bookkeeper provided).
            bookkeeper: Bookkeeper object to read fit information from. Could also
                        be the path to a bookkeeper configuration file.
            variation: Name of the variation to fetch from the bookkeeper.
            tracer: Tracer type (e.g., 'lya').
            bin_type: Binning type (e.g., 'sep', 'mu').
            region: Sky region (e.g., 'NGC', 'SGC').
            label: Optional label to identify the fit run.
            colour: Optional plotting color.
            blinded: If True, subtracts ap/at baseline values from reported values.
            ap_baseline: Reference ap value to subtract if blinded is True.
            at_baseline: Reference at value to subtract if blinded is True.

        Raises:
        ----------
            ValueError: If neither `fit_file` nor a valid bookkeeper + variation combo is provided.
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

        self.values: Dict = dict()
        self.errors: Dict = dict()
        self.covs: Dict = dict()
        self.names: List = []

    def __str__(self) -> str:
        if self.label is None:
            return self.fit_file.parents[1].name
        else:
            return self.label

    def read_fit(self) -> None:
        """
        Read and parse fit results from the FITS file.

        Populates parameter names, values, errors, and covariances from the
        'BESTFIT' HDU. Also reads fit metadata including chi^2, number of data
        points, and p-value. If 'alpha' and 'phi' parameters are found, converts
        to 'ap' and 'at' and propagates their errors.

        Handles optional subtraction of ap/at baselines if set.

        Raises:
        ----------
            FileNotFoundError: If the fit file is missing.
            ValueError: If neither a bookkeeper nor a fit file is provided.
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
        """
        Generate a Gaussian approximation MCMC chain for the fit parameters.

        Constructs a GetDist `MCSamples` object from the best-fit mean and
        covariance matrix. This can be used for triangle plots or other
        sampling-based visualizations.

        Stores the result in `self.chain`.
        """
        res = {}
        res["chisq"] = self.chi2
        res["mean"] = list(self.values.values())
        res["cov"] = list(self.covs.values())

        res["pars"] = {
            name: {"val": self.values[name], "err": self.errors[name]}
            for name in self.names
        }

        self.chain = make_chain(res["pars"].keys(), res["mean"], res["cov"])

    @staticmethod
    def table_from_fit_data(
        fits: List[Type[ReadFits]],
        params: List[str] = ["ap", "at", "bias_LYA", "beta_LYA"],
        params_names: Optional[List[str]] = None,
        precision: int = 3,
        float_presentation: str = "f",
        blinded: bool = True,  # handle blinded(true) / unblined(false), default = blinded
    ) -> pd.DataFrame:
        """
        Construct a summary table of fit results from a list of `ReadFits` instances.

        Parameters can be shown blinded (differences from a baseline apv/vat) or unblinded
        (absolute values). The table includes parameter values ± errors, chi^2 per DoF,
        and p-values.

        Arguments:
        ----------
            fits: List of `ReadFits` instances to include in the table.
            params: List of parameter keys to include (default: ap, at, bias_LYA, beta_LYA).
            params_names: Optional list of display names (defaults to same as `params`).
            precision: Decimal precision for numerical formatting.
            float_presentation: Format style (e.g., "f" for fixed-point).
            blinded: If True, shows ap/at as deltas from the baseline fit.

        Returns:
        ----------
            pd.DataFrame: Table of formatted parameter summaries.
        """        
        if params_names is None:
            params_names = params
        else:
            params_names = params_names

        header = ["name"]
        header += params_names
        header += ["chi2 / DoF", "pvalue"]

        rows = []

        # check if blinded boolian is true / false
        if blinded:
            # Store baseline values for 'ap' and 'at' from the first fit
            baseline_ap = None
            baseline_at = None

            # First pass to determine baseline values
            for fit in fits:
                if "ap" in fit.values:
                    if baseline_ap is None:
                        baseline_ap = fit.values["ap"]
                if "at" in fit.values:
                    if baseline_at is None:
                        baseline_at = fit.values["at"]

            # Process each fit and calculate differences from the baseline
            for index, fit in enumerate(fits):
                row = []
                row.append(fit.label)

                for param in params:
                    if param in fit.values.keys():
                        if param == "ap":
                            if index == 0:
                                # For the first fit, display placeholder text with precision
                                row.append(
                                    rf"xxx ± {fit.errors[param]:.{precision}{float_presentation}}"
                                )
                            else:
                                # For subsequent fits, display the difference from the baseline
                                value_diff = fit.values[param] - baseline_ap
                                error_diff = fit.errors[
                                    param
                                ]  # Assuming the error remains the same
                                row.append(
                                    rf"{value_diff:+.{precision}{float_presentation}} ± {error_diff:.{precision}{float_presentation}}"
                                )
                        elif param == "at":
                            if index == 0:
                                # For the first fit, display placeholder text with precision
                                row.append(
                                    rf"xxx ± {fit.errors[param]:.{precision}{float_presentation}}"
                                )
                            else:
                                # For subsequent fits, display the difference from the baseline
                                value_diff = fit.values[param] - baseline_at
                                error_diff = fit.errors[
                                    param
                                ]  # Assuming the error remains the same
                                row.append(
                                    rf"{value_diff:+.{precision}{float_presentation}} ± {error_diff:.{precision}{float_presentation}}"
                                )
                        else:
                            row.append(
                                rf"{fit.values[param]:.{precision}{float_presentation}} ± {fit.errors[param]:.{precision}{float_presentation}}"
                            )
                    else:
                        row.append("")

                row.append(
                    f"{fit.chi2:.{precision}{float_presentation}}/({fit.ndata}-{fit.nparams})"
                )
                row.append(f"{fit.pvalue:.{precision}{float_presentation}}")

                rows.append(row)
        else:
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
        Plot the correlation function model from a Vega fit result and corresponding 
        correlation data, optionally saving the output or returning raw values.
    
        Arguments:
        ----------
            bookkeeper (Bookkeeper, optional): Object for managing file paths and metadata. 
                If not provided, both `fit_file` and `correlation_file` must be set manually.
            fit_file (Path or str): Path to the Vega fit output file. 
                Required if no bookkeeper is used.
            correlation_file (Path or str): Path to the correlation output FITS file. 
                Required if no bookkeeper is used.
            region (str): Spectral region label for the first tracer (default: "lya").
            region2 (str, optional): Spectral region label for the second tracer. 
                Defaults to `region` above if not specified.
            absorber (str): Absorber name for the first tracer (default: "lya").
            absorber2 (str, optional): Absorber name for the second tracer. 
                Defaults to `absorber` above if not specified.
            mumin (float): Minimum μ value for the angular wedge (default: 0).
            mumax (float): Maximum μ value for the angular wedge (default: 1).
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object to plot on. 
                A new axis is created if None.
            r_factor (int): Exponent applied to the radial coordinate for scaling (default: 2).
            plot_kwargs (dict): Additional keyword arguments passed to `ax.plot()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}
            just_return_values (bool): If True, skip plotting and return the radial and 
                model ξ(r) values only.
            output_prefix (Path or str, optional): If set, used as the prefix path for 
                saving the output plot data (e.g., `.npz` file).
            save_data (bool): If True, save the plotted data to disk using `output_prefix`.
            save_dict (dict): Extra metadata to include in the saved `.npz` output file 
                (default: empty dict).
    
        Returns:
        ----------
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Radial distances (r) after wedge integration.
                - Scaled model correlation function values: r^r_factor * ξ(r).
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if fit_file == "" != correlation_file == "":
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
        abs_mu: bool = True,
        ax: Optional[Axes] = None,
        r_factor: int = 2,
        plot_kwargs: Dict = dict(),
        just_return_values: bool = False,
        output_prefix: Optional[Path | str] = None,
        save_data: bool = False,
        save_dict: Dict = dict(),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot the cross-correlation function (XCF) model from a Vega fit and correlation dataset, 
        with optional data saving or value return.
       
        Arguments:
        ----------
            bookkeeper (Bookkeeper, optional): Object for managing file paths and metadata. 
                If not provided, both `fit_file` and `correlation_file` must be set manually.
            fit_file (Path or str): Path to the Vega fit output file. Required if no bookkeeper is used.
            correlation_file (Path or str): Path to the cross-correlation FITS file. 
                Required if no bookkeeper is used.
            region (str): Spectral region of the absorption field (default: "lya").
            absorber (str): Name of the absorber field (default: "lya").
            mumin (float): Minimum μ value for the angular wedge (default: 0).
            mumax (float): Maximum μ value for the angular wedge (default: 1).
            abs_mu (bool): If True, use absolute μ values in wedge integration (default: True).
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object to plot on. 
                A new axis is created if None.
            r_factor (int): Exponent applied to the radial coordinate for scaling (default: 2).
            plot_kwargs (dict): Additional keyword arguments passed to `ax.plot()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}
            just_return_values (bool): If True, skip plotting and return the radial and 
                model ξ(r) values only.
            output_prefix (Path or str, optional): If set, used as the prefix path for 
                saving the output plot data (e.g., `.npz` file).
            save_data (bool): If True, save the plotted data to disk using `output_prefix`.
            save_dict (dict): Extra metadata to include in the saved `.npz` output file 
                (default: empty dict).
    
        Returns:
        ----------
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Radial distances (r) after wedge integration.
                - Scaled model cross-correlation values: r^r_factor * ξ(r).
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if fit_file == "" != correlation_file == "":
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
                abs_mu=abs_mu,
            )

        with fitsio.FITS(fit_file) as ffile:
            colnames = ffile["MODEL"].get_colnames()
            for name in (
                f"qsox{region}_MODEL",
                f"qsox{absorber}{region}_MODEL",
                f"{region}xqso_MODEL",
                f"{absorber}{region}xqso_MODEL",
            ):
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
        Plot the 2D correlation function model map from a Vega fit and correlation data,
        optionally saving the figure or returning raw matrix values.
    
        Arguments:
        ----------
            bookkeeper (Bookkeeper, optional): Object for managing file paths and metadata.
                If not provided, both `fit_file` and `correlation_file` must be set manually.
            fit_file (Path or str): Path to the Vega fit output file. Required if no bookkeeper is used.
            correlation_file (Path or str): Path to the correlation FITS file. 
                Required if no bookkeeper is used.
            region (str): Spectral region label for the first tracer (default: "lya").
            region2 (str, optional): Spectral region label for the second tracer. 
                Defaults to `region` if not specified.
            absorber (str): Absorber name for the first tracer (default: "lya").
            absorber2 (str, optional): Absorber name for the second tracer. 
                Defaults to `absorber` if not specified.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object to plot on.
                Required if `fig` is also set; if None, a new axis is created.
            r_factor (int): Exponent applied to the radial coordinate for scaling (default: 2).
            vmin (float): Minimum color scale value for the colormap (default: -0.04).
            vmax (float): Maximum color scale value for the colormap (default: 0.04).
            fig (matplotlib.figure.Figure, optional): Figure object for plotting. Required 
                if `ax` is set.
            plot_kwargs (dict): Additional keyword arguments passed to `imshow()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}            
            just_return_values (bool): If True, skip plotting and return extent and 
                scaled model matrix only.
            output_prefix (Path or str, optional): If set, used as the prefix path for 
                saving output files (e.g., `.png`, `.npz`).
            save_data (bool): If True, save the image matrix to a `.npz` file.
            save_plot (bool): If True, save the rendered plot as an image file.
            save_dict (dict): Additional metadata to include in the saved `.npz` file 
                (default: empty dict).
    
        Returns:
        ----------
            Tuple[Tuple[float, float, float, float], np.ndarray]: 
                - Image extent as (rt_min, rt_max, rp_min, rp_max).
                - Scaled model correlation matrix: r^r_factor * ξ(r_parallel, r_perpendicular).
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if fit_file == "" != correlation_file == "":
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
        Plot the line-of-sight (rp) correlation model (auto or cross), averaged over a specified 
        transverse (rt) range, from a Vega fit output and correlation data.
    
        Arguments:
        ----------
            bookkeeper (Bookkeeper, optional): Object to manage file paths and metadata.
                If not provided, both `fit_file` and `correlation_file` must be given.
            fit_file (Path or str): Path to the Vega model `.fit.pkl` file.
                Required if no bookkeeper is used.
            correlation_file (Path or str): Path to the 2D correlation `.fits` file.
                Required if no bookkeeper is used.
            auto (bool, optional): Whether to use the auto-correlation model. If False, 
                cross-correlation with quasars is assumed (default: False).
            region (str): Region name of the first tracer (default: "lya").
            region2 (str, optional): Region name of the second tracer. Defaults to `region`.
            absorber (str): Absorber name for the first tracer (default: "lya").
            absorber2 (str, optional): Absorber name for the second tracer. Defaults to `absorber`.
            rtmin (float): Minimum transverse separation (rt) to include in average (default: 0).
            rtmax (float): Maximum transverse separation (rt) to include in average (default: 1).
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, a new figure is created.
            r_factor (int): Not currently used in this function but reserved for compatibility (default: 2).
            plot_kwargs (dict): Additional keyword arguments passed to `imshow()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}    
            just_return_values (bool): If True, skip plotting and return the rp/data arrays directly.
            output_prefix (Path or str, optional): If set, prefix path to save output `.npz` data file.
            save_data (bool): If True, save the computed rp and data arrays to a `.npz` file.
            save_dict (dict): Additional metadata to include in the saved file.
    
        Returns:
        ----------
            Tuple[np.ndarray, np.ndarray]: 
                - `rp`: Array of line-of-sight distances.
                - `data`: Averaged correlation function values over the given rt wedge.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if region2 is None:
            region2 = region
        if absorber2 is None:
            absorber2 = absorber

        if fit_file != "" or correlation_file != "":
            if fit_file == "" != correlation_file == "":
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
                for name in (
                    f"qsox{region}_MODEL",
                    f"qsox{absorber}{region}_MODEL",
                    f"{region}xqso_MODEL",
                    f"{absorber}{region}xqso_MODEL",
                ):
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
        Plot a 2D cross-correlation model map as a function of line-of-sight (rp) and 
        transverse (rt) separations, weighted by a radial factor.
    
        This function retrieves and reshapes model output from a Vega `.fit.pkl` file, 
        aligns it to the dimensions of a correlation function `.fits` file, and plots 
        a colormap of the result.
    
        Arguments:
        ----------
            bookkeeper (Bookkeeper, optional): Object to manage file paths and metadata.
                If not provided, both `fit_file` and `correlation_file` must be given.
            fit_file (Path or str): Path to the Vega model `.fit.pkl` file. Required 
                if no bookkeeper is used.
            correlation_file (Path or str): Path to the 2D correlation `.fits` file.
                Required if no bookkeeper is used.
            region (str): Region name of the Lyman-alpha forest or tracer (default: "lya").
            absorber (str): Absorber name to cross-correlate with quasars (default: "lya").
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, one is created.
            r_factor (int): Exponential factor applied to the radial distance weighting in the colormap (default: 2).
            vmin (float): Minimum value for the colormap scale (default: -0.4).
            vmax (float): Maximum value for the colormap scale (default: 0.4).
            fig (matplotlib.figure.Figure, optional): Required if `ax` is given; figure where the axis belongs.
            plot_kwargs (dict): Additional keyword arguments passed to `imshow()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}    
            just_return_values (bool): If True, skip plotting and return extent and data matrix directly.
            output_prefix (Path or str, optional): If set, prefix path for saving output plot and/or data.
            save_data (bool): Whether to save the computed matrix to a `.npz` file.
            save_plot (bool): Whether to save the plot as a `.png` image.
            save_dict (dict): Extra metadata to include in the saved `.npz` file if `save_data` is True.
    
        Returns:
        ----------
            Tuple[Tuple[float, float, float, float], np.ndarray]:
                - `extent`: Plotting extent (rtmin, rtmax, rpmin, rpmax) for `imshow`.
                - `mat`: 2D model matrix of correlation values scaled by `r**r_factor`.
        """
        if output_prefix is not None:
            output_prefix = Path(output_prefix)

        if fit_file != "" or correlation_file != "":
            if fit_file == "" != correlation_file == "":
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
            for name in (
                f"qsox{region}_MODEL",
                f"qsox{absorber}{region}_MODEL",
                f"{region}xqso_MODEL",
                f"{absorber}{region}xqso_MODEL",
            ):
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

    @staticmethod
    def plot_errorbars_from_fit(
        readfits: List[ReadFits],
        param: str,
        param_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        reference: Optional[ReadFits] = None,
        blinded: bool = True,  # Add the blinded parameter, default is True
    ) -> List[matplotlib.container.Container | Any]:
        """
        Plot error bars for a specified fit parameter across multiple 
        `ReadFits` objects, optionally comparing to a reference fit.
    
        This function supports both blinded and unblinded plotting modes. 
        In blinded mode, it shifts values like `ap` and `at` relative to a reference and 
        suppresses their absolute scale. 
        In unblinded mode, it shows true parameter values and highlights reference bands. 
    
        Arguments:
        ----------
            readfits (List[ReadFits]): List of `ReadFits` objects containing fit 
                results and metadata to plot.
            param (str): Parameter key to plot (e.g., "ap", "at", or any other scalar).
            param_name (str, optional): Label to display for the x-axis. 
                If None, defaults to `param`.
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If not provided, 
                a new figure and axis will be created.
            plot_kwargs (dict): Additional keyword arguments passed to `ax.errorbar()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}    
            reference (ReadFits, optional): Reference `ReadFits` object used to define 
                shaded bands and baseline value.
            blinded (bool): If True, hides the absolute parameter values (e.g., shows 
                delta_ap = ap - baseline) and centers the reference at 0 (default: True).
    
        Returns:
        ----------
            List[matplotlib.container.Container | Any]: 
                List of matplotlib handles for plotted objects (e.g., error bars, lines).
        """
        if param_name is None:
            param_name = param

        if ax is None:
            fig, ax = plt.subplots()

        handles: List[Any] = []

        # Check if blinded boolean is true / false
        if blinded:
            # Blinded code block

            # Baseline setup
            baseline_value = None
            baseline_error = None

            if param in ("ap", "at") and reference:
                # Set the baseline using the first fit in the list
                baseline_value = reference.values.get(param, None)
                baseline_error = reference.errors.get(
                    param, 0
                )  # Assume first fit error is baseline error

                if baseline_value is None:
                    baseline_value = reference.model_header.get(param, None)
                if baseline_value is None and reference is not None:
                    baseline_value = reference.values.get(param, None)
                if baseline_value is None:
                    baseline_value = 0

                # Plot the shaded regions for ap/at, centered at 0
                ax.axvspan(
                    -baseline_error,
                    baseline_error,
                    alpha=0.2,
                    color="gray",
                )
                ax.axvspan(
                    -baseline_error / 3,
                    baseline_error / 3,
                    alpha=0.2,
                    color="red",
                )

                # Plot the baseline vertical line at 0
                handles.append(
                    ax.axvline(
                        0,  # Baseline is represented as 0
                        color="black",
                        ls="--",
                        lw=0.6,
                        alpha=1,
                        label="Baseline",
                    )
                )

            # Plot the error bars for each fit
            for i, fit in enumerate(readfits):
                value = fit.values.get(param, None)
                error = fit.errors.get(param, 0)

                if value is None:
                    value = fit.model_header.get(param, None)
                    if value is None and reference is not None:
                        value = reference.values.get(param, None)
                    error = 0

                if param in ("ap", "at") and baseline_value is not None:
                    value -= baseline_value  # Adjust value to be the difference from baseline
                    # Error remains the same

                if fit.colour is not None:
                    plot_kwargs = {**plot_kwargs, **dict(color=fit.colour)}

                # Plot the errorbar with adjusted value (difference from baseline for ap/at)
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

            # Adjust x-axis label for ap/at
            if param == "ap":
                ax.set_xlabel("delta_ap")
            elif param == "at":
                ax.set_xlabel("delta_at")
            else:
                ax.set_xlabel(param_name)

            # Add shaded regions to the 3rd, 4th, and 5th plots
            if reference is not None and param not in ("ap", "at"):
                if param in reference.values.keys():
                    value = reference.values.get(param, None)
                    error = reference.errors.get(param, 0)
                    ax.axvspan(
                        value - error,
                        value + error,
                        alpha=0.2,
                        color="gray",
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

            ax.set_yticks(range(len(readfits)))  # Label each fit
            ax.grid(visible=True)

        else:
            # Unblinded code block (this runs when blinded=False)
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
            ax.set_yticks([])  # No ticks on y-axis for unblinded case
            ax.grid(visible=True)

        return handles

    @staticmethod
    def plot_p_value_from_fit(
        readfits: List[ReadFits],
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        reference: Optional[ReadFits] = None,
    ) -> List[matplotlib.container.Container | Any]:
        """
        Plot horizontal bars showing p-values for a set of fits, optionally compared 
        to a reference value.
    
        A vertical dashed line indicates the reference p-value if provided. 
        Useful for comparing goodness-of-fit across multiple analyses or variations.
    
        Arguments:
        ----------
            readfits (List[ReadFits]): List of `ReadFits` objects containing p-value 
                results and metadata to plot.
            ax (matplotlib.axes.Axes, optional): Axis to plot on. If not provided, 
                a new figure and axis will be created.
            plot_kwargs (dict): Additional keyword arguments passed to `ax.errorbar()` for customization.
                For example:
                    plot_kwargs = {"ls": '--',
                                   "color": 'black',
                                   "aspect": "auto"}    
            reference (ReadFits, optional): Reference fit whose p-value is shown as a 
                vertical dashed line.
    
        Returns:
        ----------
            List[matplotlib.container.Container | Any]: 
                List of matplotlib handles for plotted objects (e.g., error bars, lines),
                useful for legend construction.
        """
        if ax is None:
            fig, ax = plt.subplots()

        handles: List[Any] = []

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
        """
        Plot a GetDist triangle plot for MCMC chains associated with ReadFits or provided directly.
    
        This generates 1D and 2D marginalized posterior distributions for the specified 
        parameters using `getdist`. You can pass either a list of `ReadFits` objects with 
        attached chains or the `chains` themselves directly. 
        
        Arguments:
        ----------
            param_names (List[str]): List of parameter names to plot (must match those in the chains).
            readfits (List[ReadFits], optional): List of fits with MCMC chains to plot. 
                If provided, chains will be extracted from `fit.chain`.
            chains (List[getdist.MCSamples], optional): List of `MCSamples` chains to plot directly.
            labels (List[str | None], optional): Labels for each chain. If `None`, uses `fit.label`.
            colours (List[str | None], optional): Contour colors for each chain. If `None`, uses `fit.colour`.
            g (getdist.plots.GetDistPlotter, optional): A `GetDistPlotter` instance. If not provided,
                a new one is created.
            plot_kwargs (dict): Additional keyword arguments forwarded to `g.triangle_plot`.
    
        Raises:
        ----------
            ValueError: If neither `readfits` nor `chains` are provided.
            AttributeError: If `fit.chain` is missing for any `ReadFits`.
    
        Returns:
        ----------
            None

        Note: 
        ----------
            Automatically sets labels and colors from the `ReadFits` metadata if not explicitly given.
        """
        
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
