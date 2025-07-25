"""
Module for reading, processing, and visualizing 'delta' data.

This file provides the ReadDeltas classes and plotting tools to access,
interpret, and visualize delta extraction results from FITS files, as produced
by the picca_bookkeeper framework. It supports handling configuration,
reading attributes, computing key statistics, and producing diagnostic plots.

Key classes:
 - ReadDeltas: Loads delta data products given a Bookkeeper instance or path,
               reads configuration and attribute files, computes statistics,
               and handles interpolation of relevant quantities
               (e.g., eta, fudge, mean continuum).
 - ReadDeltasNoBookkeeper: Variant enabling direct access via paths rather than
               through the Bookkeeper.
 - Plots: Static methods for generating plots of quantities like eta, var_lss,
               fudge, stack, mean continuum, etc., from the delta attribute files.

Main dependencies/interactions:
 - Relies on picca_bookkeeper.bookkeeper.Bookkeeper for tracking file locations
   and metadata.
 - Uses picca_bookkeeper.utils for continuum and spectra extraction functions.
 - Reads and interpolates data from delta FITS files and associated
   attribute / config files.
 - Designed to work with output from other modules, e.g., after continuum fitting
   and delta extraction, and supports visualization for analysis and validation.
 - Plots interact with matplotlib and FITS data.

Typical workflow:
 1. Instantiate ReadDeltas with a Bookkeeper object or direct paths.
 2. Load configuration and delta attribute files.
 3. Compute or aggregate statistics for delta products.
 4. Use Plots methods to visualize quantities for analysis.

"""
from __future__ import annotations

import itertools
import os
import warnings
from configparser import ConfigParser
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.utils import compute_cont, get_spectra_from_los_id

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple

    from picca_bookkeeper.hints import Axes, wave_grid, wave_grid_int, wave_grid_rf


class ReadDeltas:
    """
    Class for reading and processing delta extraction results.

    This class loads configurations, attributes, and relevant metadata
    required for downstream analysis and visualization.

    It supports interpolation of key arrays (like lambda grids), configuration
    parsing, and handles multiple calibration steps or blind-analysis modes.

    Parameters
    ----------
    bookkeeper : Bookkeeper, str, or Path
        A Bookkeeper object, or a path to the directory containing the
        bookkeeper run, from which to load delta outputs.
    region : str, optional
        Region name to identify the delta dataset.
        (e.g., "lya", "lyb", etc.)
    calib_step : int, optional
        Calibration step number, if applicable.
        (e.g. 1, 2)
    blind : bool, optional
        Whether the dataset is blinded (e.g., for blinded analyses).
    label : str, optional
        An optional label to identify the instance, used in display or debugging.
    """

    def __init__(
        self,
        bookkeeper: Bookkeeper | Path | str,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        blind: bool = False,
        label: Optional[str] = None,
    ):
        self.region = region
        self.calib_step = calib_step

        if isinstance(bookkeeper, Bookkeeper):
            self.bookkeeper = bookkeeper
        else:
            self.bookkeeper = Bookkeeper(bookkeeper)

        self.blind = blind
        self.set_var_lss_mod()
        self.define_lambda_arrays()
        self.define_interp_quantities()
        self.label = label

    def __str__(self) -> str:
        """
        Return a string representation of the instance
        (its label, if defined, 'None' if not provided).
        """
        return str(self.label)

    @property
    def config_file_in(self) -> Path:
        """
        Get the config file used for delta extraction input, based on region
        and calibration step.

        Returns
        -------
        Path
            Full path to the delta extraction config input file.
        """
        if self.calib_step is not None:
            return (
                self.bookkeeper.paths.run_path
                / "configs"
                / f"delta_extraction_{self.region}_calib_step_{self.calib_step}.ini"
            )
        else:
            return (
                self.bookkeeper.paths.run_path
                / "configs"
                / f"delta_extraction_{self.region}.ini"
            )

    @property
    def config_file(self) -> Path:
        """
        Get the internal delta extraction config.ini file written by the pipeline,
        in the deltas directory.
        """
        return self.deltas_path.parent / ".config.ini"

    @property
    def deltas_path(self) -> Path:
        """
        Return the full path to the deltas output directory based on the region
        and calibration step.
        """
        return self.bookkeeper.paths.deltas_path(
            region=self.region, calib_step=self.calib_step
        )

    @property
    def attributes_file(self) -> Path:
        """
        Return the path to the delta attributes file associated with this region
        and calibration step.
        """
        return self.bookkeeper.paths.delta_attributes_file(
            region=self.region, calib_step=self.calib_step
        )

    def set_var_lss_mod(self) -> None:
        """
        Read and store the 'var lss mod' value from the internal config.

        This value is used as a multiplicative modifier when computing
        variance due to large-scale structure (LSS).
        """
        config = ConfigParser()
        config.read(self.config_file)
        self.var_lss_mod = config["expected flux"].getfloat("var lss mod", 1)

    def define_lambda_arrays(self) -> None:
        """
        Define the observed and rest-frame wavelength grids and bin edges.

        Uses the config file to determine if the wavelength solution is linear
        or logarithmic, and computes arrays accordingly:
          - self.lambda_grid: central observed-frame wavelengths.
          - self.lambda_rf_grid: central rest-frame wavelengths.
          - self.lambda_edges: bin edges in observed frame.
          - self.lambda_rf_edges: bin edges in rest frame.
        """
        config = ConfigParser()
        config.read(self.config_file)

        if config["data"]["wave solution"] == "lin":
            self.lambda_grid = np.arange(
                float(config["data"]["lambda min"]),
                float(config["data"]["lambda max"])
                + float(config["data"]["delta lambda"]),
                float(config["data"]["delta lambda"]),
            )

            self.lambda_rf_grid = np.arange(
                float(config["data"]["lambda min rest frame"]),
                float(config["data"]["lambda max rest frame"])
                + float(config["data"]["delta lambda"]),
                float(config["data"]["delta lambda"]),
            )

        else:
            self.lambda_grid = np.arange(
                np.log10(float(config["data"]["lambda min"])),
                np.log10(float(config["data"]["lambda max"]))
                + float(config["data"]["delta log lambda"]),
                float(config["data"]["delta log lambda"]),
            )
            self.lambda_grid = 10**self.lambda_grid

            self.lambda_rf_grid = np.arange(
                np.log10(float(config["data"]["lambda min rest frame"])),
                np.log10(float(config["data"]["lambda max rest frame"]))
                + float(config["data"]["delta log lambda"]),
                float(config["data"]["delta log lambda"]),
            )
            self.lambda_rf_grid = 10**self.lambda_grid

        self.lambda_edges: np.ndarray = np.concatenate(
            (  # type: ignore
                ((1.5 * self.lambda_grid[0] - 0.5 * self.lambda_grid[1]),),
                0.5 * (self.lambda_grid[1:] + self.lambda_grid[:-1]),
                ((1.5 * self.lambda_grid[-1] - 0.5 * self.lambda_grid[-2]),),
            )
        )

        self.lambda_rf_edges: np.ndarray = np.concatenate(
            (  # type: ignore
                ((1.5 * self.lambda_rf_grid[0] - \
                 0.5 * self.lambda_rf_grid[1]),),
                0.5 * (self.lambda_rf_grid[1:] + self.lambda_rf_grid[:-1]),
                ((1.5 * self.lambda_rf_grid[-1] - \
                 0.5 * self.lambda_rf_grid[-2]),),
            )
        )

    def define_interp_quantities(self) -> None:
        """
        Define interpolated functions and arrays for key delta-related quantities.

        Reads various interpolated quantities from the delta attributes FITS file
        and builds interpolation functions on a uniform wavelength grid:

        Attributes set
        --------------
        eta : np.ndarray
            Interpolated eta values on `lambda_grid`, or ones if missing.
        fudge : np.ndarray
            Interpolated fudge values on `lambda_grid`, or zeros if missing.
        var_lss : np.ndarray
            Interpolated large-scale structure (LSS) variance on `lambda_grid`.
        mean_cont : np.ndarray
            Interpolated mean continuum on `lambda_rf_grid`.
        mean_cont_weight : np.ndarray
            Weights associated with the mean continuum.
        delta_stack : np.ndarray
            Interpolated stacked deltas (if available).
        delta_stack_weight : np.ndarray
            Corresponding weights for stacked deltas (if available).

        Notes
        -----
        This method can handle missing columns, defaulting to zeros or ones.
        All interpolations use `fill_value='extrapolate' to cover the full
        wavelength range.
        """
        with fitsio.FITS(self.attributes_file) as attrs_fits:
            if "ETA" in [
                name.upper() for name in attrs_fits["VAR_FUNC"].get_colnames()
            ]:
                self.eta_interp = interp1d(
                    10 ** attrs_fits["VAR_FUNC"]["loglam"][:],
                    attrs_fits["VAR_FUNC"]["eta"][:],
                    fill_value="extrapolate",
                )
                self.eta = self.eta_interp(self.lambda_grid)
            else:
                self.eta = np.ones_like(self.lambda_grid)

            if "FUDGE" in [
                name.upper() for name in attrs_fits["VAR_FUNC"].get_colnames()
            ]:
                self.fudge_interp = interp1d(
                    10 ** attrs_fits["VAR_FUNC"]["loglam"][:],
                    attrs_fits["VAR_FUNC"]["fudge"][:],
                    fill_value="extrapolate",
                )
                self.fudge = self.fudge_interp(self.lambda_grid)
            else:
                self.fudge = 0 * np.ones_like(self.lambda_grid)

            self.var_lss_interp = interp1d(
                10 ** attrs_fits["VAR_FUNC"]["loglam"][:],
                attrs_fits["VAR_FUNC"]["var_lss"][:],
                fill_value="extrapolate",
            )
            self.var_lss = self.var_lss_interp(self.lambda_grid)

            self.mean_cont_interp = interp1d(
                10 ** attrs_fits["CONT"]["loglam_rest"][:],
                attrs_fits["CONT"]["mean_cont"][:],
                fill_value="extrapolate",
            )

            self.mean_cont_weight_interp = interp1d(
                10 ** attrs_fits["CONT"]["loglam_rest"][:],
                attrs_fits["CONT"]["weight"][:],
                fill_value="extrapolate",
            )

            self.mean_cont = self.mean_cont_interp(self.lambda_rf_grid)
            self.mean_cont_weight = self.mean_cont_weight_interp(
                self.lambda_rf_grid)

            if "STACK_DELTAS" in attrs_fits:
                self.delta_stack_interp = interp1d(
                    10 ** attrs_fits["STACK_DELTAS"]["loglam"][:],
                    attrs_fits["STACK_DELTAS"]["stack"][:],
                    fill_value="extrapolate",
                )

                self.delta_stack_weight_interp = interp1d(
                    10 ** attrs_fits["STACK_DELTAS"]["loglam"][:],
                    attrs_fits["STACK_DELTAS"]["weight"][:],
                    fill_value="extrapolate",
                )

                self.delta_stack = self.delta_stack_interp(self.lambda_grid)
                self.delta_stack_weight = self.delta_stack_weight_interp(
                    self.lambda_grid
                )

    def get_statistics(
        self, downsampling: float = 1, random_seed: int = 0, read_flux: bool = False
    ) -> None:
        """
        Compute and aggregate statistics from all delta files.

        Loads per-sightline delta data, variance, metadata, and optional flux information
        across all delta files in parallel, applying optional downsampling.

        Parameters
        ----------
        downsampling : float, optional
            Fraction of sightlines to randomly retain
            (default is 1.0 = no downsampling).
        random_seed : int, optional
            Seed for reproducible downsampling.
        read_flux : bool, optional
            If True, load flux and inverse variance data from the companion
            flux files.

        Attributes set
        --------------
        statistics : list
            List of tuples returned by `get_file_statistics` for each file.
        deltas_arr : ndarray
            Array of delta values across all sightlines.
        var_pipe_arr : ndarray
            Pipeline variance values.
        weights_arr : ndarray
            Inverse variance weights.
        lambda_rf_arr : ndarray
            Rest-frame wavelength grids per sightline.
        id_arr : ndarray
            LOS IDs of sightlines.
        cont_arr : ndarray
            Continuum estimates.
        flux_arr : ndarray
            Flux values (zeros if `read_flux=False`).
        flux_ivar_arr : ndarray
            Flux inverse variances (zeros if `read_flux=False`).
        z_arr : ndarray
            Redshifts of each sightline.
        meansnr_arr : ndarray
            Mean signal-to-noise ratios.
        deltas2_arr : ndarray
            Squared delta values.
        """
        delta_files = list(self.deltas_path.glob("delta*fits*"))

        pool = Pool(processes=5)
        self.statistics = pool.starmap(
            self.get_file_statistics,
            zip(
                delta_files,
                itertools.repeat(downsampling),
                itertools.repeat(random_seed),
                itertools.repeat(read_flux),
            ),
        )
        pool.close()

        self.deltas_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[0])
        self.var_pipe_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[1])
        self.weights_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[2])
        self.lambda_rf_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[3])
        self.id_arr = np.concatenate(np.asarray(
            self.statistics, dtype="object").T[4])
        self.cont_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[5])
        self.flux_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[6])
        self.flux_ivar_arr = np.vstack(np.asarray(
            self.statistics, dtype="object").T[7])
        self.z_arr = np.concatenate(np.asarray(
            self.statistics, dtype="object").T[8])
        self.meansnr_arr = np.concatenate(
            np.asarray(self.statistics, dtype="object").T[9]
        )
        self.deltas2_arr = self.deltas_arr**2

    def get_file_statistics(
        self,
        deltas_filename: Path,
        downsampling: float = 1,
        random_seed: int = 0,
        read_flux: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Extract statistics from a single delta FITS file.

        Computes quantities such as deltas, variances, rest-frame wavelengths,
        and optionally flux values for each sightline.

        Parameters
        ----------
        deltas_filename : Path
            Path to the delta FITS file.
        downsampling : float, optional
            Fraction of sightlines to retain (default is 1.0).
        random_seed : int, optional
            Seed for reproducible sampling.
        read_flux : bool, optional
            Whether to load associated flux data from flux-info files.

        Returns
        -------
        Tuple of arrays containing:
            - delta values
            - pipeline variance
            - weights
            - rest-frame wavelengths
            - LOS IDs
            - continuum estimates
            - flux values (zeros if `read_flux` is False)
            - flux inverse variances (zeros if `read_flux` is False)
            - redshifts
            - mean signal-to-noise ratios

        Notes
        -----
        Uses a quadratic equation to invert for pipeline variance based on
        total variance and LSS/eta/fudge parameters.
        Applies optional downsampling mask to simulate reduced dataset.
        """
        delta_field = "DELTA" if not self.blind else "DELTA_BLIND"
        with fitsio.FITS(deltas_filename) as delta_fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                VAR = 1 / delta_fits["WEIGHT"].read()
                var_pipe = (
                    VAR
                    - self.var_lss * self.var_lss_mod
                    + np.sqrt(
                        (self.var_lss * self.var_lss_mod - VAR) ** 2
                        - 4 * self.eta * self.fudge
                    )
                ) / (2 * self.eta)
                # var_pipe = (VAR - self.var_lss) / self.eta

            lambda_rf = np.asarray(
                [self.lambda_grid / (1 + x)
                 for x in delta_fits["METADATA"]["Z"][:]]
            )

            if downsampling != 1:
                np.random.seed(random_seed)
                msk = np.random.choice(
                    [True, False],
                    delta_fits["METADATA"].read_header()["NAXIS2"],
                    p=[downsampling, 1 - downsampling],
                )
            else:
                msk = np.ones(
                    delta_fits["METADATA"].read_header()["NAXIS2"], dtype=bool
                )

            if not read_flux:
                return (
                    delta_fits[delta_field].read()[msk],
                    var_pipe[msk],
                    delta_fits["WEIGHT"].read()[msk],
                    lambda_rf[msk],
                    delta_fits["METADATA"]["LOS_ID"][:][msk],
                    delta_fits["CONT"].read()[msk],
                    np.zeros_like(lambda_rf[msk]),
                    np.zeros_like(lambda_rf[msk]),
                    delta_fits["METADATA"]["Z"][:][msk],
                    delta_fits["METADATA"]["MEANSNR"][:][msk],
                )
            else:
                healpix = deltas_filename.name[6:]
                with fitsio.FITS(
                    deltas_filename.parent.parent /
                        "Flux" / f"flux-info-{healpix}"
                ) as flux_fits:
                    return (
                        delta_fits[delta_field].read()[msk],
                        var_pipe[msk],
                        delta_fits["WEIGHT"].read()[msk],
                        lambda_rf[msk],
                        delta_fits["METADATA"]["LOS_ID"][:][msk],
                        delta_fits["CONT"].read()[msk],
                        flux_fits["FLUX"].read()[msk],
                        flux_fits["FLUX_IVAR"].read()[msk],
                        delta_fits["METADATA"]["Z"][:][msk],
                        delta_fits["METADATA"]["MEANSNR"][:][msk],
                    )


class ReadDeltasNoBookkeeper(ReadDeltas):
    """
    Variant of ReadDeltas that does not require a Bookkeeper instance.

    This subclass allows users to directly specify paths to delta products
    without relying on a Bookkeeper object.

    Parameters
    ----------
    deltas_path : str or Path
        Path to the directory containing delta FITS files.
    log_path : str or Path
        Path to the directory containing delta attribute files.
    config_file : str or Path
        Path to the internal config file (typically `...config.ini`).
    region : str
        The region label (e.g., 'lya').
    config_file_in : str or Path, optional
        Optional input config file, used for reproducibility.
    blind : bool, optional
        Whether to use blinded deltas (e.g., DELTA_BLIND instead of DELTA).
    label : str, optional
        Optional label for this instance (for debugging/logging).
    """

    def __init__(
        self,
        deltas_path: Path | str,
        log_path: Path | str,
        config_file: Path | str,
        region: str,
        config_file_in: Optional[Path | str] = None,
        blind: bool = False,
        label: Optional[str] = None,
    ):
        self._config_file = Path(config_file)
        self._config_file_in = (
            Path(config_file_in) if config_file_in is not None else Path(config_file)
        )
        self._attributes_file = Path(log_path) / "delta_attributes.fits.gz"
        self._deltas_path = Path(deltas_path)

        super().__init__(
            bookkeeper=Path(deltas_path), region=region, blind=blind, label=label
        )

    @property
    def config_file_in(self) -> Path:
        """
        Return the path to the original delta extraction config file.
        """
        return self._config_file_in

    @property
    def config_file(self) -> Path:
        """
        Return the path to the internal delta config file (.config.ini).
        """
        return self._config_file

    @property
    def deltas_path(self) -> Path:
        """
        Return the path to the directory containing delta FITS files.
        """
        return self._deltas_path

    @property
    def attributes_file(self) -> Path:
        """
        Return the path to the delta attributes file.
        """
        return self._attributes_file


class Plots:
    """
    Collection of static plotting methods for visualizing delta attributes.

    Each method provides a quick interface for visualizing a specific quantity
    (e.g., η, variance, fudge factor, or stacked deltas) from delta attribute files.

    These functions can be used with or without a `Bookkeeper` object, making them
    suitable for exploratory analysis, debugging, or higher quality plots.
    """
    @staticmethod
    def eta(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid, wave_grid]:
        """
        Plot η (eta) as a function of wavelength.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Bookkeeper instance to locate the attribute file.
        region : str, optional
            Region label (e.g., "lya").
        calib_step : int, optional
            Calibration step number (e.g. 1, 2).
        attributes_file : Path or str, optional
            Path to the attribute FITS file (overrides bookkeeper object).
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, a new figure is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
            Example: `{"color": "black",
                       "ls": "--",
                       "label": "η"}`

        Returns
        -------
        wave : ndarray
            Wavelength grid (in Angstroms).
        eta : ndarray
            eta (η) values at each wavelength.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            if "VAR_FUNC" in hdul:
                card = "VAR_FUNC"
            else:
                card = "WEIGHT"

            wave = 10 ** hdul[card]["LOGLAM"].read()
            eta = hdul[card]["ETA"].read()
            ax.plot(
                wave,
                eta,
                **plot_kwargs,
            )

        ax.set_xlabel(r"$\lambda \, [\AA]$")
        ax.set_ylabel(r"$\eta$")

        return wave, eta

    @staticmethod
    def var_lss(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid, wave_grid]:
        """
        Plot the large-scale structure (LSS) variance as a function of wavelength.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Bookkeeper instance to locate the attribute file.
        region : str, optional
            Region label (e.g., "lya").
        calib_step : int, optional
            Calibration step number. (e.g. 1, 2)
        attributes_file : Path or str, optional
            Path to the attribute FITS file (overrides bookkeeper).
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, a new figure is created.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
            Example: `{"color": "black",
                       "ls": "--",
                       "label": "η"}`

        Returns
        -------
        wave : ndarray
            Wavelength grid (in Angstroms).
        var_lss : ndarray
            LSS variance values (σ_LSS^2) at each wavelength.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            if "STATS" in hdul:
                wave = hdul["STATS"]["LAMBDA"].read()
                var_lss = hdul["STATS"]["VAR"].read()
            else:
                if "VAR_FUNC" in hdul:
                    card = "VAR_FUNC"
                else:
                    card = "WEIGHT"

                wave = 10 ** hdul[card]["LOGLAM"].read()
                var_lss = hdul[card]["VAR_LSS"].read()

        ax.plot(
            wave,
            var_lss,
            **plot_kwargs,
        )

        ax.set_xlabel(r"$\lambda \, [\AA]$")
        ax.set_ylabel(r"$\sigma^2_{\rm LSS}$")

        return wave, var_lss

    @staticmethod
    def fudge(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid, wave_grid]:
        """
        Plot the fudge factor (ε(λ)), used to tune error modeling.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Bookkeeper instance to locate the attribute file.
        region : str, optional
            Region label (e.g., "lya").
        calib_step : int, optional
            Calibration step number. (e.g. 1, 2)
        attributes_file : Path or str, optional
            Path to the attribute FITS file (overrides bookkeeper).
        ax : matplotlib.axes.Axes, optional
            Axes object to draw on. If None, a new one is created.
        plot_kwargs : dict, optional
            Keyword arguments passed to `ax.plot()`
            (e.g., `color`, `linestyle`).


        Returns
        -------
        lambda_ : ndarray
            Wavelength grid (Angstroms).
        fudge : ndarray
            Fudge factor values (ε(λ)).
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            if "VAR_FUNC" in hdul:
                card = "VAR_FUNC"
            else:
                card = "WEIGHT"

            lambda_ = 10 ** hdul[card]["LOGLAM"].read()
            fudge = hdul[card]["FUDGE"].read()

        ax.plot(
            lambda_,
            fudge,
            **plot_kwargs,
        )

        ax.set_xlabel(r"$\lambda \, [\AA]$")
        ax.set_ylabel(r"$\epsilon$")

        return lambda_, fudge

    @staticmethod
    def stack(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        rebin: Optional[int] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        use_weights: bool = False,
        offset: float = 0,
    ) -> Tuple[wave_grid, wave_grid]:
        """
        Plot the stacked mean delta (1 + δ_q) as a function of wavelength.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Bookkeeper instance used to determine file location.
        region : str, optional
            Region label (e.g., "lya").
        calib_step : int, optional
            Calibration step number. (e.g. 1, 2)
        attributes_file : Path or str, optional
            Path to the delta attribute file.
        rebin : int, optional
            If set, rebin the data by averaging every `rebin` points.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new one is created if None.
        plot_kwargs : dict, optional
            Passed to `ax.plot()` for styling.
                Example: `{"label": "mean(1 + δ)",
                           "color": "C0"}`
        use_weights : bool, optional
            Whether to weight the stack by the provided weights.
        offset : float, optional
            Add a vertical offset to the stack values for multi-line plots.

        Returns
        -------
        lambda_ : ndarray
            Wavelength grid (Angstroms).
        stack : ndarray
            Mean delta (or weighted mean) at each wavelength.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            if "STATS" in hdul:
                # Raw deltas
                lambda_ = hdul["STATS"]["LAMBDA"].read()
                stack = hdul["STATS"]["MEANFLUX"].read()

                if use_weights:
                    weights = hdul["STATS"]["WEIGHTS"].read()
            else:
                lambda_ = 10 ** hdul["STACK_DELTAS"]["loglam"].read()
                stack = hdul["STACK_DELTAS"]["stack"].read()

                if use_weights:
                    weights = hdul["STACK_DELTAS"]["weight"].read()

        if rebin is not None:
            # Repeat last values till having a number of data multiple of rebin.
            lambda_ = np.concatenate(
                (lambda_, lambda_[-1] * np.ones(rebin - len(lambda_) % rebin))
            )
            stack = np.concatenate(
                (stack, stack[-1] * np.ones(rebin - len(stack) % rebin))
            )

            lambda_ = np.mean(lambda_.reshape(-1, rebin), axis=1)
            stack = np.mean(stack.reshape(-1, rebin), axis=1)

            if use_weights:
                weights = np.concatenate(
                    (weights, np.full(rebin - len(weights) %
                     rebin, weights[-1]))
                )
                weights = np.mean(weights.reshape(-1, rebin), axis=1)

        if use_weights:
            stack *= weights

        ax.plot(lambda_, stack + offset, **plot_kwargs)
        ax.set_xlabel(r"$\lambda \, [\AA]$")
        ax.set_ylabel(r"$\overline{1 + \delta_q(\lambda)}$")

        return lambda_, stack

    @staticmethod
    def num_pixels(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid, wave_grid_int]:
        """
        Plot the number of contributing pixels per wavelength bin.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Bookkeeper instance to locate the attributes file.
        region : str, optional
            Region name (e.g., "lya").
        calib_step : int, optional
            Calibration step index. (e.g. 1,2)
        attributes_file : Path or str, optional
            Direct path to attribute FITS file.
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw the plot. A new one is created if None.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
                Example: `{"color": "black",
                           "ls": "--"}.

        Returns
        -------
        lambda_ : ndarray
            Wavelength grid in Angstroms.
        num_pixels : ndarray
            Number of pixels contributing to each wavelength bin.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            lambda_ = 10 ** hdul["VAR_FUNC"]["loglam"].read()
            num_pixels = hdul["VAR_FUNC"]["num_pixels"].read()

        ax.plot(
            lambda_,
            num_pixels,
            **plot_kwargs,
        )
        ax.set_xlabel(r"$\lambda \, [\AA]$")
        ax.set_ylabel(r"# pixels")

        return lambda_, num_pixels

    @staticmethod
    def valid_fit(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> None:
        """
        Plot the fraction of pixels with a valid fit for each wavelength bin.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Used to locate the attribute file path.
        region : str, optional
            Spectral region (e.g., "lya").
        calib_step : int, optional
            Calibration step number. (e.g. 1,2)
        attributes_file : Path or str, optional
            Path to delta attributes FITS file.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new one is created if None.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
                Example: `{"color": "black",
                           "ls": "--"}.

        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            ax.plot(
                10 ** hdul["VAR_FUNC"]["loglam"].read(),
                hdul["VAR_FUNC"]["valid_fit"].read(),
                **plot_kwargs,
            )

    @staticmethod
    def mean_cont(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid_rf, wave_grid_rf]:
        """
        Plot the mean continuum in rest-frame wavelength.

        Parameters
        ----------
        bookkeeper : Bookkeeper, optional
            Instance to fetch the attribute file path.
        region : str, optional
            Spectral region (e.g., "lya").
        calib_step : int, optional
            Calibration step index. (e.g. 1,2)
        attributes_file : Path or str, optional
            Delta attribute FITS file location.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new one is created if None.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
                Example: `{"color": "black",
                           "ls": "--"}.

        Returns
        -------
        wave : ndarray
            Rest-frame wavelength grid.
        mean_cont : ndarray
            Mean continuum per bin.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if isinstance(bookkeeper, Bookkeeper):
            attributes_file = bookkeeper.paths.delta_attributes_file(
                region=region, calib_step=calib_step
            )

        with fitsio.FITS(attributes_file) as hdul:
            wave = 10 ** hdul["CONT"]["loglam_rest"].read()
            mean_cont = hdul["CONT"]["mean_cont"].read()

        ax.plot(
            wave,
            mean_cont,
            **plot_kwargs,
        )

        return wave, mean_cont

    @staticmethod
    def line_masking(
        mask_file: Path
        | str = Path(str(os.getenv("pr")))
        / "Continuum_fitting/config_files/sharp-lines-mask.txt",
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        use_labels: bool = False,
        lambda_lim: Optional[List[float]] = None,
        standard_width: Optional[float] = None,
    ) -> None:
        """
        Overlay vertical shaded bands for known masked regions (e.g., sky lines).

        Parameters
        ----------
        mask_file : Path or str, optional
            File containing the list of masked lines and ranges.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new one is created if None.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.axvspan()` for customization.
                Example: `{"color": "black",
                           "alpha": "0.5"}.
        use_labels : bool, optional
            If True, show labels for each line category (once).
        lambda_lim : list of float, optional
            Only mask lines within this wavelength range [min, max].
        standard_width : float, optional
            If set, override individual line widths with a uniform width.

        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()

        with open(mask_file) as f:
            lines = f.readlines()
        colors = dict(K="green", H="pink", Na="olive", sky="cyan")

        if use_labels:
            label_used = dict()
            for key in colors.keys():
                label_used[key] = False

        for line in lines:
            if line[0] == "#":
                continue

            label = line.split(" ")[0]
            if use_labels:
                label_to_use = label if not label_used[label] else None
                label_used[label] = True
            else:
                label_to_use = None

            lambda_min = float(line.split(" ")[-3])
            lambda_max = float(line.split(" ")[-2])

            if lambda_lim is not None:
                if (lambda_max < lambda_lim[0]) or (lambda_min > lambda_lim[1]):
                    continue

            if standard_width:
                mean_lambda = 0.5 * (lambda_max + lambda_min)
                lambda_min = mean_lambda - 0.5 * standard_width
                lambda_max = mean_lambda + 0.5 * standard_width

            ax.axvspan(
                lambda_min,
                lambda_max,
                facecolor=colors[label],
                label=label_to_use,
                alpha=0.4,
            )

    @staticmethod
    def plot_flux(
        bookkeeper: Bookkeeper,
        los_id: int,
        plot_kwargs: Dict = dict(),
        ax: Optional[Axes] = None,
        z: Optional[float] = None,
    ) -> None:
        """
        Plot raw observed-frame flux for a single line of sight.

        Parameters
        ----------
        bookkeeper : Bookkeeper
            Used to locate catalog and spectra paths.
        los_id : int
            Line-of-sight ID.
        plot_kwargs : dict, optional
            Additional keyword arguments passed to `ax.plot()` for customization.
                Example: `{"color": "black",
                           "ls": "--"}.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. A new one is created if None.
        z : float, optional
            If set, shift to rest-frame using given redshift (z).

        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()

        flux_data = get_spectra_from_los_id(
            los_id,
            catalog=bookkeeper.paths.catalog,
            input_healpix=bookkeeper.paths.healpix_data,
        )

        if z is not None:
            grid = flux_data[0] / (1 + z)
        else:
            grid = flux_data[0]

        ax.plot(
            grid,
            flux_data[1],
            **plot_kwargs,
        )

    @staticmethod
    def plot_cont(
        los_id: int,
        attrs_file: Path | str,
        ax: Optional[Axes] = None,
        region: str = "lya",
        z: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[wave_grid_rf, wave_grid_rf]:
        """
        Helper function to compute and plot the fitted continuum for a given LOS.

        Parameters
        ----------
        los_id : int
            Line-of-sight ID for the quasar.
        attrs_file : Path or str
            Path to the delta attributes file.
        ax : matplotlib.axes.Axes, optional
            Axes object to draw the plot. If none, one will be created.
        region : str, optional
            Region in which to compute the continuum (default "lya").
        z : float, optional
            If provided, transform to observed frame using provided redshift (z).
        **kwargs : dict
            Additional keyword arguments for `ax.plot()`.

        Returns
        -------
        wave : ndarray
            Wavelength array (rest or observed frame).
        cont : ndarray
            Continuum values.
        """
        if ax is None:
            fig, ax = plt.subplots()

        lambda_rest, cont = compute_cont(los_id, attrs_file, region)

        if z is None:
            ax.plot(lambda_rest, cont, **kwargs)
            return lambda_rest, cont
        else:
            ax.plot(lambda_rest * (1 + z), cont, **kwargs)
            return lambda_rest * (1 + z), cont
