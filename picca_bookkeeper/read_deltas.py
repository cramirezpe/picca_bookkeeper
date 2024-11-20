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
        return str(self.label)

    @property
    def config_file_in(self) -> Path:
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
        return self.deltas_path.parent / ".config.ini"

    @property
    def deltas_path(self) -> Path:
        return self.bookkeeper.paths.deltas_path(
            region=self.region, calib_step=self.calib_step
        )

    @property
    def attributes_file(self) -> Path:
        return self.bookkeeper.paths.delta_attributes_file(
            region=self.region, calib_step=self.calib_step
        )

    def set_var_lss_mod(self) -> None:
        config = ConfigParser()
        config.read(self.config_file)
        self.var_lss_mod = config["expected flux"].getfloat("var lss mod", 1)

    def define_lambda_arrays(self) -> None:
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
                ((1.5 * self.lambda_rf_grid[0] - 0.5 * self.lambda_rf_grid[1]),),
                0.5 * (self.lambda_rf_grid[1:] + self.lambda_rf_grid[:-1]),
                ((1.5 * self.lambda_rf_grid[-1] - 0.5 * self.lambda_rf_grid[-2]),),
            )
        )

    def define_interp_quantities(self) -> None:
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
            self.mean_cont_weight = self.mean_cont_weight_interp(self.lambda_rf_grid)

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

        self.deltas_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[0])
        self.var_pipe_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[1])
        self.weights_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[2])
        self.lambda_rf_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[3])
        self.id_arr = np.concatenate(np.asarray(self.statistics, dtype="object").T[4])
        self.cont_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[5])
        self.flux_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[6])
        self.flux_ivar_arr = np.vstack(np.asarray(self.statistics, dtype="object").T[7])
        self.z_arr = np.concatenate(np.asarray(self.statistics, dtype="object").T[8])
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
                [self.lambda_grid / (1 + x) for x in delta_fits["METADATA"]["Z"][:]]
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
                    deltas_filename.parent.parent / "Flux" / f"flux-info-{healpix}"
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
        return self._config_file_in

    @property
    def config_file(self) -> Path:
        return self._config_file

    @property
    def deltas_path(self) -> Path:
        return self._deltas_path

    @property
    def attributes_file(self) -> Path:
        return self._attributes_file


class Plots:
    @staticmethod
    def eta(
        bookkeeper: Optional[Bookkeeper] = None,
        region: Optional[str] = None,
        calib_step: Optional[int] = None,
        attributes_file: Optional[Path | str] = None,
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
    ) -> Tuple[wave_grid, wave_grid]:
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
                    (weights, np.full(rebin - len(weights) % rebin, weights[-1]))
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
        mask_file: Path | str = Path(str(os.getenv("pr")))
        / "Continuum_fitting/config_files/sharp-lines-mask.txt",
        ax: Optional[Axes] = None,
        plot_kwargs: Dict = dict(),
        use_labels: bool = False,
        lambda_lim: Optional[List[float]] = None,
        standard_width: Optional[float] = None,
    ) -> None:
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
        """Helper function to compute continuum

        Arguments:
        ----------
        ax: matplotlib axis where to draw the pot.
        los_id: Line of sight id of quasar.
        attrs_file: Path to delta attributes file.
        region: Region where to compute the continuum.
        z: Redshfit to plot in observed wavelength.
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
