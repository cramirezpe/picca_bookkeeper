"""
Module for reading and analyzing DESI QSO spectra masks.

This file provides tools to:
    - Read flux, inverse variance, mask, and wavelength data from DESI coadd
      files for a set of given QSO target IDs, supporting parallel processing
      and optional downsampling.
    - Parse a QSO catalogue and organize targets by HEALPIX and survey.
    - Produce visualizations and summary statistics on masked pixels, including
      percentage plots and histograms of QSOs per masked bin, with options to
      save plots and/or data.

Main classes and functions:
    - ReadDESILyaData: Handles catalogue parsing, file grouping, and bulk
      data extraction.
    - Plots: Static methods for generating and saving diagnostic plots of
      mask properties.
    - read_file: Reads individual DESI coadd files for specified target IDs.
    - main/getArgs: CLI entry point and argument parser for running analyses.

Typical usage:
    - Import and use ReadDESILyaData for data access, or run as a standalone
      script to generate mask statistics and plots from catalogue and DESI
      input directory.

See CLI help (`python read_desi_masks.py --help`) for run-time options and
parameters.
"""
from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def read_file(file: Path | str, valid_targetids: List[str | int]) -> np.ndarray:
    """
    Read relevant spectral data from a DESI coadd FITS file for a set of
    target IDs.

    Parameters
    ----------
    file : Path or str
        Path to a DESI coadd file.
    valid_targetids : list of str or int
        Target IDs to extract from the file.

    Returns
    -------
    np.ndarray
        Array containing the following, in order for each color channel (B, R, Z):
            - wavelength array
            - flux array (for valid target IDs)
            - inverse variance array (for valid target IDs)
            - mask array (for valid target IDs)
        Followed by the list of matched target IDs.

    Notes
    -----
    This function is typically used in parallel to process multiple coadd files.
    """
    ffile = fitsio.FITS(file)
    targetid = ffile["FIBERMAP"]["TARGETID"].read()

    in1d = np.in1d(targetid, valid_targetids)

    values = []
    for color in "B", "R", "Z":
        values.append(ffile[f"{color}_WAVELENGTH"].read())
        values.append(ffile[f"{color}_FLUX"].read()[in1d])
        values.append(ffile[f"{color}_IVAR"].read()[in1d])
        values.append(ffile[f"{color}_MASK"].read()[in1d])

    values.append(targetid[in1d])

    return np.asarray(values)


class Plots:
    """
    Visualization tools for analyzing masking statistics in Lyman-alpha forest data.

    This class provides diagnostic plots focused on masking behavior in QSO spectra,
    such as:
        - The percentage of masked pixels per quasar or redshift bin.
        - Histograms showing the number of quasars affected by different
          levels of masking.

    Attributes:
    ----------
        bk (Bookkeeper | FakeBookkeeper): Interface to correlation and masking metadata.
        config (dict): Configuration dictionary specifying plotting and data behavior.
        suffix (str): Optional file suffix for identifying output or variations.
        tracer (str): Tracer type, typically 'qso' for quasar analyses.
    """
    @staticmethod
    def masked_pixels_percentage(
        properties: Dict,
        output_prefix: Path,
        plot_kwargs: Dict = dict(),
        downsampling: float = 1,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot the percentage of masked pixels per wavelength bin, for B, R, Z bands.

        Parameters
        ----------
        properties : dict
            Dictionary with keys like "B_WAVELENGTH", "B_MASK", etc. Each band
            should have associated wavelength and mask arrays with shape
            (n_qsos, n_pix).
        output_prefix : Path
            Prefix path used to save plots or data files.
        plot_kwargs : dict, optional
            Extra keyword arguments passed to `ax.plot()` for customization.
        downsampling : float, optional
            Fraction of the data used in the analysis (e.g., 0.1 means 10% of QSOs).
            Used for labeling purposes only.
        save_data : bool, optional
            If True, save the numerical values used in the plot to a .npz file.
        save_plot : bool, optional
            If True, save the resulting figure as a .png file.
        save_dict : dict, optional
            Additional metadata or variables to include in the saved .npz file.

        Returns
        -------
        None
        """
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 5))

        for ax, color in zip(axs, ("B", "R", "Z")):
            ax.plot(
                properties[f"{color}_WAVELENGTH"],
                (properties[f"{color}_MASK"] != 0).sum(axis=0)
                / len(properties[f"{color}_MASK"])
                * 100,
                **plot_kwargs,
            )
            ax.set_title(f"{color}-band")

        axs[0].set_ylabel(r"% of masked pixels")
        fig.supxlabel(r"$\lambda \, [\AA]$")
        if downsampling < 1:
            plt.suptitle(
                f"Percentage of masked pixels for a subsample of fugu data (subsample {round(downsampling*100, 2)}%)"
            )
        else:
            plt.suptitle("Percentage of masked pixels")

        plt.tight_layout()
        if save_plot:
            plt.savefig(
                output_prefix.parent
                / (output_prefix.name + "-masked_pixels_percentage.png"),
                dpi=300,
            )

        if save_data:
            data_dict = {}
            for color in "B", "R", "Z":
                data_dict[f"{color}_WAVELENGTH"] = properties[f"{color}_WAVELENGTH"]
                data_dict[f"{color}_VALUE"] = (
                    (properties[f"{color}_MASK"] != 0).sum(axis=0)
                    / len(properties[f"{color}_MASK"])
                    * 100
                )

            np.savez(
                output_prefix.parent
                / (output_prefix.name + "-masked_pixels_percentage.npz"),
                **{**save_dict, **data_dict},
            )

    @staticmethod
    def histogram_qsos_per_masked_bins(
        properties: Dict,
        output_prefix: Path,
        hist_kwargs: Dict = dict(),
        downsampling: float = 1,
        save_data: bool = False,
        save_plot: bool = False,
        save_dict: Dict = dict(),
    ) -> None:
        """
        Plot histograms of the number of masked pixels per QSO, for each band
        (B, R, Z).

        Parameters
        ----------
        properties : dict
            Dictionary containing "B_MASK", "R_MASK", and "Z_MASK" arrays of shape
            (n_qsos, n_pix). Each value is a bitmask array.
        output_prefix : Path
            Prefix used to construct filenames for saved plots or data files.
        hist_kwargs : dict, optional
            Extra keyword arguments for `ax.hist()`, such as `bins`, `color`, etc.
            Defaults to bins spanning 0 to 80 in 10 steps.
        downsampling : float, optional
            Fraction of data used, included in the figure title for clarity.
        save_data : bool, optional
            If True, saves histogram data to a .npz file.
        save_plot : bool, optional
            If True, saves the histogram plot as a .png file.
        save_dict : dict, optional
            Extra metadata to include in saved .npz data.

        Returns
        -------
        None
        """
        if save_data:
            data_dict = {}

        hist_kwargs = {
            **dict(bins=np.linspace(0, 80, 10)),
            **hist_kwargs,
        }
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 5))

        for ax, color in zip(axs, ("B", "R", "Z")):
            hist_ret = ax.hist(
                (properties[f"{color}_MASK"] != 0).sum(axis=1),
                **hist_kwargs,
            )

            if save_data:
                data_dict[f"{color}_hist"] = hist_ret[:2]

            ax.set_title(f"{color}-band")

        fig.supxlabel("# masked pixels per QSO")
        if downsampling < 1:
            fig.suptitle(
                f"Histogram: Histogram of QSOs per number of pixel masked (subsample {round(downsampling*100, 2)}%)"
            )
        else:
            fig.suptitle(
                "Histogram: Histogram of QSOs per number of pixel masked")

        axs[0].set_ylabel("# QSOs")
        plt.tight_layout()

        if save_plot:
            plt.savefig(
                output_prefix.parent
                / (output_prefix.name + "-qso_histogram_per_masked_bin.png"),
                dpi=300,
            )

        if save_data:
            np.savez(
                output_prefix.parent
                / (output_prefix.name + "-qso_hsitogram_per_masked_bin.npz"),
                **{**save_dict, **data_dict},
            )


class ReadDESILyaData:
    """
    Class for reading DESI Lyman-alpha QSO spectra and associated mask data.

    This class handles:
        - Parsing a DESI QSO catalogue and grouping targets by HEALPIX and survey.
        - Identifying the relevant coadd files in the DESI directory structure.
        - Reading flux, inverse variance (ivar), mask, and wavelength data for
          each color band (B, R, Z).
        - Optional downsampling of targets and/or input files for performance.

    Typical usage involves instantiating this class with a DESI directory and
    catalogue, then calling `read_data()` to retrieve the data needed for
    downstream plotting or analysis.

    Attributes:
    ----------
        input_directory (Path): Base directory containing coadd files.
        catalogue (Table): Astropy Table loaded from the input QSO catalogue.
    """

    def __init__(self, input_directory: Path, catalogue: Path | str):
        """
        Read flux, ivar, mask and wavelength information for DESI data

        Arguments:
        ----------
            input_directory (Path): Input directory.
            catalogue (Path): QSO catalog.
        """
        self.input_directory = Path(input_directory)
        if not input_directory.is_dir():
            raise FileNotFoundError(
                f"Invalid input directory: {self.input_directory}")
        if not Path(catalogue).is_file():
            raise FileNotFoundError(f"Invalid catalogue file: {catalogue}")

        self.catalogue = Table(
            fitsio.read(
                catalogue,
                ext="ZCATALOG",
            )
        )

    def read_data(
        self,
        downsampling: float = 1,
        file_downsampling: float = 1,
        processes: Optional[int] = None,
    ) -> Dict:
        """
        Read data from DESI files.

        Read flux, ivar, mask, and wavelength data from DESI coadd files.
        The function optionally downsamples the QSO catalogue and/or number of
        coadd files to be read. It parallelizes file reading using Python's
        multiprocessing.

        Arguments:
        ----------
            downsampling (float): Fraction of QSOs to randomly sample from the
                                  catalogue (0 < p <= 1). (Default: 1).
            file_downsampling (float): Fraction of coadd files to use after grouping
                                  (0 < p <= 1). (Default: 1).
            processes (int, optional): Number of parallel processes to use;
                                  defaults to all cores. (Default: None).

        Returns:
        ----------
            Dict: Dictionary containing arrays of:
                - B/R/Z_WAVELENGTH
                - B/R/Z_FLUX
                - B/R/Z_IVAR
                - B/R/Z_MASK
                - TARGETID (flattened list of matched target IDs)
        """
        if downsampling < 1:
            logger.info(f"Downsampling QSO catalogue. p={downsampling}")
            np.random.seed(1)
            msk = np.random.choice(
                [True, False],
                len(self.catalogue),
                p=[downsampling, 1 - downsampling],
            )

            catalogue = self.catalogue[msk]
        else:
            catalogue = self.catalogue

        healpix = [
            hp.ang2pix(
                64,
                row["TARGET_RA"],
                row["TARGET_DEC"],
                lonlat=True,
                nest=True,
            )
            for row in catalogue
        ]

        catalogue["HEALPIX"] = healpix
        logger.info("Sorting catalog by HEALPIX")
        catalogue.sort("HEALPIX")

        logger.info("Grouping catalogue by HEALPIX, SURVEY")
        grouped_catalogue = catalogue.group_by(["HEALPIX", "SURVEY"])

        logger.info("Retreiving filenames")

        files_to_read = []
        group_valid_targets = []
        for group in grouped_catalogue.groups:
            healpix, survey = group["HEALPIX", "SURVEY"][0]

            filename = (
                self.input_directory
                / survey
                / "dark"
                / str(healpix // 100)  # type: ignore
                / str(healpix)  # type: ignore
                / f"coadd-{survey}-dark-{healpix}.fits"
            )

            if filename.is_file():
                files_to_read.append(filename)
                group_valid_targets.append(group["TARGETID"])

        logger.info(f"Number of files to read: {len(files_to_read)}.")
        if file_downsampling < 1:
            logger.info(f"Downsampling files to read.")
            np.random.seed(2)
            msk = np.random.choice(
                [True, False],
                len(files_to_read),
                p=[file_downsampling, 1 - file_downsampling],
            )

            files_to_read = np.asarray(files_to_read)[msk]
            group_valid_targets = np.asarray(group_valid_targets)[msk]

        logger.info(f"Reading files.")
        pool = Pool(processes=processes)

        results = pool.starmap(
            read_file,
            zip(
                files_to_read,
                group_valid_targets,
            ),
        )

        values = np.asarray(results)

        properties = dict(
            B_WAVELENGTH=values[0][0],
            R_WAVELENGTH=values[0][4],
            Z_WAVELENGTH=values[0][8],
            B_FLUX=np.vstack(values.T[1]),
            R_FLUX=np.vstack(values.T[5]),
            Z_FLUX=np.vstack(values.T[9]),
            B_IVAR=np.vstack(values.T[2]),
            R_IVAR=np.vstack(values.T[6]),
            Z_IVAR=np.vstack(values.T[10]),
            B_MASK=np.vstack(values.T[3]),
            R_MASK=np.vstack(values.T[7]),
            Z_MASK=np.vstack(values.T[11]),
            TARGETID=np.concatenate(values.T[12]),
        )

        return properties


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main entry point for running the DESI mask statistics analysis.

    Parses arguments, reads DESI QSO spectra and mask data, and generates
    plots summarizing masking behavior across the Lyman-alpha sample.

    Arguments:
    ----------
        args (argparse.Namespace, optional): Parsed CLI arguments;
                                             if None, parsed via `getArgs()`.
    """
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    if not args.output_prefix.parent.is_dir():
        raise ValueError(f"Invalid output-prefix: {args.output_prefix}")

    data_reader = ReadDESILyaData(
        args.input_directory,
        args.catalogue,
    )

    properties = data_reader.read_data(
        args.downsampling,
        processes=args.num_proc,
    )

    Plots.histogram_qsos_per_masked_bins(
        properties,
        args.output_prefix,
        downsampling=args.downsampling,
        save_data=args.save_data,
        save_plot=True,
        save_dict=vars(args),
    )
    Plots.masked_pixels_percentage(
        properties,
        args.output_prefix,
        downsampling=args.downsampling,
        save_data=args.save_data,
        save_plot=True,
        save_dict=vars(args),
    )


def getArgs() -> argparse.Namespace:
    """
    Parse command-line arguments for DESI mask analysis.

    Returns:
    -----------
        argparse.Namespace: Parsed arguments including input paths,
        downsampling flags, number of processes, and output options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory",
        type=Path,
        required=True,
        help="Input directory",
    )

    parser.add_argument(
        "--catalogue",
        type=Path,
        required=True,
        help="QSO catalog",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes to use for the multiprocessing tool.",
    )

    parser.add_argument(
        "--downsampling",
        type=float,
        default=1,
        help="Downsampling to apply to input data.",
    )

    parser.add_argument(
        "--save-data", action="store_true", help="Store data to be plotted in py"
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Prefix of the path where to save the plots to.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
