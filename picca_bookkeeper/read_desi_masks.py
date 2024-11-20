"""
    Read flux, ivar, mask and wavelength information for DESI data
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
            fig.suptitle("Histogram: Histogram of QSOs per number of pixel masked")

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
            raise FileNotFoundError(f"Invalid input directory: {self.input_directory}")
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

        Arguments:
        ----------
            downsampling (float, optional): Downsampling to apply to QSO catalogue (Default: 1).
            file_downsampling (float, optional): Downsampling to apply to the number of files to read (Default: 1).
            processes (int, optional): Number of processes to use for reading files. (Default: None).
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
