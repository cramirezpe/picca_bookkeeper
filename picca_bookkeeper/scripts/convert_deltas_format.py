"""
Convert delta FITS files to a new, more efficient format for fast data access.

This script reads "delta" FITS files (typically named delta*fits*) produced by
the picca pipeline, and rewrites them to a new format that is optimized for
faster reading and downstream processing.

Functionality:
--------------
    - Reads all delta FITS files in a specified input directory.
    - For each file, rebins the data onto a common wavelength grid
      (linear or logarithmic, as defined by the config file).
    - Aggregates and stores spectra, weights, continua, and relevant metadata
      into new FITS extensions.
    - Optionally, if survey data is provided, computes and stores flux and flux
      variance arrays aligned with the delta files.
    - Metadata such as RA, DEC, redshift, line-of-sight ID, SNR, target ID,
      observation night, petal, and tile are preserved and output as a
      METADATA extension.
    - Handles both blinded and unblinded delta data.

Dependencies:
--------------
    - Internal modules: picca_bookkeeper.utils (find_bins, wavelength grid mapping),
                        picca.delta_extraction.survey (Survey class, data handeling)

Usage:
------
Run the script from the command line:
    python convert_deltas_format.py --in-dir <input_delta_dir> --out-dir <output_dir>
    --config-file <delta_config_file> --nproc <num_processes> [--config-for-flux <survey_config_file>]

Arguments:
----------
    --in-dir           Directory containing input delta FITS files.
    --out-dir          Directory to write reformatted delta FITS files.
    --config-file      Path to a picca delta extraction config file (ini format).
    --nproc            Number of parallel processes to use.
    --log-level        Logging level (default: INFO).
    --config-for-flux  (Optional) Path to a config file for extracting flux properties.

Example:
--------
    python convert_deltas_format.py \
        --in-dir /data/picca/deltas \
        --out-dir /data/picca/deltas_converted \
        --config-file /data/picca/config.ini \
        --nproc 4

Notes:
------
    - The script is parallelized using multiprocessing.Pool for speed.
    - All output FITS files are overwritten if they already exist in the
      output directory.
    - Logging is provided for progress and error reporting.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from configparser import ConfigParser
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import numpy as np

from picca_bookkeeper.utils import find_bins

if TYPE_CHECKING:
    from typing import Dict, Optional, Tuple
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main entry point for the delta conversion script.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]): Parsed command-line arguments.
            If None, arguments are parsed from the command line using `getArgs()`.

    Behavior:
    ----------
        - Initializes logging based on user-specified log level.
        - Creates the output directory if it does not exist.
        - Optionally loads survey data if a flux config is provided.
        - Loads delta extraction configuration from file.
        - Uses multiprocessing to convert all delta files in the input directory.

    Raises:
    ----------
        FileNotFoundError: If any required file paths do not exist.
        ValueError: If the configuration file is invalid or incomplete.
    """
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)

    if args.config_for_flux is not None:
        survey_data = read_survey_data(str(args.config_for_flux))
    else:
        survey_data = None

    config = ConfigParser()
    config.read(args.config_file)

    pool = Pool(processes=args.nproc)
    results = pool.starmap(
        convert_delta_file,
        zip(
            args.in_dir.glob("delta*fits*"),
            itertools.repeat(args.out_dir),
            itertools.repeat(config),
            itertools.repeat(survey_data),
        ),
    )
    pool.close()


def convert_delta_file(
    delta_path: Path,
    out_dir: Path,
    config: Dict,
    survey_data: Optional[Tuple[np.ndarray, ...]] = None,
) -> None:
    """
    Reads a single delta FITS file and converts it to the new format.

    Arguments:
    ----------
        - delta_path (Path): Path to the input delta FITS file.
        - out_dir (Path): Directory to write the reformatted FITS file.
        - config (Dict): Configuration dictionary parsed from the ini file.
        - survey_data (Optional[Tuple[np.ndarray, ...]]): Tuple of preprocessed
            survey arrays: (LOS_IDs, fluxes, inverse variances, wavelengths).
            Used to align flux data with delta entries if provided.

    Behavior:
    ---------
        - Rebins delta, weight, and continuum data onto a common wavelength grid.
        - Preserves metadata such as RA, DEC, Z, LOS_ID, MEANSNR, etc.
        - Detects whether the file is blinded and handles accordingly.
        - Optionally inserts FLUX and FLUX_IVAR extensions from matched survey data.
        - Writes all processed arrays to a new FITS file in the output directory.

    Raises:
    -------
        - KeyError: If required FITS header keywords are missing.
        - ValueError: If the input file format or contents are malformed.
        - IOError: If the output file cannot be written.
    """
    logger.info(f"Reading file: {delta_path.name}")
    if config["data"]["wave solution"] == "lin":
        lambda_grid = np.arange(
            float(config["data"]["lambda min"]),
            float(config["data"]["lambda max"]) +
            float(config["data"]["delta lambda"]),
            float(config["data"]["delta lambda"]),
        )
    else:
        lambda_grid = np.arange(
            np.log10(float(config["data"]["lambda min"])),
            np.log10(float(config["data"]["lambda max"]))
            + float(config["data"]["delta log lambda"]),
            float(config["data"]["delta log lambda"]),
        )
        lambda_grid = 10**lambda_grid

    original_delta = fitsio.FITS(delta_path)

    num_forests = len(original_delta) - 1
    num_lambda_bins = len(lambda_grid)

    delta = np.zeros((num_forests, num_lambda_bins), dtype=float)
    weight = np.zeros((num_forests, num_lambda_bins), dtype=float)
    cont = np.zeros((num_forests, num_lambda_bins), dtype=float)
    ra = np.zeros(num_forests, dtype=float)
    dec = np.zeros(num_forests, dtype=float)
    los_id = np.zeros(num_forests, dtype=int)
    z = np.zeros(num_forests, dtype=float)
    meansnr = np.zeros(num_forests, dtype=float)
    tid = np.zeros(num_forests, dtype=int)
    night = np.zeros(num_forests, dtype=str)
    petal = np.zeros(num_forests, dtype=str)
    tile = np.zeros(num_forests, dtype=str)

    if "DELTA_BLIND" in original_delta[1].get_colnames():
        blind = True
    else:
        blind = False

    for i, hdu in enumerate(original_delta[1:]):
        bins = find_bins(
            hdu["LAMBDA"][:],
            lambda_grid,
        )

        if blind:
            delta[i][bins] = hdu["DELTA_BLIND"][:]
        else:
            delta[i][bins] = hdu["DELTA"][:]

        weight[i][bins] = hdu["WEIGHT"][:]
        cont[i][bins] = hdu["CONT"][:]

        header = hdu.read_header()

        ra[i] = header["RA"]
        dec[i] = header["DEC"]
        z[i] = header["Z"]
        los_id[i] = header["LOS_ID"]
        meansnr[i] = header["MEANSNR"]
        tid[i] = header["TARGETID"]
        night[i] = header["NIGHT"]
        petal[i] = header["PETAL"]
        tile[i] = header["TILE"]

    if survey_data is not None:
        raw_ids, raw_flux, raw_ivar, raw_lambda = survey_data
        # Obtain flux information, this only happens if survey
        # data is provided.
        flux = np.full_like(delta, np.nan)
        flux_ivar = np.full_like(delta, np.nan)

        for i, id_ in enumerate(los_id):
            indx = np.searchsorted(raw_ids, id_)

            if indx:
                bins = find_bins(
                    raw_lambda[indx],
                    lambda_grid,
                )

                flux[i][bins] = raw_flux[indx]
                flux_ivar[i][bins] = raw_ivar[indx]
            else:
                print("Forest not found in survey data: ", id_)

    new_header = {
        "WAVE_SOLUTION": header["WAVE_SOLUTION"],
        "DELTA_LAMBDA": header["DELTA_LAMBDA"],
    }
    if blind:
        new_header["BLINDING"] = header["BLINDING"]

    with fitsio.FITS(out_dir / delta_path.name, "rw", clobber=True) as results:
        logger.info(f"Writting file: {delta_path.name}")
        results.write(
            [ra, dec, z, los_id, meansnr, tid, night, petal, tile],
            names=[
                "RA",
                "DEC",
                "Z",
                "LOS_ID",
                "MEANSNR",
                "TARGETID",
                "NIGHT",
                "PETAL",
                "TILE",
            ],
            header=new_header,
            extname="METADATA",
        )

        results.write(
            lambda_grid,
            extname="LAMBDA",
        )

        results.write(
            delta,
            extname="DELTA_BLIND" if blind else "DELTA",
        )

        results.write(
            weight,
            extname="WEIGHT",
        )

        results.write(
            cont,
            extname="CONT",
        )

        if survey_data is not None:
            results.write(flux, extname="FLUX")

            results.write(
                flux_ivar,
                extname="FLUX_IVAR",
            )


def read_survey_data(config_file: Path | str) -> Tuple[np.ndarray, ...]:
    """
    Reads and processes survey data used for aligning flux arrays with delta files.

    Arguments:
    ----------
        config_file (Path | str): Path to a valid survey configuration file
            compatible with picca.delta_extraction.survey.

    Returns:
    ----------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Sorted arrays:
            - LOS_IDs (int64): Line-of-sight IDs.
            - fluxes (float64): Flux values per forest.
            - inverse variances (float64): Inverse variance of flux values.
            - wavelengths (float64): Wavelengths corresponding to each flux entry.

    Raises:
    -------
        - FileNotFoundError: If the config file is missing.
        - RuntimeError: If survey corrections or data cannot be processed.
    """
    from picca.delta_extraction.survey import Survey

    survey = Survey()
    survey.load_config(str(config_file))
    survey.read_corrections()
    survey.read_masks()
    survey.read_data()
    survey.apply_corrections()
    survey.apply_masks()
    survey.filter_forests()

    ids = np.asarray([forest.los_id for forest in survey.data.forests])
    fluxes = np.asarray([forest.flux for forest in survey.data.forests])
    ivars = np.asarray([forest.ivar for forest in survey.data.forests])
    lambdas = np.asarray(
        [10**forest.log_lambda for forest in survey.data.forests])

    sortinds = ids.argsort()
    return ids[sortinds], fluxes[sortinds], ivars[sortinds], lambdas[sortinds]


def getArgs() -> argparse.Namespace:
    """
    Parses command-line arguments for the delta conversion script.

    Returns:
    --------
        argparse.Namespace: Parsed arguments with the following attributes:
            - in_dir (Path): Directory containing input delta FITS files.
            - out_dir (Path): Output directory for converted delta files.
            - config_file (Path): Path to a delta config file (INI format).
            - config_for_flux (Optional[Path]): Optional path to a config for
              survey data.
            - nproc (int): Number of processes to use for parallel conversion.
            - log_level (str): Logging level (default: INFO).

    Raises:
    -------
        argparse.ArgumentError: If required arguments are missing or invalid.

    Example:
    --------
        python convert_deltas_format.py \
            --in-dir ./deltas --out-dir ./converted \
            --config-file delta_config.ini --nproc 4
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    parser.add_argument(
        "--in-dir",
        type=Path,
        help="Input directory to read deltas from.",
    )

    parser.add_argument(
        "--out-dir", type=Path, help="Output directory to save deltas to."
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        help="picca delta extraction config file (could be output config file .config.ini).",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        help="Number of processes to use.",
    )

    parser.add_argument(
        "--config-for-flux",
        type=Path,
        help="Compute flux properties as flux and flux variance and add them to delta files. Should use an input picca configuration.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
