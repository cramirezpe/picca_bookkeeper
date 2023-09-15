"""
    Build new deltas format from original deltas. This new format will allow for faster reading of data.
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
    logger.info(f"Reading file: {delta_path.name}")
    if config["data"]["wave solution"] == "lin":
        lambda_grid = np.arange(
            float(config["data"]["lambda min"]),
            float(config["data"]["lambda max"]) + float(config["data"]["delta lambda"]),
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
    lambdas = np.asarray([10**forest.log_lambda for forest in survey.data.forests])

    sortinds = ids.argsort()
    return ids[sortinds], fluxes[sortinds], ivars[sortinds], lambdas[sortinds]


def getArgs() -> argparse.Namespace:
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
