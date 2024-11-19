"""Correct zeff in input config files by reading corresponding correlation files"""

from __future__ import annotations

import argparse
import configparser
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import numpy as np
import yaml

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Optional


logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    if not args.ini_input:
        bookkeeper = Bookkeeper(args.bookkeeper_config)
        main_file = bookkeeper.paths.fit_main_fname()
    else:
        main_file = args.bookkeeper_config

    logger.info("Obtaining export files.")
    # Identifying which correlations are affected:
    main_config = configparser.ConfigParser()
    main_config.optionxform = str  # type: ignore
    main_config.read(main_file)

    ini_files = main_config["data sets"].get("ini files").split(" ")

    zeff_list = []
    weights = []
    for ini_file in ini_files:
        config = configparser.ConfigParser()
        config.read(ini_file)

        export_file = config["data"].get("filename")
        rp_min = config["cuts"].getfloat("rp-min", 0)
        rp_max = config["cuts"].getfloat("rp-max", 300)
        rt_min = config["cuts"].getfloat("rt-min", 0)
        rt_max = config["cuts"].getfloat("rt-max", 300)
        rmin = config["cuts"].getfloat("r-min", 0)
        rmax = config["cuts"].getfloat("r-max", 300)

        with fitsio.FITS(export_file) as hdul:
            r = np.sqrt(hdul[1].read()["RP"] ** 2 + hdul[1].read()["RT"] ** 2)

            w = hdul["COR"].read()["RP"] >= rp_min
            w &= hdul["COR"].read()["RP"] <= rp_max
            w &= hdul["COR"].read()["RT"] >= rt_min
            w &= hdul["COR"].read()["RT"] <= rt_max
            w &= r >= rmin
            w &= r <= rmax

            co = hdul["COR"]["CO"][:]

            inverse_variance = 1 / np.diag(co)

            zeff = np.average(hdul[1].read()["Z"][w], weights=inverse_variance[w])
            weight = np.sum(inverse_variance[w])

        logger.info(
            f"File: {export_file}\n\trp-min: {rp_min}\n\trp-max:{rp_max}"
            f"\n\trt-min: {rt_min}\n\trt-max: {rt_max}"
            f"\n\tr-min: {rmin}\n\tr-max: {rmax}"
            f"\n\tzeff: {zeff}\n\tweight: {weight}\n"
        )
        zeff_list.append(zeff)
        weights.append(weight)

    zeff = np.average(zeff_list, weights=weights)

    logger.info(f"zeff: {zeff}")

    computed_parameters_config = {
        "fits": {
            "extra args": {
                "vega_main": {
                    "general": {
                        "data sets": {
                            "zeff": float(zeff),
                        },
                        "parameters": {},
                    }
                }
            }
        }
    }

    if args.apply_edmond_qso_bias:
        qso_bias = 0.214 * ((1 + zeff) ** 2 - 6.565) + 2.206

        logger.info(f"bias_QSO: {qso_bias}")

        computed_parameters_config["fits"]["extra args"]["vega_main"]["general"]["parameters"][  # type: ignore
            "bias_QSO"
        ] = float(
            qso_bias
        )  # type: ignore

    # convert main into dict, add changes, write main into main.ini
    main_dict = {s: dict(main_config.items(s)) for s in main_config.sections()}
    main_dict["data sets"]["zeff"] = zeff
    if args.apply_edmond_qso_bias:
        main_dict["parameters"]["bias_QSO"] = qso_bias
    Bookkeeper.write_ini(main_dict, main_file)

    # Now I save computed_parameters so next time I run bookkeeper this step can be
    # skipped. But only if bookkeeper available:
    if not args.ini_input:
        with open(bookkeeper.paths.fit_computed_params_out(), "w") as f:
            yaml.safe_dump(computed_parameters_config, f, sort_keys=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use."
    )

    parser.add_argument(
        "--ini-input",
        action="store_true",
        help="Use main.ini file as input instead of bookkeeper config.",
    )

    parser.add_argument(
        "--apply-edmond-qso-bias",
        action="store_true",
        help="Apply Edmond derived quasar bias evolution to set bias_QSO.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
