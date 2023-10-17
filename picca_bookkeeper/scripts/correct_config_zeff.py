"""Correct zeff in input config files by reading corresponding correlation files"""
from __future__ import annotations

import argparse
import configparser
import logging
import re
import sys
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

import fitsio
import numpy as np

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.utils import compute_zeff

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
    main_config.read(main_file)

    ini_files = main_config["data sets"].get("ini files").split(' ')

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

            inverse_variance = 1/ np.diag(co)

            zeff = np.average(
                hdul[1].read()["Z"][w], weights=inverse_variance[w]
            )
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

    (Path(main_file).parent / ".zeff.status").write_text("COMPLETED")
            

    # Apply change to main.ini
    pattern = re.compile(r'zeff.*$', re.MULTILINE)
    main_file.write_text(
        pattern.sub(f'zeff = {zeff}', main_file.read_text())
    )

    
    if not args.ini_input:
        # Apply change to input bookkeeper_config.yaml
        if args.bookkeeper_config.is_file():
            args.bookkeeper_config.write_text(
                pattern.sub(f'zeff: {zeff}', args.bookkeeper_config.read_text())
            )
        else:
            logger.warn("Input bookkeeper config file moved.")

        # And also to stored bookkeeper
        bookkeeper.paths.fit_config_file.write_text(
            pattern.sub(f'zeff: {zeff}', bookkeeper.paths.fit_config_file.read_text())
        )
        
    if args.apply_edmond_qso_bias:
        qso_bias = 0.214 * ( (1 + zeff)**2 - 6.565) + 2.206

        logger.info(f"bias_QSO: {qso_bias}")


        # get value
        qso_bias_ini = main_config["parameters"].get("bias_QSO", None)
        
        if qso_bias_ini is None:
            raise ValueError("bias_QSO not defined in ini.")


        if not args.ini_input:
            # Apply change to input bookkeeper_config.yaml
            pattern = re.compile(rf'bias_QSO\s*:\s*{re.escape(qso_bias_ini)}.*', re.MULTILINE)

            # if not pattern.search(bookkeeper.paths.fit_config_file.read_text()):
            #     raise ValueError("bias_QSO not defined in bookkeeper config.")

            if args.bookkeeper_config.is_file():
                args.bookkeeper_config.write_text(
                    pattern.sub(f"bias_QSO: {qso_bias}", args.bookkeeper_config.read_text())
                )
            else:
                logger.warn("Input bookkeeper config file moved.")

            # And also to stored bookkeeper
            bookkeeper.paths.fit_config_file.write_text(
                pattern.sub(f"bias_QSO: {qso_bias}", bookkeeper.paths.fit_config_file.read_text())
            )


        ## Apply change to main.ini
        pattern = re.compile(rf'bias_QSO\s*=\s*{re.escape(qso_bias_ini)}.*', re.MULTILINE)

        main_file.write_text(
            pattern.sub(f'bias_QSO = {qso_bias}', main_file.read_text())
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bookkeeper_config", type=Path, help="Path to bookkeeper file to use.")

    parser.add_argument("--ini-input", action="store_true", help="Use main.ini file as input instead of bookkeeper config.")

    parser.add_argument("--apply-edmond-qso-bias", action="store_true", help="Apply Edmond derived quasar bias evolution to set bias_QSO.")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
