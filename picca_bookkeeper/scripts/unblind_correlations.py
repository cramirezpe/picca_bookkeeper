""" Script to run add_extra_deltas_data
given a bookkeeper config file.
Originally written by Ignasi Pérez-Ràfols"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.io import fits

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

    logger.warn(
        "THIS SHOULD ONLY BE USED WITH THE PERMISSION"
        "OF KP6 CONVENERS FOR SPECIFIC TESTS"
    )

    if args.cf is not None:
        logger.info(f"Unblinding correlation function")
        unblind_cf(args.cf)
    if args.dmat is not None:
        logger.info("Unblinding distortion matrix")
        unblind_dmat(args.dmat)
    if args.metal_dmat is not None:
        logger.info("Unblinding metal distortion matrix")
        unblind_metal_dmat(args.metal_dmat)
    logger.info("Done")


def unblind_cf(cf_file: Path) -> None:
    assert cf_file.is_file()
    with fits.open(cf_file, mode="update") as hdul:
        # modify ATTRI HDU
        assert "ATTRI" in hdul, "missing HDU ATTRI"
        hdul["ATTRI"].header["BLINDING"] = "none"
        hdul["ATTRI"].header["HISTORY"] = "unblinded using force_unblinding.py"

        # modify COR HDU
        assert "ATTRI" in hdul, "missing HDU COR"
        for index in range(1, hdul["COR"].header["TFIELDS"] + 1):
            if hdul["COR"].header[f"TTYPE{index}"] == "DA_BLIND":
                hdul["COR"].header[f"TTYPE{index}"] = "DA"
        hdul["COR"].header["HISTORY"] = "unblinded using force_unblinding.py"


def unblind_dmat(dmat_file: Path) -> None:
    assert dmat_file.is_file()
    with fits.open(dmat_file, mode="update") as hdul:
        # modify DMAT HDU
        assert "DMAT" in hdul, "missing HDU DMAT"
        hdul["DMAT"].header["BLINDING"] = "none"
        for index in range(1, hdul["DMAT"].header["TFIELDS"] + 1):
            if hdul["DMAT"].header[f"TTYPE{index}"] == "DM_BLIND":
                hdul["DMAT"].header[f"TTYPE{index}"] = "DM"
        hdul["DMAT"].header["HISTORY"] = "unblinded using force_unblinding.py"


def unblind_metal_dmat(metal_dmat_file: Path) -> None:
    assert metal_dmat_file.is_file()
    with fits.open(metal_dmat_file, mode="update") as hdul:
        # modify ATTRI HDU
        assert "ATTRI" in hdul, "missing HDU ATTRI"
        hdul["ATTRI"].header["BLINDING"] = "none"
        hdul["ATTRI"].header["HISTORY"] = "unblinded using force_unblinding.py"

        # modify MDMAT HDU
        assert "MDMAT" in hdul, "missing HDU ATTRI"
        for index in range(1, hdul["MDMAT"].header["TFIELDS"] + 1):
            if "DM_BLIND" in hdul["MDMAT"].header[f"TTYPE{index}"]:
                name = hdul["MDMAT"].header[f"TTYPE{index}"]
                mod_name = name.replace("DM_BLIND", "DM")
                hdul["MDMAT"].header[f"TTYPE{index}"] = mod_name
        hdul["MDMAT"].header["HISTORY"] = "unblinded using force_unblinding.py"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf", type=Path, default=None, help="Correlation function file"
    )
    parser.add_argument(
        "--dmat", type=Path, default=None, help="Distortion matrix file"
    )
    parser.add_argument(
        "--metal-dmat", type=Path, default=None, help="Metal distortion matrix file"
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
