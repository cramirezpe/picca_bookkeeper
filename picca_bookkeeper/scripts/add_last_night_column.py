"""
add_last_night_column.py

Adds a 'LAST_NIGHT' column to the 'ZCATALOG' HDU of a FITS catalog file,
filled with a specified (default: YYYYMMDD) date value.

Functionality:
--------------
    - Opens the input FITS catalog in read-write mode using fitsio.
    - Determines the number of rows in the 'ZCATALOG' HDU.
    - Inserts a new column 'LAST_NIGHT' with the provided fill value
      (as an integer date, e.g., 20210403) for every row.
    - Provides configurable logging output.

Usage:
------
    python add_last_night_column.py <input_cat> [--fill-value YYYYMMDD] [--log-level LEVEL]

    Arguments:
    ----------
        --input_cat: Path to the FITS file to be modified.
        --fill-value: Date value to fill in the new column (default: 20210403).
        --log-level: Logging verbosity (default: INFO; options: CRITICAL,
                                        ERROR, WARNING, INFO, DEBUG, NOTSET).

Interactions:
-------------
    - This script is intended to be run as a standalone utility within the
      'scripts' directory.
    - It modifies FITS catalog files (as used elsewhere in picca_bookkeeper,
      e.g., for catalog management and processing).
    - The added 'LAST_NIGHT' column may be used by downstream analysis scripts
      or pipeline steps that require catalog date tracking.
    - Assumes the input FITS file contains a 'ZCATALOG' HDU.

Example:
--------
    python add_last_night_column.py mycatalog.fits --fill-value 20230501 --log-level DEBUG

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import numpy as np

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Routine for adding the 'LAST_NIGHT' column to the input FITS catalog.

    Opens the input file in read-write mode and inserts a new column named
    'LAST_NIGHT' into the 'ZCATALOG' HDU, filled with the specified integer date.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]): Parsed command-line arguments.
            If None, arguments are parsed using getArgs().
    """
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info(f"Using fill value: {args.fill_value}")

    with fitsio.FITS(args.input_cat, "rw") as hdul:
        data = np.asarray(
            [args.fill_value for i in range(
                hdul["ZCATALOG"].read_header()["NAXIS2"])]
        )
        hdul["ZCATALOG"].insert_column(name="LAST_NIGHT", data=data)

    logger.info("Done.")


def getArgs() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
    --------
        argparse.Namespace: Namespace with parsed arguments:
            - input_cat (Path): Path to the input FITS catalog.
            - fill_value (int): Integer date (YYYYMMDD) to fill in the new column.
            - log_level (str): Logging verbosity level.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_cat",
        type=Path,
        help="Input catalog",
    )

    parser.add_argument(
        "--fill-value",
        default="20210403",
        type=int,
        help="Date to use as fill value. Format: YYYYMMDD",
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
