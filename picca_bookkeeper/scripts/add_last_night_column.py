"""
    Add column LAST_NIGHT with a default time value to a given catalog.
"""
import argparse
from pathlib import Path
import logging
import sys
import fitsio
import numpy as np

logger = logging.getLogger(__name__)


def main(args=None):
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
            [args.fill_value for i in range(hdul["ZCATALOG"].read_header()["NAXIS2"])]
        )
        hdul["ZCATALOG"].insert_column(name="LAST_NIGHT", data=data)

    logger.info("Done.")


def getArgs():
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
