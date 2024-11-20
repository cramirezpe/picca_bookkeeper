"""Script to run all the analyis from terminal"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.scripts.run_cf import main as run_cf
from picca_bookkeeper.scripts.run_delta_extraction import main as run_delta_extraction
from picca_bookkeeper.scripts.run_fit import main as run_fit
from picca_bookkeeper.scripts.run_sampler import main as run_sampler
from picca_bookkeeper.scripts.run_xcf import main as run_xcf

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

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )

    config = DictUtils.merge_dicts(
        bookkeeper.defaults,
        bookkeeper.config,
    )

    raw = config["general"].get("raw mocks", False)
    true = config["general"].get("true mocks", False)

    ########################################
    ## Identifying needed runs from names
    ########################################
    regions = []

    autos = []
    auto_correlations = args.auto_correlations
    if config.get("fits", dict()).get("auto correlations", None) not in (None, ""):
        auto_correlations += config["fits"]["auto correlations"].split(" ")
    for auto in auto_correlations:
        absorber, region, absorber2, region2 = auto.replace("-", ".").split(".")

        region = bookkeeper.validate_region(region)
        absorber = bookkeeper.validate_absorber(absorber)
        region2 = bookkeeper.validate_region(region2)
        absorber2 = bookkeeper.validate_absorber(absorber2)

        autos.append([absorber, region, absorber2, region2])
        regions.append(region)
        regions.append(region2)

    crosses = []
    cross_correlations = args.cross_correlations
    if config.get("fits", dict()).get("cross correlations", None) not in (None, ""):
        cross_correlations += config["fits"]["cross correlations"].split(" ")
    for cross in cross_correlations:
        absorber, region = cross.split(".")
        region = bookkeeper.validate_region(region)
        absorber = bookkeeper.validate_absorber(absorber)

        regions.append(region)
        crosses.append([absorber, region])

    regions = np.unique(regions)

    ########################################
    ## Running delta extraction for calibration
    ## and then all the deltas needed.
    ########################################
    if not args.no_deltas:
        if (not raw and not true) and config["delta extraction"]["calib"] != 0:
            calib_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                region="lya",  # It doesn't really matter
                overwrite_config=False,
                system=args.system,
                debug=args.debug,
                only_calibration=True,  # Because we are only running calib
                skip_calibration=False,
                only_write=args.only_write,
                wait_for=args.wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            run_delta_extraction(calib_args)

        for region in regions:
            region_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                region=region,
                overwrite_config=False,
                debug=args.debug,
                system=args.system,
                only_calibration=False,
                skip_calibration=True,
                only_write=args.only_write,
                wait_for=args.wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            run_delta_extraction(region_args)

    ########################################
    ## Running all the correlations needed
    ########################################
    if not args.no_correlations:
        for auto in autos:
            absorber, region, absorber2, region2 = auto

            auto_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                overwrite_config=False,
                region=region,
                region2=region2,
                absorber=absorber,
                absorber2=absorber2,
                system=args.system,
                debug=False,  # Debug, only set deltas
                only_write=args.only_write,
                wait_for=args.wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            run_cf(auto_args)

        for cross in crosses:
            absorber, region = cross

            cross_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                overwrite_config=False,
                region=region,
                absorber=absorber,
                system=args.system,
                debug=False,  # Debug, only set deltas,
                only_write=args.only_write,
                wait_for=args.wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            run_xcf(cross_args)

    ########################################
    ## Running fits
    ########################################
    if not args.no_fits:
        fit_args = argparse.Namespace(
            bookkeeper_config=args.bookkeeper_config,
            overwrite_config=False,
            system=args.system,
            only_write=args.only_write,
            wait_for=args.wait_for,
            log_level=args.log_level,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        run_fit(fit_args)

    if args.sampler:
        sampler_args = argparse.Namespace(
            bookkeeper_config=args.bookkeeper_config,
            system=args.system,
            overwrite_config=False,
            only_write=args.only_write,
            wait_for=args.wait_for,
            log_level=args.log_level,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        run_sampler(sampler_args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--system",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite output data."
    )

    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config.",
    )

    parser.add_argument(
        "--skip-sent", action="store_true", help="Skip runs that were already sent."
    )

    parser.add_argument(
        "--auto-correlations",
        type=str,
        nargs="+",
        default=[],
        help="List of auto-correlations to include in the vega "
        "fits. The format of the strings should be 'lya.lya-lya.lyb'. "
        "which reads as Lyman-alpha absorption in the Lyman-alpha region "
        "correlated with lyman alpha in the lyman beta region. "
        "This is to allow splitting.",
    )

    parser.add_argument(
        "--cross-correlations",
        type=str,
        nargs="+",
        default=[],
        help="List of cross-correlations to include in the vega "
        "fits. The format of the strings should be 'lya.lya'.",
    )

    parser.add_argument(
        "--no-deltas", action="store_true", help="Don't measure deltas."
    )

    parser.add_argument(
        "--no-correlations", action="store_true", help="Don't measure correlations."
    )

    parser.add_argument("--no-fits", action="store_true", help="Don't measure fits.")

    parser.add_argument("--sampler", action="store_true", help="Run the sampler.")

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--only-write", action="store_true", help="Only write scripts, not send them."
    )

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
