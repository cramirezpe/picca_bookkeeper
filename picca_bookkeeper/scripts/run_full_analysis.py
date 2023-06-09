"""Script to run all the analyis from terminal"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from picca_bookkeeper import bookkeeper
from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.scripts.run_cf import main as run_cf
from picca_bookkeeper.scripts.run_delta_extraction import main as run_delta_extraction
from picca_bookkeeper.scripts.run_fit import main as run_fit
from picca_bookkeeper.scripts.run_xcf import main as run_xcf

logger = logging.getLogger(__name__)


def main(args=None):
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )
    continuum_type = bookkeeper.config["delta extraction"]["prefix"]

    ########################################
    ## Identifying needed runs from names
    ########################################
    regions = []

    autos = []
    auto_correlations = args.auto_correlations
    if bookkeeper.config.get("fits", dict()).get("auto correlations", None) is not None:
        auto_correlations += bookkeeper.config["fits"]["auto correlations"].split(" ")
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
    if bookkeeper.config.get("fits", dict()).get("cross correlations", None) is not None:
        cross_correlations += bookkeeper.config["fits"]["cross correlations"].split(" ")
    for cross in cross_correlations:
        absorber, region = cross.split(".")
        region = bookkeeper.validate_region(region)
        absorber = bookkeeper.validate_absorber(absorber)

        regions.append(region)
        crosses.append([absorber, region])

    regions = np.unique(regions)

    wait_for = list(np.array(args.wait_for).flatten())
    ########################################
    ## Running delta extraction for calibration
    ## and then all the deltas needed.
    ########################################
    if not args.no_deltas:
        if (str(continuum_type) not in ("raw", "True")) and bookkeeper.config[
            "delta extraction"
        ]["calib"] != 0:
            calib_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                region="lya",  # It doesn't really matter
                overwrite_config=False,
                debug=args.debug,
                only_calibration=True,  # Because we are only running calib
                skip_calibration=False,
                only_write=args.only_write,
                wait_for=args.wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            wait_for.append(run_delta_extraction(calib_args))

        region_jobids = dict()
        for region in regions:
            region_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                region=region,
                overwrite_config=False,
                debug=args.debug,
                only_calibration=False,
                skip_calibration=True,
                only_write=args.only_write,
                wait_for=wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            region_jobids[region] = run_delta_extraction(region_args)
    else:
        # If deltas are not computed, wait_for should be generated resembling
        # the output from this step
        region_jobids = dict()
        for region in regions:
            region_jobids[region] = args.wait_for

    ########################################
    ## Running all the correlations needed
    ########################################
    if not args.no_correlations:
        correlation_jobids = []
        for auto in autos:
            absorber, region, absorber2, region2 = auto

            wait_for = [region_jobids[region] for region in (region, region2)]

            auto_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                overwrite_config=False,
                region=region,
                region2=region2,
                absorber=absorber,
                absorber2=absorber2,
                no_dmat=args.no_dmat,
                no_metal=args.no_metal,
                debug=False,  # Debug, only set deltas
                only_write=args.only_write,
                wait_for=wait_for,
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            correlation_jobids.append(run_cf(auto_args))

        for cross in crosses:
            absorber, region = cross

            cross_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                overwrite_config=False,
                region=region,
                absorber=absorber,
                no_dmat=args.no_dmat,
                no_metal=args.no_metal,
                debug=False,  # Debug, only set deltas,
                only_write=args.only_write,
                wait_for=region_jobids[region],
                log_level=args.log_level,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )

            correlation_jobids.append(run_xcf(cross_args))
    else:
        # Again, if correlations are not computed, we should
        # create a wait_for array of the same structure
        # as the one that would have been created.
        correlation_jobids = [region_jobids[region] for region in regions]

    ########################################
    ## Running fits
    ########################################
    if not args.no_fits:
        fit_args = argparse.Namespace(
            bookkeeper_config=args.bookkeeper_config,
            overwrite_config=False,
            auto_correlations=args.auto_correlations,
            cross_correlations=args.cross_correlations,
            only_write=args.only_write,
            wait_for=correlation_jobids,
            log_level=args.log_level,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        fit_jobid = run_fit(fit_args)
    else:
        fit_jobid = correlation_jobids

    if not args.only_write:
        return fit_jobid
    else:
        return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
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

    parser.add_argument(
        "--no-dmat", action="store_true", help="Do not use distortion matrix."
    )

    parser.add_argument(
        "--no-metal", action="store_true", help="Do not compute metal distortion matrix"
    )

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
