"""
run_full_analysis.py
--------------------

Script to run entire analyis (end-to-end) from terminal.

Comprehensive script for running the full suite of data analysis steps in the
picca_bookkeeper workflow. This script automates delta extraction, correlation
computations (auto and cross), fitting, and optional sampling- handling all
dependencies and order of operations according to a user-provided Bookkeeper
config file.

Functionality:
-------------
    - Reads and merges configuration from a Bookkeeper config YAML file.
    - Determines which regions and correlations (auto and cross) to analyze
      based on both config and command-line arguments.
    - Runs delta extraction for calibration and regions as required.
    - Runs auto- and cross-correlation computations.
    - Runs fitting procedures and, optionally, a sampler.
    - Offers control over which analysis steps to run or skip.

Script Usage:
-------------
Run this script from the command line to perform the full analysis pipeline:
    python run_full_analysis.py /path/to/bookkeeper_config.yaml [options]

Key command-line arguments:
    --system SYSTEM         : Specify the computing system (optional).
    --overwrite             : Force overwrite of output data.
    --overwrite-config      : Force overwrite of bookkeeper config.
    --skip-sent             : Skip jobs that were already sent.
    --auto-correlations ... : List of auto-correlations to include,
                              e.g. 'lya.lya-lya.lyb'.
    --cross-correlations ...: List of cross-correlations to include,
                              e.g. 'lya.lya'.
    --no-deltas             : Skip delta extraction step.
    --no-correlations       : Skip correlation computations.
    --no-fits               : Skip fitting procedure.
    --sampler               : Run the sampler after fits.
    --only-write            : Only write job scripts, do not submit them.
    --wait-for ...          : Wait for specific job IDs before running.
    --log-level LEVEL       : Set logging verbosity (default: INFO).
    --debug                 : Enable debug mode.

Example:
--------
    python run_full_analysis.py myconfig.yaml --auto-correlations lya.lya-lya.lyb
    --no-fits --log-level DEBUG

Interactions:
-------------
    - Instantiates and uses the `Bookkeeper` class to manage config and bookkeeping.
    - Uses `DictUtils.merge_dicts` to combine defaults and user configs.
    - Directly imports and executes the main functions of:
        - `picca_bookkeeper.scripts.run_delta_extraction`
        - `picca_bookkeeper.scripts.run_cf` (auto-correlations)
        - `picca_bookkeeper.scripts.run_xcf` (cross-correlations)
        - `picca_bookkeeper.scripts.run_fit`
        - `picca_bookkeeper.scripts.run_sampler` (optional)
    - Logging is handled consistently and verbosity can be controlled per run.
    - Designed to be the main entry point for running the entire analysis
      in a reproducible and automated way.

"""

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
    """
    Execute the full data analysis pipeline based on a Bookkeeper config file.

    This function orchestrates all major stages of the analysis:
        - Delta extraction (optionally including calibration).
        - Auto- and cross-correlation computations.
        - Fitting and optional sampler.

    The behavior is controlled via a combination of the provided YAML/JSON
    config file and optional command-line arguments. Tasks are executed in the
    appropriate order and may be written to file or submitted to a job system
    (e.g., SLURM).

    Arguments:
    ----------
        args (argparse.Namespace, optional): Parsed command-line arguments.
            If None, arguments are parsed using `get_args()`.

    Raises:
    -------
        - ValueError: If correlation region or absorber names cannot be validated
            via the Bookkeeper.
        - FileNotFoundError: If the config file path is invalid.
        - KeyError: If expected configuration keys are missing or improperly
            structured.

    Example:
    --------
        >>> main(argparse.Namespace(
                bookkeeper_config="myconfig.yaml",
                auto_correlations=["lya.lya-lya.lyb"],
                cross_correlations=["lya.lya"],
                no_deltas=False,
                no_correlations=False,
                no_fits=False,
                sampler=True,
                system="slurm",
                only_write=False,
                wait_for=None,
                overwrite=True,
                overwrite_config=False,
                skip_sent=True,
                log_level="INFO",
                debug=False,
        ))
    """
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
    # Identifying needed runs from names
    ########################################
    regions = []

    autos = []
    auto_correlations = args.auto_correlations
    if config.get("fits", dict()).get("auto correlations", None) not in (None, ""):
        auto_correlations += config["fits"]["auto correlations"].split(" ")
    for auto in auto_correlations:
        absorber, region, absorber2, region2 = auto.replace(
            "-", ".").split(".")

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
        cross_correlations += (
            config["fits"]["cross correlations"].replace("-", ".").split(" ")
        )
    for cross in cross_correlations:
        tracer, absorber, region = cross.split(".")
        region = bookkeeper.validate_region(region)
        absorber = bookkeeper.validate_absorber(absorber)

        regions.append(region)
        crosses.append([absorber, region, tracer])

    regions = np.unique(regions)

    ########################################
    # Running delta extraction for calibration
    # and then all the deltas needed.
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
    # Running all the correlations needed
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
            absorber, region, tracer = cross

            cross_args = argparse.Namespace(
                bookkeeper_config=args.bookkeeper_config,
                overwrite_config=False,
                region=region,
                absorber=absorber,
                tracer=tracer,
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
    # Running fits
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
    """
    Parse and return command-line arguments for the full analysis.

    Returns:
    --------
        argparse.Namespace: Namespace containing parsed arguments with attributes:
            - bookkeeper_config (Path): Path to the bookkeeper YAML/JSON file.
            - system (str, optional): Job system (e.g., "slurm", "local").
            - overwrite (bool): Whether to overwrite existing output data.
            - overwrite_config (bool): Whether to overwrite the Bookkeeper
                config file.
            - skip_sent (bool): Whether to skip jobs that have already been sent.
            - auto_correlations (List[str]): List of auto-correlations to compute.
            - cross_correlations (List[str]): List of cross-correlations to
                compute.
            - no_deltas (bool): Skip delta extraction.
            - no_correlations (bool): Skip correlation computation.
            - no_fits (bool): Skip fitting step.
            - sampler (bool): Run sampler after fitting.
            - debug (bool): Enable debug mode.
            - only_write (bool): Only write scripts, do not submit them.
            - wait_for (List[int] or None): Job IDs to wait for before running.
            - log_level (str): Logging verbosity. Default is "INFO".

    Raises:
    --------
        - SystemExit: If invalid arguments are passed from the command line.

    Example:
    --------
        >>> args = get_args()
        >>> print(args.bookkeeper_config)
        PosixPath('myconfig.yaml')
    """
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

    parser.add_argument("--no-fits", action="store_true",
                        help="Don't measure fits.")

    parser.add_argument("--sampler", action="store_true",
                        help="Run the sampler.")

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--only-write", action="store_true", help="Only write scripts, not send them."
    )

    parser.add_argument("--wait-for", nargs="+", type=int,
                        default=None, required=False)

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
