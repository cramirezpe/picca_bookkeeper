"""
run_delta_extraction.py
-----------------------

Script to run picca_delta_extraction or picca_convert_transmission given a
bookkeeper config file.

This script runs eithert the delta extraction or conversion processes (such as
`picca_delta_extraction` or `picca_convert_transmission`) based on a provided
Bookkeeper config file. It manages calibration and delta extraction tasks,
handling job creation and submission according to user-specified options.

Functionality:
--------------
    - Loads and merges config settings from a Bookkeeper YAML config file.
    - Determines the necessary extraction steps, including calibration, and
      schedules these as tasks.
    - Supports chaining of calibration and extraction jobs.
    - Handles job writing and optional submission, with logging of actions and
      job IDs.
    - Allows granular control over which steps are executed
      (e.g., only calibration, skipping calibration, etc.).
    - Uses flexible command-line arguments to control workflow, logging,
      and system selection.

Interactions:
-------------
    - Bookkeeper class: loads configurations and provides methods to generate
            appropriate taskers for extraction and calibration steps.
    - DictUtils: used to merge default and user-provided configurations.
    - Taskers (ChainedTasker, DummyTasker): used to represent, write, and
            submit jobs for both calibration and delta extraction.

Usage:
------
Run this script from the command line, specifying a Bookkeeper configuration
file and any optional arguments:
    python run_delta_extraction.py <bookkeeper_config.yaml> [options]

Key Options:
  --system <name>        : Specify the system for which to run extraction.
  --region <region>      : Region to compute deltas in (default: "lya").
  --overwrite            : Force overwrite of output data.
  --overwrite-config     : Force overwrite of the bookkeeper config.
  --skip-sent            : Skip jobs that have already been submitted.
  --debug                : Enable debug-level logging.
  --only-calibration     : Only compute calibration steps.
  --skip-calibration     : Skip calibration if already computed.
  --only-write           : Only write job scripts, do not submit them.
  --wait-for <ids>       : List of job IDs to wait for before running.
  --log-level <level>    : Logging verbosity (default: INFO).

Example:
--------
    python run_delta_extraction.py configs/my_run.yaml --region lya
    --system supercomp --only-calibration --log-level DEBUG

This would perform only the calibration step for the given config, using the
'supercomp' system, and log in debug mode.

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.tasker import ChainedTasker, DummyTasker

if TYPE_CHECKING:
    from typing import Callable, Optional, Type
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Execute delta extraction using a Bookkeeper config file.

    This function handles delta extraction and optional
    calibration tasks, based on the provided config and CLI arguments.
    It sets up logging, loads and merges configuration settings, determines
    the appropriate tasker to use (standard delta extraction, raw, or calibration),
    and writes/submits job scripts accordingly.

    Arguments
    ----------
    args : Optional[argparse.Namespace], default=None
        Parsed command-line arguments. If not provided, `get_args()` will be
        called to obtain them.

    Behavior
    --------
        - If `--only-calibration` is passed, only the calibration task is executed.
        - If `--skip-calibration` is passed, skips calibration and proceeds to
          delta extraction.
        - If `raw mocks` or `true mocks` is specified in the config, uses the
          appropriate tasker for raw data extraction.
        - Uses `--only-write` to write jobs without sending.
        - Logs all task submissions and job IDs (unless in dummy/debug mode).

    Raises
    ------
    KeyError
        If required configuration keys are missing or improperly formatted.
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

    if args.only_calibration or (
        (
            (not raw and not true)
            and config["delta extraction"]["calib"] != 0
            and not args.skip_calibration
        )
    ):
        logger.info("Adding calibration step(s).")
        calibration = bookkeeper.get_calibration_extraction_tasker(
            system=args.system,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        calibration.write_job()
        if not args.only_write:
            calibration.send_job()

            if isinstance(calibration, ChainedTasker):
                for tasker in calibration.taskers:
                    if not isinstance(tasker, DummyTasker):
                        logger.info(
                            f"Sent calibration step:\n\t{tasker.jobid}")
            elif not isinstance(calibration, DummyTasker):
                logger.info(f"Sent calibration step:\n\t{calibration.jobid}")

        logger.info("Done.\n")

        if args.only_calibration:
            return

        get_tasker: Callable = bookkeeper.get_delta_extraction_tasker
    elif raw:
        get_tasker = bookkeeper.get_raw_deltas_tasker
    else:
        get_tasker = bookkeeper.get_delta_extraction_tasker

    logger.info(f"Adding deltas for region: {args.region}.")
    deltas = get_tasker(
        region=args.region,
        system=args.system,
        debug=args.debug,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    deltas.write_job()
    if not args.only_write:
        deltas.send_job()

        if not isinstance(deltas, DummyTasker):
            logger.info(
                f"Sent deltas for region:\n\t{args.region}: {deltas.jobid}")

    logger.info("Done\n")


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the delta extraction script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing all user-specified CLI flags and options.

    CLI Arguments
    -------------
    bookkeeper_config : str (positional)
        Path to the Bookkeeper configuration file (YAML or INI).
    --system : str, optional
        Name of the target system (e.g., 'slurm', 'bash') to use for job execution.
    --region : str, default='lya'
        Region in which to compute deltas (e.g., 'lya', 'civ').
    --overwrite : bool, default=False
        If set, overwrite existing output data.
    --overwrite-config : bool, default=False
        If set, overwrite the bookkeeper configuration file.
    --skip-sent : bool, default=False
        Skip job steps that have already been submitted or completed.
    --debug : bool, default=False
        Enable debug mode (may shorten job or dataset).
    --only-calibration : bool, default=False
        If set, run only the calibration steps and exit.
    --skip-calibration : bool, default=False
        If set, skip calibration steps even if enabled in config.
    --only-write : bool, default=False
        Only write job scripts; do not submit them for execution.
    --wait-for : List[int], optional
        List of job IDs to wait for before running this workflow.
    --log-level : {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'},
        default='INFO'
        Logging verbosity level.

    Example
    -------
        >>> args = get_args()
        >>> args.region
        'lya'
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
        "--region",
        type=str,
        default="lya",
        help="Region to compute deltas in",
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
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--only-calibration",
        action="store_true",
        help="Only compute calibration steps.",
    )

    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration step if already computed.",
    )

    parser.add_argument(
        "--only-write",
        action="store_true",
        help="Only write scripts, do not send them.",
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
