"""
run_fit.py
----------

Script to run vega fit given a bookkeeper config file

This script automates fitting using Vega. It uses a YAML/JSON config file
(the "bookkeeper config") to define which steps to execute: e.g. the covariance
matrix, smoothing the covariance, calculating zeff, and running the fit itself.
Each job is managed through the Bookkeeper, which handles dependencies, job
writing, and job submission.

Functionality:
-------------
    - Reads config from a specified bookkeeper config file, merging with defaults.
    - Optionally computes and submits a covariance matrix job.
    - Optionally smooths the covariance matrix, waiting for the covariance job
      if needed.
    - Optionally computes zeff.
    - Always submits a fit job.
    - All jobs can be written without execution (for inspection or batch submission).
    - Supports job dependency management and logging of job status and locations.

Usage:
------
python run_fit.py [bookkeeper_config] [options]

Arguments:
    bookkeeper_config     : Path to the bookkeeper YAML/JSON configuration file.

Options:
    --system SYSTEM       : Specify the system for job submission (e.g., slurm, local).
    --overwrite           : Force overwrite of output data.
    --overwrite-config    : Force overwrite of the bookkeeper config.
    --skip-sent           : Skip jobs that have already been submitted.
    --only-write            Only write scripts; do not submit jobs.
    --wait-for JOBID [JOBID ...] : List of job IDs to wait for before execution.
    --log-level LEVEL     : Set logging verbosity (default: INFO).

Example:
--------
    python run_fit.py my_config.yaml --system slurm --overwrite

Interactions:
-------------
    - Bookkeeper: Class coordinating config reading, job creation, and path
                  management.
    - DictUtils: Used to merge default and user-provided configs.
    - Taskers: Job wrappers (covariance, smoothing, zeff, fit), returned by
                  Bookkeeper.
    - Jobs are written/submitted using methods on these taskers and may depend
      on one another (e.g., smoothing waits for covariance).

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.tasker import DummyTasker

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Execute the Vega fitting using a Bookkeeper config file.

    This function organizes a sequence of fitting-related jobs (e.g., computing
    the covariance matrix, smoothing it, calculating zeff, and running the final
    fit) according to settings provided in a Bookkeeper YAML or JSON config file.

    Arguments
    ----------
    args : Optional[argparse.Namespace], default=None
        Parsed command-line arguments. If not supplied, they are obtained by
        calling `get_args()`.

    Behavior
    --------
        - Initializes logging and reads configuration via the Bookkeeper.
        - Submits a covariance matrix computation job if `compute covariance`
          is True.
        - If configured, submits a smoothing job after covariance.
        - Submits a zeff computation job if `compute zeff` is True.
        - Always submits the final fit job.
        - Supports deferred execution (`--only-write`) and job chaining via
          `--wait-for`.

    Logging
    -------
        - Job submission IDs are logged unless jobs are `DummyTasker` placeholders.
        - The final fit path is printed at the end of execution.

    Raises
    ------
    KeyError
        If expected configuration sections (e.g., "fits") are missing or malformed.

    Example
    -------
        >>> main()
        INFO:Adding compute covariance matrix.
        INFO:Sent compute covariance:
            123456
        ...
        INFO:Fit location: /output/path/to/fits/
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

    cov_jobid = None

    if config["fits"].get("compute covariance", False):
        logger.info("Adding compute covariance matrix.")
        compute_covariance = bookkeeper.get_covariance_matrix_tasker(
            wait_for=args.wait_for,
            system=args.system,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        compute_covariance.write_job()
        if not args.only_write:
            compute_covariance.send_job()
            if not isinstance(compute_covariance, DummyTasker):
                cov_jobid = compute_covariance.jobid
                logger.info(f"Sent compute covariance:\n\t{cov_jobid}")

        logger.info("Done.\n")

        if config["fits"].get("smooth covariance", False):
            logger.info("Adding smooth covariance matrix.")
            smooth_covariance = bookkeeper.get_smooth_covariance_tasker(
                wait_for=[cov_jobid] if cov_jobid else args.wait_for,
                system=args.system,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            smooth_covariance.write_job()
            if not args.only_write:
                smooth_covariance.send_job()
                if not isinstance(smooth_covariance, DummyTasker):
                    logger.info(
                        f"Sent smooth covariance:\n\t{smooth_covariance.jobid}")

            logger.info("Done.\n")

    if config["fits"].get("compute zeff", False):
        logger.info("Adding compute zeff.")
        compute_zeff = bookkeeper.get_compute_zeff_tasker(
            wait_for=args.wait_for,
            system=args.system,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        compute_zeff.write_job()
        if not args.only_write:
            compute_zeff.send_job()

            if not isinstance(compute_zeff, DummyTasker):
                logger.info(f"Sent compute zeff:\n\t{compute_zeff.jobid}")

        logger.info("Done.\n")

    logger.info("Adding fit.")
    fit = bookkeeper.get_fit_tasker(
        wait_for=args.wait_for,
        system=args.system,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    fit.write_job()
    if not args.only_write:
        fit.send_job()

        if not isinstance(fit, DummyTasker):
            logger.info(f"Sent fit:\n\t{fit.jobid}")

    logger.info("Done.\n")

    logger.info(f"Fit location: {bookkeeper.paths.fits_path}\n")


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for the Vega fitting process.

    This function defines the CLI interface to `run_fit.py`, specifying required
    and optional flags that control system selection, job behavior, and verbosity.

    Returns
    -------
    argparse.Namespace
        An object containing all parsed arguments and flags.

    CLI Arguments
    -------------
    bookkeeper_config : Path
        Path to the Bookkeeper YAML or JSON configuration file.

    --system : str, optional
        The name of the execution system (e.g., 'slurm', 'local') to use for
        job submission.

    --overwrite : bool, default=False
        If specified, forces output files to be overwritten.

    --overwrite-config : bool, default=False
        If specified, overwrites the Bookkeeper config file during processing.

    --skip-sent : bool, default=False
        Skip submitting jobs that have already been sent.

    --only-write : bool, default=False
        Write job scripts to disk but do not submit them.

    --wait-for : List[int], optional
        A list of job IDs to wait for before executing any task.

    --log-level : {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'},
        default='INFO'
        Set the verbosity level of logging output.

    Example
    -------
        >>> args = get_args()
        >>> print(args.bookkeeper_config)
        configs/vega_run.yaml
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
