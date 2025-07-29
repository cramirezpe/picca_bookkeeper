"""
run_sampler.py
--------------

This script automates the process of running the vega sampler and associated
tasks (such as covariance computation, smoothing, and zeff computation),
based on a user-provided bookkeeper config file.

Functionality:
--------------
    - Reads a bookkeeper config file and merges it with default settings.
    - Optionally computes and submits jobs for covariance matrix calculation,
      covariance smoothing, and zeff computation, depending on the config.
    - Submits the main sampler (fit) job using the vega sampler.
    - Handles job dependencies by managing "wait_for" relationships between tasks.
    - Provides options to write job scripts without submitting them,
      overwrite outputs, and skip previously sent jobs.
    - Uses logging to inform users about the progress and job submissions.

Usage:
------
Run from the command line:
    python run_sampler.py <bookkeeper_config> [options]

Positional Arguments:
    bookkeeper_config       : Path to the YAML config file for the bookkeeper.

Optional Arguments:
    --overwrite             : Force overwrite of output data.
    --overwrite-config      : Force overwrite of the bookkeeper config file in
                              folder.
    --skip-sent             : Skip jobs that were already sent.
    --only-write            : Only write scripts, do not send jobs.
    --wait-for              : List of job IDs to wait for before running the
                              current step.
    --log-level             : Set the logging level (default: INFO; choices:
                              CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET).

Example:
--------
    python run_sampler.py config/bookkeeper.yaml --overwrite --log-level DEBUG

Dependencies:
-------------
    - vega sampler: External tool assumed to be accessible via generated
      job scripts.

Interactions:
-------------
    - Bookkeeper class from picca_bookkeeper.bookkeeper: handles config parsing
      and job/task creation.
    - DictUtils from picca_bookkeeper.dict_utils: merge default and user configs.
    - Methods like get_covariance_matrix_tasker(), get_smooth_covariance_tasker(),
      get_compute_zeff_tasker(), and get_sampler_tasker() to generate appropriate
      job scripts and, optionally, submit them.

Notes:
------
    - The script manages dependencies between jobs, ensuring that (for example)
      smoothing only happens after covariance computation if both are enabled.
    - Log output provides job submission information, including job IDs,
      which may be needed for further processing or job tracking.

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Entry point for running the vega sampler and optional pre-processing tasks.

    Submits SLURM jobs for:
    -----------------------
        - Covariance matrix computation (if enabled)
        - Covariance smoothing (if enabled)
        - Zeff parameter computation (if enabled)
        - The final sampling / fit task

    Uses the Bookkeeper to configure tasks and manages dependencies via the
    `wait_for` option. Tasks can be optionally written without submission,
    and previously submitted jobs can be skipped.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]):
            Parsed command-line arguments. If None, arguments are parsed via
            get_args().

    Raises:
    -------
        - FileNotFoundError:
            If the provided Bookkeeper config file does not exist.
        - KeyError:
            If expected fields like 'fits' or its subkeys are missing in the config.
    """
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info("Adding sampler.")

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
        compute_covariance = bookkeeper.get_covariance_matrix_tasker(
            wait_for=args.wait_for,
            system=args.system,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        compute_covariance.write_job()
        if not args.only_write:
            compute_covariance.send_job()
            logger.info(
                f"Sent compute covariance:\n\t{compute_covariance.jobid}")
            cov_jobid = compute_covariance.jobid

    if config["fits"].get("smooth covariance", False):
        # Use jobid from compute_covariance only if it exists and we're not in only-write mode
        wait_jobid = [
            cov_jobid] if cov_jobid and not args.only_write else args.wait_for

        smooth_covariance = bookkeeper.get_smooth_covariance_tasker(
            wait_for=wait_jobid,
            system=args.system,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        smooth_covariance.write_job()

        if not args.only_write:
            import time
            time.sleep(15)  # Optional delay for file system sync
            smooth_covariance.send_job()
            logger.info(
                f"Sent smooth covariance:\n\t{smooth_covariance.jobid}")

    if config["fits"].get("compute zeff", False):
        compute_zeff = bookkeeper.get_compute_zeff_tasker(
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        compute_zeff.write_job()
        if not args.only_write:
            compute_zeff.send_job()
            logger.info(f"Sent compute zeff:\n\t{compute_zeff.jobid}")

    sampler = bookkeeper.get_sampler_tasker(
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    sampler.write_job()
    if not args.only_write:
        sampler.send_job()
        logger.info(f"Sent fit:\n\t{sampler.jobid}")


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for run_sampler.py.

    Returns:
    --------
        argparse.Namespace: Parsed arguments containing:
            - bookkeeper_config (Path): Path to the Bookkeeper YAML config file.
            - overwrite (bool): If True, force overwrite of output data.
            - overwrite_config (bool): If True, overwrite Bookkeeper-generated
                    config files.
            - skip_sent (bool): If True, skip jobs that have already been
                    submitted.
            - only_write (bool): If True, generate SLURM scripts but do not
                    submit them.
            - wait_for (Optional[List[int]]): List of job IDs that submitted
                    jobs should wait for.
            - log_level (str): Logging verbosity level (default: INFO).

    Raises:
    -------
        SystemExit: If required arguments are missing or invalid.
    """
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
