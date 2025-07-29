"""
run_xcf.py
----------

Script to run picca_xcf and export given a bookkeeper config file.

Script to automate the creation and (optionally) submission of jobs for running
picca cross-correlation (xcf), distortion matrix computation, metal matrix
calculation, and exporting results, as defined in a Bookkeeper config file.

Functionality:
--------------
Performs the following steps:
    1. Initializes a Bookkeeper object from the provided config file.
    2. Sets up and writes (and optionally submits) jobs for:
       - Cross-correlation (xcf)
       - Distortion matrix (xdmat), if required by config
       - Metal matrix (xmetal), if required by config
       - Exporting results (xcf_exp)
    3. Handles logging, job dependencies, and customizable execution parameters
       via CLI.

Dependencies:
-------------
    - picca_bookkeeper.bookkeeper.Bookkeeper: Manages the workflow and job generation.
    - picca_bookkeeper.tasker.DummyTasker:    Used to detect dry-run operations.

Usage:
------
To run the script from the command line:
    python run_xcf.py <bookkeeper_config> [options]

Required arguments:
    bookkeeper_config     : Path to the YAML configuration file for the workflow.

Optional arguments:
    --system <str>        : System name for job submission (e.g., SLURM cluster).
    --region <str>        : Region to compute correlation in ('lya' or 'lyb').
                            Default: 'lya'.
    --tracer <str>        : Tracer for the correlation (e.g., 'qso').
                            Default: 'qso'.
    --absorber <str>      : Absorber for the correlation (e.g., 'lya').
                            Default: 'lya'.
    --overwrite           : Overwrite existing output data.
    --overwrite-config    : Overwrite the bookkeeper configuration.
    --skip-sent           : Skip jobs that were already sent.
    --debug               : Enable debug mode for verbose output.
    --only-write          : Only write job scripts, do not submit them.
    --wait-for <int ...>  : List of job IDs to wait for before submitting.
    --log-level <LEVEL>   : Set logging level (CRITICAL, ERROR, WARNING, INFO,
                            DEBUG, NOTSET). Default: INFO.

Example:
--------
python run_xcf.py config/my_bookkeeper.yml --region lya --tracer qso --absorber lya --overwrite

Interactions:
-------------
    - Bookkeeper class: interpret and manage the workflow described by the config file.
    - Tasker classes (e.g., XcfTasker, XdmatTasker, XmetalTasker, XcfExpTasker):
        - `get_xcf_tasker`, `get_xdmat_tasker`, `get_xmetal_tasker`, and `get_xcf_exp_tasker`.
    - Writes and optionally submits job scripts for each phase.

See Also:
---------
    - picca_bookkeeper.bookkeeper.Bookkeeper
    - picca_bookkeeper/tasker.py
    - Bookkeeper YAML config documentation / examples for details on available options.

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import DummyTasker

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Entry point for running picca_xcf via a Bookkeeper config.

    This function initializes a Bookkeeper object from the provided YAML config
    file and sets up job submission (or script generation) for the following steps:
        1. Cross-correlation computation (picca_xcf)
        2. Distortion matrix computation (xdmat), if enabled
        3. Metal matrix computation (xmetal), if enabled and not handled by Vega
        4. Exporting cross-correlation results (xcf_exp)

    For each step, the function:
        - Writes job scripts to file
        - Optionally submits them to the job scheduler (sbatch)
        - Logs job IDs if submission occurs
        - Respects dependency chains via `wait_for`

    Arguments:
    ---------
        args (Optional[argparse.Namespace]):
            Parsed command-line arguments. If None, they are automatically
            parsed using `get_args()`.

    Raises:
    -------
        - FileNotFoundError:
            If the specified Bookkeeper YAML config file does not exist.
        - KeyError:
            If required config fields (e.g., 'fits') are missing or malformed.

    Notes:
    ------
        - The actual job execution is system-dependent and managed by the
          Tasker classes.
        - Skips sending jobs if `--only-write` is used.
        - Skips redundant submissions if `--skip-sent` is provided.
    """
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info(
        f"Adding cross-correlation: {args.absorber}{args.region}_{args.tracer}")

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )

    xcf = bookkeeper.get_xcf_tasker(
        region=args.region,
        absorber=args.absorber,
        tracer=args.tracer,
        debug=args.debug,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    xcf.write_job()
    if not args.only_write:
        xcf.send_job()

        if not isinstance(xcf, DummyTasker):
            logger.info(
                "Sent cross-correlation "
                f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                f"{xcf.jobid}"
            )
    logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get("no distortion", False):
        logger.info(
            f"Adding distortion matrix: {args.absorber}{args.region}_{args.tracer}"
        )
        xdmat = bookkeeper.get_xdmat_tasker(
            region=args.region,
            absorber=args.absorber,
            tracer=args.tracer,
            debug=args.debug,
            system=args.system,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        xdmat.write_job()
        if not args.only_write:
            xdmat.send_job()

            if not isinstance(xdmat, DummyTasker):
                logger.info(
                    "Sent distortion matrix "
                    f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                    f"{xdmat.jobid}"
                )
        logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get(
        "no metals", False
    ) and not bookkeeper.config.get("fits", dict()).get("vega metals", False):
        # Compute metals if metals should be included and metals are not going
        # to be computed by vega.

        logger.info(
            f"Adding metal matrix: {args.absorber}{args.region}_{args.tracer}")
        metal = bookkeeper.get_xmetal_tasker(
            region=args.region,
            absorber=args.absorber,
            tracer=args.tracer,
            system=args.system,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        metal.write_job()
        if not args.only_write:
            metal.send_job()

            if not isinstance(metal, DummyTasker):
                logger.info(
                    "Sent metal matrix "
                    f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                    f"{metal.jobid}"
                )
        logger.info("Done.\n")

    logger.info(f"Adding export: {args.absorber}{args.region}_{args.tracer}")
    xcf_exp = bookkeeper.get_xcf_exp_tasker(
        region=args.region,
        absorber=args.absorber,
        tracer=args.tracer,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    xcf_exp.write_job()
    if not args.only_write:
        xcf_exp.send_job()

        if not isinstance(xcf_exp, DummyTasker):
            logger.info(
                "Sent export "
                f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                f"{xcf_exp.jobid}"
            )
    logger.info("Done.\n")


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for run_xcf.py.

    Returns:
    --------
        argparse.Namespace:
            Parsed arguments with the following attributes:

            - bookkeeper_config (Path): Path to the YAML Bookkeeper config file.

            - --system (Optional[str]):
                Optional name of the computing system or scheduler (e.g., SLURM cluster).
                Used to tailor job script generation and submission behavior.

            - --region (str, default='lya'):
                Spectral region to analyze ('lya' or 'lyb').

            - --tracer (str, default='qso'):
                Tracer to correlate against (e.g., 'qso', 'gal').

            - --absorber (str, default='lya'):
                Absorber type in the correlation (e.g., 'lya', 'lyb').

            - --overwrite (bool):
                If True, overwrite existing outputs for each task.

            - --overwrite_config (bool):
                If True, overwrite Bookkeeper-generated config files.

            - --skip_sent (bool):
                If True, skip submission of jobs that were already sent previously.

            - --debug (bool):
                If True, enable verbose debug output in the generated job scripts.

            - --only_write (bool):
                If True, generate job scripts but do not submit them.

            - --wait_for (Optional[List[int]]):
                List of job IDs that current jobs should wait on before execution.

            - --log_level (str, default='INFO'):
                Logging verbosity level. Choices:
                ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'].

    Raises:
    -------
        SystemExit:
            If required positional arguments are missing or invalid.
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
        choices=["lya", "lyb"],
        default="lya",
        help="Region to compute correlation in",
    )

    parser.add_argument(
        "--tracer", type=str, default="qso", help="Tracer to use for the correlation."
    )

    parser.add_argument(
        "--absorber", type=str, default="lya", help="Absorber to use for correlations"
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
