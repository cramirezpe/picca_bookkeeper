"""
run_cf.py
---------

Script to computate and export correlation functions (CFs) and related matrices,
adnd export a given bookkeeper config file.

Functionality:
--------------
This script automates generating, post-processing, and exporting correlation functions
and associated matrices (distortion and metal matrices) based on a config file.
    - Running the main CF computation (`picca_cf`)
    - Optionally generating distortion and metal matrices, depending on config flags
    - Exporting results
    - Managing job scripts and submission (e.g., SLURM or bash), with support
      for dry-run / only-write mode

Usage:
------
To use this script, run it from the command line, specifying your bookkeeper
config file and any desired options. The script will write and submit
(unless --only-write is specified) all necessary job scripts.

Example:
    python run_cf.py config/mybookkeeper.ini --region lya --absorber lya --overwrite

Main CLI Arguments:
-------------------
    - bookkeeper_config (positional): Path to the bookkeeper YAML/INI config file.
    - --system: Target execution system (e.g., 'slurm_perlmutter', 'bash', or
                                         as specified in config).
    - --overwrite: Overwrite output data if present.
    - --overwrite-config: Overwrite the bookkeeper configuration file.
    - --skip-sent: Skip tasks/jobs that have already been submitted.
    - --region/--region2: Specify primary/secondary spectral regions
                          (for cross-correlations).
    - --absorber/--absorber2: Specify primary/secondary absorber species.
    - --debug: Enable debug mode (short jobs, limited data).
    - --only-write: Only write job scripts, do not submit.
    - --wait-for: Specify job IDs to wait for before running jobs.
    - --log-level: Set logging verbosity.

Interactions:
-------------
- `picca_bookkeeper.bookkeeper.Bookkeeper` to manage workflow, job writing, and
   submission.
- `picca_bookkeeper.tasker.DummyTasker` to avoid real job submission in certain
   modes.
- Calls Bookkeeper methods:
    - `get_cf_tasker`: Prepares and manages the main CF calculation.
    - `get_dmat_tasker`: (If enabled) Sets up distortion matrix calculation.
    - `get_metal_tasker`: (If enabled) Sets up metal matrix calculation.
    - `get_cf_exp_tasker`: Handles export/postprocessing of CF results.
- Relies on config logic and file path management from Bookkeeper and associated
  utilities.
- Job scripts and logs are written to paths derived from the config.
- Results can be chained together via the `--wait-for` argument for workflow control.

See Also:
---------
- picca_bookkeeper/bookkeeper.py : Bookkeeper class and job orchestration logic.
- picca_bookkeeper/tasker.py     : Tasker and DummyTasker job wrappers.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

import argparse
import logging
import sys
from pathlib import Path

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import DummyTasker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main routine for generating correlation functions (CFs),
    distortion matrices, metal matrices, and export scripts based on the
    provided Bookkeeper config.

    Functionality:
    --------------
        - Initializes a Bookkeeper object.
        - Constructs and optionally submits SLURM/bash jobs for:
            - CF computation (`picca_cf`)
            - Distortion matrix (unless disabled via config)
            - Metal matrix (unless disabled or delegated to Vega)
            - Export/postprocessing of CF results
        - Logs submission info and job IDs when applicable.

    Arguments:
    ----------
        args (argparse.Namespace, optional): Parsed command-line arguments.
            If None, the function will internally call `get_args()` to parse
            CLI input.

    Raises:
    -------
        - ValueError: If required arguments are missing or incompatible.
        - FileNotFoundError: If config or input paths from Bookkeeper are invalid.
        - RuntimeError: If job creation or submission fails unexpectedly.

    Example:
    --------
        main(get_args())
    """
    if args is None:
        args = get_args()

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info(
        f"Adding auto-correlation: "
        f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
    )
    cf = bookkeeper.get_cf_tasker(
        region=args.region,
        region2=args.region2,
        absorber=args.absorber,
        absorber2=args.absorber2,
        system=args.system,
        debug=args.debug,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    cf.write_job()
    if not args.only_write:
        cf.send_job()
        if not isinstance(cf, DummyTasker):
            logger.info(
                "Sent auto-correlation "
                f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                f"{cf.jobid}"
            )
    logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get("no distortion", False):
        logger.info(
            f"Adding distortion matrix: "
            f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
        )
        dmat = bookkeeper.get_dmat_tasker(
            region=args.region,
            region2=args.region2,
            absorber=args.absorber,
            absorber2=args.absorber2,
            system=args.system,
            wait_for=args.wait_for,
            debug=args.debug,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        dmat.write_job()
        if not args.only_write:
            dmat.send_job()

            if not isinstance(dmat, DummyTasker):
                logger.info(
                    "Sent distortion matrix "
                    f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                    f"{dmat.jobid}"
                )
        logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get(
        "no metals", False
    ) and not bookkeeper.config.get("fits", dict()).get("vega metals", False):
        # Compute metals if metals should be included and metals are not going
        # to be computed by vega.

        logger.info(
            f"Adding metal matrix: "
            f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
        )
        metal = bookkeeper.get_metal_tasker(
            region=args.region,
            region2=args.region2,
            absorber=args.absorber,
            absorber2=args.absorber2,
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
                    f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                    f"{metal.jobid}"
                )
        logger.info("Done.\n")

    logger.info(
        f"Adding export: "
        f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
    )

    cf_exp = bookkeeper.get_cf_exp_tasker(
        region=args.region,
        region2=args.region2,
        absorber=args.absorber,
        absorber2=args.absorber2,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    cf_exp.write_job()
    if not args.only_write:
        cf_exp.send_job()

        if not isinstance(cf_exp, DummyTasker):
            logger.info(
                "Sent export "
                f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                f"{cf_exp.jobid}"
            )
    logger.info("Done.\n")


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for CF generation and export.

    Returns:
    --------
        argparse.Namespace: Parsed arguments with the following attributes:
            - bookkeeper_config (Path): Path to the Bookkeeper YAML or INI config.
            - system (str | None): Execution system ('slurm_perlmutter', 'bash', etc.).
            - overwrite (bool): Whether to overwrite output directories if they exist.
            - overwrite_config (bool): Whether to overwrite the Bookkeeper config file.
            - skip_sent (bool): Whether to skip tasks that have already been submitted.
            - region (str): Primary spectral region (e.g., 'lya').
            - region2 (str | None): Secondary region (for cross-correlations).
            - absorber (str): Primary absorber (e.g., 'lya').
            - absorber2 (str | None): Secondary absorber for cross-correlation
                (default: None).
            - debug (bool): Enable debug mode (smaller data, shorter run time).
            - only_write (bool): Only write job scripts; do not submit them.
            - wait_for (List[int] | None): SLURM job IDs this job should wait for.
            - log_level (str): Logging verbosity level.

    Example:
    --------
        Namespace(
            bookkeeper_config=Path("config.yaml"),
            system="slurm_perlmutter",
            overwrite=True,
            overwrite_config=False,
            skip_sent=False,
            region="lya",
            region2="lya",
            absorber="lya",
            absorber2=None,
            debug=False,
            only_write=False,
            wait_for=[12345],
            log_level="INFO"
        )
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
        "--region",
        type=str,
        default="lya",
        help="Region to compute correlation in",
    )

    parser.add_argument(
        "--region2",
        type=str,
        default=None,
        help="Second region (for cross-correlations between forests",
    )

    parser.add_argument(
        "--absorber", type=str, default="lya", help="Absorber to use for correlations"
    )

    parser.add_argument(
        "--absorber2",
        type=str,
        default=None,
        help="Second absorber (for cross-correlations between forests)",
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
