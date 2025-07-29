"""
run_multiple_fits.py
--------------------

Script to run multiple fit tasks sequentially using the picca_bookkeeper framework,
ideal for interactive nodes and multiple fits (and only fits) to be run.

This script is designed to automate the process of running multiple fits
(and optionally, the computation of zeff and related parameters) as specified
by one or more Bookkeeper configuration files.
It is especially useful for batch or interactive environments where many fits
need to be performed in sequence.

Functionality:
--------------
    - Iterates over given Bookkeeper config files, initializing a Bookkeeper
        instance for each.
    - Merges user configs with Bookkeeper defaults to prepare the fit setup.
    - Optionally computes zeff and related parameters if requested by the
        config and not explicitly skipped.
    - Runs the fit task for each configuration, managing output and error
        logging per run.
    - Skips redundant computation if fit output already exists (
        unless overwrite is specified).

Arguments:
-----------
    - bookkeeper_configs    : One or more paths to Bookkeeper config YAML files.
    - --system              : (Optional) System name, passed through to Bookkeeper
                              (default: None).
    - --overwrite           : Overwrite fit outputs even if they already exist.
    - --overwrite-config    : Overwrite Bookkeeper-generated configs.
    - --skip-zeff           : Skip computation of zeff and related quantities.
    - --log-level           : Set logging verbosity (default: INFO).

Usage:
------
    python run_multiple_fits.py config1.yaml config2.yaml --overwrite --log-level DEBUG

    python picca_bookkeeper/scripts/run_multiple_fits.py my_config.yaml --skip-zeff

Interactions:
-------------
    - Relies heavily on Bookkeeper objects for organizing fits and optional
        zeff computation.
    - Uses DictUtils to merge default and user configurations.
    - Calls Bookkeeper's get_compute_zeff_tasker and get_fit_tasker methods
        to prepare execution commands.
    - Interacts with Bookkeeper file paths for input, output, and error tracking.

Note:
------
    - The script expects valid Bookkeeper config files specifying fit details.
    - Output and error logs for each fit and zeff computation are managed
      according to the Bookkeeper paths and configuration.

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function for running multiple fit tasks via Bookkeeper configs.

    This function iterates over one or more provided Bookkeeper config
    files, performing the following for each:
        - Loads and merges Bookkeeper default and user configurations.
        - Computes zeff parameters if enabled and not skipped.
        - Runs the fit task if no prior output exists or overwrite is requested.
        - Logs task outcomes for each operation (zeff and fit).

    Arguments:
    ----------
        args (Optional[argparse.Namespace]):
            Parsed command-line arguments. If None, they are parsed via get_args().

    Raises:
    -------
        - FileNotFoundError:
            If a specified Bookkeeper config file does not exist.
        - KeyError:
            If required config keys (e.g. 'fits') are missing.
    """
    if args is None:
        args = get_args()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    for bookkeeper_config in args.bookkeeper_configs:
        bookkeeper = Bookkeeper(
            bookkeeper_config,
            read_mode=False,
            overwrite_config=args.overwrite_config,
        )

        config = DictUtils.merge_dicts(
            bookkeeper.defaults,
            bookkeeper.config,
        )

        logger.info(f"Adding fit:{bookkeeper.paths.fits_path}")

        if config["fits"].get("compute zeff", False) and not args.skip_zeff:
            computed_params_file = bookkeeper.paths.fit_computed_params_out()
            if (
                not computed_params_file.is_file()
                or computed_params_file.stat().st_size < 20
                or args.overwrite
            ):
                # In this case we need to compute first the params
                compute_zeff = bookkeeper.get_compute_zeff_tasker(
                    overwrite=True,
                    skip_sent=False,
                )
                commands = compute_zeff._make_command().split('"')[
                    1].strip().split(" ")
                with open(compute_zeff.slurm_header_args["output"], "w") as out, open(
                    compute_zeff.slurm_header_args["error"], "w"
                ) as err:
                    retcode = run(commands, stdout=out, stderr=err)
                logger.info(f"\tcompute zeff: {retcode.returncode}")

        fit = bookkeeper.get_fit_tasker(
            overwrite=True,
            skip_sent=False,
        )
        if bookkeeper.paths.fit_out_fname().stat().st_size < 20 or args.overwrite:
            commands = fit._make_command().split('"')[1].strip().split(" ")
            with open(fit.slurm_header_args["output"], "w") as out, open(
                fit.slurm_header_args["error"], "w"
            ) as err:
                retcode = run(commands, stdout=out, stderr=err)
            logger.info(f"\tfit: {retcode.returncode}")
        else:
            logger.info(f"\tfit already finished.")


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments for run_multiple_fits.py.

    Returns:
    -------
        argparse.Namespace: Parsed arguments with the following attributes:
            - bookkeeper_configs (List[Path]): One or more paths to Bookkeeper
              config YAML files.
            - --system (str or None): Optional system label to pass into Bookkeeper.
            - --overwrite (bool): If True, overwrite existing outputs.
            - --overwrite-config (bool): If True, overwrite the Bookkeeper-generated
              config files.
            - --skip-zeff (bool): If True, skip computation of zeff and related
              quantities.
            - --log-level (str): Logging verbosity level (default: 'INFO').

    Raises:
    -------
        SystemExit: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_configs",
        type=Path,
        nargs="*",
        help="Path to bookkeeper file to use",
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
        "--skip-zeff",
        action="store_true",
        help="Skip computation of zeff and related quantities.",
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
