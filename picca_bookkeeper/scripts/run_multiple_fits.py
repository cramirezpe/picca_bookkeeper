"""Script to run multiple fits one after another, ideal for interactive nodes and multiple fits (and only fits) to be run"""

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
                commands = compute_zeff._make_command().split('"')[1].strip().split(" ")
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
