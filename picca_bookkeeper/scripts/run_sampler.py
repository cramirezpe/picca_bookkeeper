""" Script to run vega sampler given a bookkeeper config file"""

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
            logger.info(f"Sent compute covariance:\n\t{compute_covariance.jobid}")

        if config["fits"].get("smooth covariance", False):
            smooth_covariance = bookkeeper.get_smooth_covariance_tasker(
                wait_for=args.wait_for,
                system=args.system,
                overwrite=args.overwrite,
                skip_sent=args.skip_sent,
            )
            smooth_covariance.write_job()
            if not args.only_write:
                smooth_covariance.send_job()
                logger.info(f"Sent smooth covariance:\n\t{smooth_covariance.jobid}")

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
