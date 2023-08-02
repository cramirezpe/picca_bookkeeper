""" Script to run vega fit given a bookkeeper config file"""
import argparse
import logging
import sys
from pathlib import Path

from picca_bookkeeper.bookkeeper import Bookkeeper

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

    logger.info("Adding fit.")

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )

    fit = bookkeeper.get_fit_tasker(
        auto_correlations=args.auto_correlations,
        cross_correlations=args.cross_correlations,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
    )

    fit.write_job()
    if not args.only_write:
        fit.send_job()
        logger.info(f"Sent fit:\n\t{fit.jobid}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite output data."
    )

    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config.",
    )

    parser.add_argument(
        "--auto-correlations",
        type=str,
        nargs="+",
        default=[],
        help="List of auto-correlations to include in the vega fits. The format "
        "of the strings should be 'lya.lya-lya.lya'. This is to allow splitting",
    )

    parser.add_argument(
        "--cross-correlations",
        type=str,
        nargs="+",
        default=[],
        help="List of cross-correlations to include in the vega fits. The format of "
        "the strings should be 'lya.lya'",
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
