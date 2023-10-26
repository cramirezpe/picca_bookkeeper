""" Script to run picca_delta_extraction or picca_convert_transmission
given a bookkeeper config file."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import Tasker

if TYPE_CHECKING:
    from typing import Callable, Optional, Type
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

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config, read_mode=False,
    )

    config = DictUtils.merge_dicts(
        bookkeeper.defaults,
        bookkeeper.config,
    )

    continuum_type = config["delta extraction"]["prefix"]

    if args.only_calibration or (
        (
            str(continuum_type) not in ("raw", "True")
            and config["delta extraction"]["calib"] != 0
            and not args.skip_calibration
        )
    ):
        logger.info("Adding calibration step(s).")
        calibration = bookkeeper.get_calibration_extraction_tasker(
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        calibration.write_job()
        if not args.only_write:
            calibration.send_job()
            logger.info(f"Sent calibration step(s):\n\t{calibration.jobid}")

        if args.only_calibration:
            return

        get_tasker: Callable = bookkeeper.get_delta_extraction_tasker
    elif continuum_type == "raw":
        get_tasker = bookkeeper.get_raw_deltas_tasker
    else:
        get_tasker = bookkeeper.get_delta_extraction_tasker

    logger.info(f"Adding deltas for region: {args.region}.")
    deltas = get_tasker(
        region=args.region,
        system=None,
        debug=args.debug,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    deltas.write_job()
    if not args.only_write:
        deltas.send_job()
        logger.info(f"Sent deltas for region:\n\t{args.region}: {deltas.jobid}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
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
