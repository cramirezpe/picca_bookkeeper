""" Script to run picca_delta_extraction or picca_convert_transmission
given a bookkeeper config file."""
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

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )

    continuum_type = bookkeeper.config["delta extraction"]["prefix"]

    if args.only_calibration or (
        (str(continuum_type) not in ("raw", "True"))
        and bookkeeper.config["delta extraction"]["calib"] != "0"
        and not args.skip_calibration
    ):
        logger.info("Adding calibration step(s).")
        calibration = bookkeeper.get_calibration_extraction_tasker(
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
        )
        calibration.write_job()
        if not args.only_write:
            calibration.send_job()

        if args.only_calibration:
            if args.only_write:
                return
            else:
                print(calibration.jobid)
                return calibration.jobid

        tasker = bookkeeper.get_delta_extraction_tasker
        wait_for = calibration
    elif continuum_type == "raw":
        tasker = bookkeeper.get_raw_deltas_tasker
        wait_for = args.wait_for
    else:
        tasker = bookkeeper.get_delta_extraction_tasker
        wait_for = args.wait_for

    logger.info(f"Adding deltas for region: {args.region}.")
    deltas = tasker(
        region=args.region,
        system=None,
        debug=args.debug,
        wait_for=wait_for,
        overwrite=args.overwrite,
    )
    deltas.write_job()
    if args.only_write:
        return
    else:
        deltas.send_job()
        print(deltas.jobid)
        return deltas.jobid


def get_args():
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
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--only-calibration",
        action="store_true",
        help="Only compute calibration steps.",
    )

    parser.add_argument(
        "--skip_calibration",
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
