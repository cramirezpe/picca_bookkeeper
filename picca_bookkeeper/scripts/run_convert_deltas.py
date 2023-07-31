""" Script to run build_new_deltas_format.py
through the bookkeeper.
"""
import argparse
import logging
import sys
from pathlib import Path

from picca_bookkeeper.bookkeeper import Bookkeeper


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

    script_args = {
        "log-level": args.log_level,
    }
    if args.nproc is not None:
        script_args["nproc"] = args.nproc

    task = bookkeeper.get_convert_deltas_tasker(
        region=args.region,
        calib_step=args.calibration_step,
        add_flux_properties=args.add_flux_properties,
        wait_for=args.wait_for,
        debug=args.debug,
        in_files=[
            bookkeeper.paths.delta_attributes_file(args.region),
        ]
        extra_args=script_args,
    )

    task.write_job()
    if not args.only_write:
        task.send_job()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config.",
    )

    parser.add_argument(
        "--region",
        type=str,
        choices=["lya", "lyb"],
        default="lya",
        help="Region to compute correlation in. Leave empty for calibration",
    )

    parser.add_argument(
        "--calibration-step",
        type=int,
        default=None,
        help="Calibration step to compute new deltas from. Only for calibration",
    )

    parser.add_argument(
        "--add-flux-properties",
        action="store_true",
        help="Compute flux properties as flux and flux variance and add them to delta files.",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
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
