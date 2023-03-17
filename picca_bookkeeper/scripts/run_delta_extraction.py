""" Script to run picca_delta_extraction or picca_convert_transmission
given a bookkeeper config file."""
from pathlib import Path
import argparse
from picca_bookkeeper.bookkeeper import Bookkeeper


def main(args=None):
    if args is None:
        args = get_args()
    bookkeeper = Bookkeeper(args.bookkeeper_config)

    continuum_type = bookkeeper.config["continuum fitting"]["prefix"]

    if args.only_calibration or (
        (continuum_type in ("dMdB20", "custom"))
        and bookkeeper.config["continuum fitting"]["calib"] != "0"
    ):
        calibration = bookkeeper.get_calibration_extraction_tasker(
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
        )
        calibration.write_job()
        calibration.send_job()

        if args.only_calibration:
            print(calibration.jobid)
            return calibration.jobid

        tasker = bookkeeper.get_delta_extraction_tasker
        wait_for = calibration
    elif (
        (continuum_type == "dMdB20")
        or (continuum_type == "true")
        or (continuum_type == "custom")
    ):
        tasker = bookkeeper.get_delta_extraction_tasker
        wait_for = args.wait_for
    elif continuum_type == "raw":
        tasker = bookkeeper.get_raw_deltas_tasker
        wait_for = args.wait_for

    deltas = tasker(
        region=args.region,
        system=None,
        debug=args.debug,
        wait_for=wait_for,
    )

    deltas.write_job()
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
        choices=["lya", "lyb"],
        default="lya",
        help="Region to compute deltas in",
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

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
