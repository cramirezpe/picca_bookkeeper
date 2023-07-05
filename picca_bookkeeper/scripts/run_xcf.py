""" Script to run picca_xcf and export given a bookkeeper config file."""
import argparse
import logging
import sys
from pathlib import Path

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import Tasker

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

    logger.info(f"Adding cross-correlation: {args.absorber}{args.region}_qso")

    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )

    xcf = bookkeeper.get_xcf_tasker(
        region=args.region,
        debug=args.debug,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
    )
    xcf.write_job()
    if not args.only_write:
        xcf.send_job()
        wait_for = xcf
    else:
        wait_for = None

    if not args.no_dmat:
        xdmat = bookkeeper.get_xdmat_tasker(
            region=args.region,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
        )
        xdmat.write_job()
        if not args.only_write:
            xdmat.send_job()
            wait_for = [xcf, xdmat]

    if not args.no_metal:
        metal = bookkeeper.get_xmetal_tasker(
            region=args.region,
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
        )
        # Only run metal if it has not been copied
        # only if metal is a tasker instance
        if isinstance(metal, Tasker):
            metal.write_job()
            if not args.only_write:
                metal.send_job()
                metal_jobid = metal.jobid
            else:
                metal_jobid = None
        else:
            metal_jobid = None
    else:
        metal_jobid = None

    xcf_exp = bookkeeper.get_xcf_exp_tasker(
        region=args.region,
        wait_for=wait_for,
        no_dmat=args.no_dmat,
        overwrite=args.overwrite,
    )

    xcf_exp.write_job()
    if not args.only_write:
        xcf_exp.send_job()
        print(xcf_exp.jobid)
        return [metal_jobid, xcf_exp.jobid]
    else:
        return


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
        help="Region to compute correlation in",
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
        "--no-dmat", action="store_true", help="Do not use distortion matrix."
    )

    parser.add_argument(
        "--no-metal", action="store_true", help="Do not compute metal distortion matrix"
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
