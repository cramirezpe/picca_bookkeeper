""" Script to run picca_xcf and export given a bookkeeper config file."""
from pathlib import Path
import argparse
from picca_bookkeeper.bookkeeper import Bookkeeper


def main(args=None):
    if args is None:
        args = get_args()
    bookkeeper = Bookkeeper(args.bookkeeper_config, overwrite_config=args.overwrite_config)

    xcf = bookkeeper.get_xcf_tasker(
        region=args.region,
        debug=args.debug,
        wait_for=args.wait_for,
    )
    xcf.write_job()
    xcf.send_job()
    wait_for = xcf

    if not args.no_dmat:
        xdmat = bookkeeper.get_xdmat_tasker(
            region=args.region,
            debug=args.debug,
            wait_for=args.wait_for,
        )
        xdmat.write_job()
        xdmat.send_job()
        wait_for = [xcf, xdmat]

    if not args.no_metal:
        metal = bookkeeper.get_xmetal_tasker(
            region=args.region,
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
        )
        metal.write_job()
        metal.send_job()

    xcf_exp = bookkeeper.get_xcf_exp_tasker(
        region=args.region,
        wait_for=wait_for,
        no_dmat=args.no_dmat,
    )

    xcf_exp.write_job()
    xcf_exp.send_job()

    print(xcf_exp.jobid)
    return xcf_exp.jobid


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
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config."
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

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
