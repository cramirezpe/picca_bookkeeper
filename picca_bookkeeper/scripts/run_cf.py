""" Script to run picca_cf and export
given a bookkeeper config file."""
from pathlib import Path
import argparse
from picca_bookkeeper.bookkeeper import Bookkeeper


def main(args=None):
    if args is None:
        args = get_args()

    bookkeeper = Bookkeeper(args.bookkeeper_config, overwrite_config=args.overwrite_config)

    cf = bookkeeper.get_cf_tasker(
        region=args.region,
        region2=args.region2,
        system=None,
        debug=args.debug,
        wait_for=args.wait_for,
    )
    cf.write_job()
    if not args.only_write:
        cf.send_job()
        wait_for = cf
    else:
        wait_for = None

    if not args.no_dmat:
        dmat = bookkeeper.get_dmat_tasker(
            region=args.region,
            region2=args.region2,
            wait_for=args.wait_for,
            debug=args.debug,
        )
        dmat.write_job()
        if not args.only_write:
            dmat.send_job()
            wait_for = [cf, dmat]

    if not args.no_metal:
        metal = bookkeeper.get_metal_tasker(
            region=args.region,
            region2=args.region2,
            system=None,
            debug=args.debug,
            wait_for=args.wait_for,
        )
        metal.write_job()
        if not args.only_write:
            metal.send_job()

    cf_exp = bookkeeper.get_cf_exp_tasker(
        region=args.region,
        region2=args.region2,
        system=None,
        wait_for=wait_for,
        no_dmat=args.no_dmat,
    )

    cf_exp.write_job()
    if not args.only_write:
        cf_exp.send_job()
        print(cf_exp.jobid)
        return cf_exp.jobid
    else:
        return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--overwrite_config",
        action="store_true",
        help="Force overwrite bookkeeper config."
    )

    parser.add_argument(
        "--region",
        type=str,
        choices=["lya", "lyb"],
        default="lya",
        help="Region to compute correlation in",
    )

    parser.add_argument(
        "--region2",
        type=str,
        choices=["lya", "lyb"],
        default=None,
        help="Second region (for cross-correlations between forests",
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
        "--only-write",
        action="store_true",
        help="Only write scripts, not send them."
    )

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
