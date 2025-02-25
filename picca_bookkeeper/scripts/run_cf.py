""" Script to run picca_cf and export
given a bookkeeper config file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import DummyTasker

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info(
        f"Adding auto-correlation: "
        f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
    )
    cf = bookkeeper.get_cf_tasker(
        region=args.region,
        region2=args.region2,
        absorber=args.absorber,
        absorber2=args.absorber2,
        system=args.system,
        debug=args.debug,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    cf.write_job()
    if not args.only_write:
        cf.send_job()
        if not isinstance(cf, DummyTasker):
            logger.info(
                "Sent auto-correlation "
                f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                f"{cf.jobid}"
            )
    logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get("no distortion", False):
        logger.info(
            f"Adding distortion matrix: "
            f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
        )
        dmat = bookkeeper.get_dmat_tasker(
            region=args.region,
            region2=args.region2,
            absorber=args.absorber,
            absorber2=args.absorber2,
            system=args.system,
            wait_for=args.wait_for,
            debug=args.debug,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        dmat.write_job()
        if not args.only_write:
            dmat.send_job()

            if not isinstance(dmat, DummyTasker):
                logger.info(
                    "Sent distortion matrix "
                    f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                    f"{dmat.jobid}"
                )
        logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get(
        "no metals", False
    ) and not bookkeeper.config.get("fits", dict()).get("vega metals", False):
        # Compute metals if metals should be included and metals are not going
        # to be computed by vega.

        logger.info(
            f"Adding metal matrix: "
            f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
        )
        metal = bookkeeper.get_metal_tasker(
            region=args.region,
            region2=args.region2,
            absorber=args.absorber,
            absorber2=args.absorber2,
            system=args.system,
            debug=args.debug,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        metal.write_job()
        if not args.only_write:
            metal.send_job()

            if not isinstance(metal, DummyTasker):
                logger.info(
                    "Sent metal matrix "
                    f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                    f"{metal.jobid}"
                )
        logger.info("Done.\n")

    logger.info(
        f"Adding export: "
        f"{args.absorber}{args.region}_{args.absorber2}{args.region2}"
    )

    cf_exp = bookkeeper.get_cf_exp_tasker(
        region=args.region,
        region2=args.region2,
        absorber=args.absorber,
        absorber2=args.absorber2,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    cf_exp.write_job()
    if not args.only_write:
        cf_exp.send_job()

        if not isinstance(cf_exp, DummyTasker):
            logger.info(
                "Sent export "
                f"{args.absorber}{args.region}_{args.absorber2}{args.region2}:\n\t"
                f"{cf_exp.jobid}"
            )
    logger.info("Done.\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
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
        "--skip-sent", action="store_true", help="Skip runs that were already sent."
    )

    parser.add_argument(
        "--region",
        type=str,
        default="lya",
        help="Region to compute correlation in",
    )

    parser.add_argument(
        "--region2",
        type=str,
        default=None,
        help="Second region (for cross-correlations between forests",
    )

    parser.add_argument(
        "--absorber", type=str, default="lya", help="Absorber to use for correlations"
    )

    parser.add_argument(
        "--absorber2",
        type=str,
        default=None,
        help="Second absorber (for cross-correlations between forests)",
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
