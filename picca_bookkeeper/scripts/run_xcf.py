""" Script to run picca_xcf and export given a bookkeeper config file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import DummyTasker

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

    logger.info(f"Adding cross-correlation: {args.absorber}{args.region}_{args.tracer}")

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        overwrite_config=args.overwrite_config,
        read_mode=False,
    )

    xcf = bookkeeper.get_xcf_tasker(
        region=args.region,
        absorber=args.absorber,
        tracer=args.tracer,
        debug=args.debug,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )
    xcf.write_job()
    if not args.only_write:
        xcf.send_job()

        if not isinstance(xcf, DummyTasker):
            logger.info(
                "Sent cross-correlation "
                f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                f"{xcf.jobid}"
            )
    logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get("no distortion", False):
        logger.info(
            f"Adding distortion matrix: {args.absorber}{args.region}_{args.tracer}"
        )
        xdmat = bookkeeper.get_xdmat_tasker(
            region=args.region,
            absorber=args.absorber,
            tracer=args.tracer,
            debug=args.debug,
            system=args.system,
            wait_for=args.wait_for,
            overwrite=args.overwrite,
            skip_sent=args.skip_sent,
        )
        xdmat.write_job()
        if not args.only_write:
            xdmat.send_job()

            if not isinstance(xdmat, DummyTasker):
                logger.info(
                    "Sent distortion matrix "
                    f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                    f"{xdmat.jobid}"
                )
        logger.info("Done.\n")

    if not bookkeeper.config.get("fits", dict()).get(
        "no metals", False
    ) and not bookkeeper.config.get("fits", dict()).get("vega metals", False):
        # Compute metals if metals should be included and metals are not going
        # to be computed by vega.

        logger.info(f"Adding metal matrix: {args.absorber}{args.region}_{args.tracer}")
        metal = bookkeeper.get_xmetal_tasker(
            region=args.region,
            absorber=args.absorber,
            tracer=args.tracer,
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
                    f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                    f"{metal.jobid}"
                )
        logger.info("Done.\n")

    logger.info(f"Adding export: {args.absorber}{args.region}_{args.tracer}")
    xcf_exp = bookkeeper.get_xcf_exp_tasker(
        region=args.region,
        absorber=args.absorber,
        tracer=args.tracer,
        system=args.system,
        wait_for=args.wait_for,
        overwrite=args.overwrite,
        skip_sent=args.skip_sent,
    )

    xcf_exp.write_job()
    if not args.only_write:
        xcf_exp.send_job()

        if not isinstance(xcf_exp, DummyTasker):
            logger.info(
                "Sent export "
                f"{args.absorber}{args.region}_{args.tracer}:\n\t"
                f"{xcf_exp.jobid}"
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
        "--region",
        type=str,
        choices=["lya", "lyb"],
        default="lya",
        help="Region to compute correlation in",
    )

    parser.add_argument(
        "--tracer", type=str, default="qso", help="Tracer to use for the correlation."
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
        "--skip-sent", action="store_true", help="Skip runs that were already sent."
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
