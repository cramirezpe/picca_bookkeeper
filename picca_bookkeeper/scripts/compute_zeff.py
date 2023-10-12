"""Script to easily compute zeff for a given set of files"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.utils import compute_zeff

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    auto_files = "\n\t".join(map(str, args.auto_correlations))
    cross_files = "\n\t".join(map(str, args.cross_correlations))

    logger.info(f"Auto-correlation files:\n\t{auto_files}")
    logger.info(f"Cross-correlation files:\n\t{cross_files}")

    zeff = compute_zeff(
        cf_files=args.auto_correlations,
        xcf_files=args.cross_correlations,
        rmin_cf=args.rmin_cf,
        rmax_cf=args.rmax_cf,
        rmin_xcf=args.rmin_xcf,
        rmax_xcf=args.rmax_xcf,
    )

    print("Computed zeff: ", zeff)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--auto-correlations",
        type=Path,
        nargs="+",
        default=[],
        help="auto-correlation export files.",
    )

    parser.add_argument(
        "--cross-correlations",
        type=Path,
        nargs="+",
        default=[],
        help="cross-correlation export files",
    )

    parser.add_argument(
        "--rmin-cf",
        type=float,
        default=10,
        help="Minimum distance for auto-correlations.",
    )

    parser.add_argument(
        "--rmin-xcf",
        type=float,
        default=10,
        help="Minimum distance for cross-correlations.",
    )

    parser.add_argument(
        "--rmax-cf",
        type=float,
        default=300,
        help="Maximum distance for auto-correlations.",
    )

    parser.add_argument(
        "--rmax-xcf",
        type=float,
        default=300,
        help="Maximum distance for auto-correlations.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
