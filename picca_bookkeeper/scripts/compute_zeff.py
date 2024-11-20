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

    file_names = "\n\t".join(map(str, args.export_files))

    logger.info(f"Files:\n\t{file_names}")

    zeff = compute_zeff(
        export_files=args.export_files,
        rmins=args.rmins,
        rmaxs=args.rmaxs,
    )

    print("Computed zeff: ", zeff)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "export_files",
        type=Path,
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--rmins",
        type=float,
        nargs="+",
        default=0,
        help="Minimum distances.",
    )

    parser.add_argument(
        "--rmaxs",
        type=float,
        nargs="+",
        default=300,
        help="Maximum distances.",
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
