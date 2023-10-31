"""Script to print bookkeeper path."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Optional

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = getArgs()

    bookkeeper = Bookkeeper(args.bookkeeper_config)

    if bookkeeper.fits is not None:
        return bookkeeper.paths.fits_path
    elif bookkeeper.correlations is not None:
        return bookkeeper.paths.correlations_path
    else:
        return bookkeeper.paths.run_path

def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bookkeeper_config",
        type=Path,
        help="Bookkeeper configuration file."
    )


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
