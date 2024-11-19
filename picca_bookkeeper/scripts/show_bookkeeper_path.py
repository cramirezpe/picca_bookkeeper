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

    paths = []
    for bookkeeper_config in args.bookkeeper_configs:
        bookkeeper = Bookkeeper(bookkeeper_config)

        if bookkeeper.paths.fit_out_fname().is_file():
            paths.append(bookkeeper.paths.fit_out_fname())
        elif bookkeeper.paths.fits_path.is_dir():
            paths.append(bookkeeper.paths.fits_path)
        else:
            paths.append(bookkeeper.paths.run_path)

    if args.pretty:
        print(
            "\n\n".join(
                f"{config}:\n\t{path}"
                for config, path in zip(args.bookkeeper_configs, paths)
            )
        )
    else:
        [print(x) for x in paths]


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bookkeeper_configs",
        type=Path,
        nargs="+",
        help="Bookkeeper configuration file.",
    )

    parser.add_argument(
        "--pretty", action="store_true", help="Print file name and bookkeeper paths."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
