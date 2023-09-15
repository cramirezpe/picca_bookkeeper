"""Build a full bookkeeper config provided a bookkeeper inside a run. That is
a bookkeeper that can be rerun completely in a different location."""

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Optional
logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    bookkeeper = Bookkeeper(args.bookkeeper_config)
    if not Path(args.save_path).is_file() or args.overwrite:
        bookkeeper.write_bookkeeper(bookkeeper.config, args.save_path)
    else:
        raise FileExistsError("Output file already exists.", str(args.save_path))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuilds a full yaml configuration file from its location in a bookkeeper analysis tree so it can be rerun entirely."
    )
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use."
    )

    parser.add_argument(
        "save_path",
        type=Path,
        help="Output path where to save the full bookkeeper config.",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite output path."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
