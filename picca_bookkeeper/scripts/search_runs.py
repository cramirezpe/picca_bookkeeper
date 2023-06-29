""" Script to search bookkeeper runs and print
differences in tables."""

from pathlib import Path
import argparse
from picca_bookkeeper.bookkeeper_searcher import get_bookkeeper_differences


def main(args=None):
    if args is None:
        args = get_args()

    if args.delta:
        get_bookkeeper_differences(
            locations=args.directory,
            analysis_type="delta",
            remove_identical=not args.keep_identical,
            transpose=args.transpose,
            sort_by_value=args.sort_by_value,
        )

    if args.correlation:
        get_bookkeeper_differences(
            locations=args.directory,
            analysis_type="correlation",
            remove_identical=not args.keep_identical,
            transpose=args.transpose,
            sort_by_value=args.sort_by_value,
        )

    if args.fit:
        get_bookkeeper_differences(
            locations=args.directory,
            analysis_type="fit",
            remove_identical=not args.keep_identical,
            transpose=args.transpose,
            sort_by_value=args.sort_by_value,
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=Path, nargs="+", help="Path where to search for bookkeepers."
    )

    parser.add_argument(
        "--delta",
        action="store_true",
        help="Search for differences in delta extraction.",
    )

    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Search for differences in correlations.",
    )

    parser.add_argument(
        "--fit", action="store_true", help="Search for differences in fits."
    )

    parser.add_argument(
        "--keep-identical",
        action="store_true",
        help="Keep tables where all runs have the same values.",
    )

    parser.add_argument(
        "--transpose", action="store_true", help="Tranpose output tables."
    )

    parser.add_argument(
        "--sort-by-value", action="store_true", help="Sort by value not by name."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
