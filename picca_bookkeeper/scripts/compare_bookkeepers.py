"""Script to fully compare two bookkeeper's configurations"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.scripts.fix_bookkeeper_links import strCyan, strRed

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

    bookkeeper1 = Bookkeeper(args.bookkeeper1_config)
    bookkeeper2 = Bookkeeper(args.bookkeeper2_config)

    if bookkeeper1.paths.healpix_data != bookkeeper2.paths.healpix_data:
        print(
            "Different healpix data:"
            + strRed(f"\n\t-{bookkeeper1.paths.healpix_data}")
            + strCyan(f"\n\t+{bookkeeper2.paths.healpix_data}\n")
        )

    if bookkeeper1.paths.catalog != bookkeeper2.paths.catalog:
        print(
            "Different QSO catalog:"
            + strRed(f"\n\t-{bookkeeper1.paths.catalog}")
            + strCyan(f"\n\t+{bookkeeper2.paths.catalog}\n")
        )

    if bookkeeper1.paths.catalog_dla != bookkeeper2.paths.catalog_dla:
        print(
            "Different DLA catalog:"
            + strRed(f"\n\t-{bookkeeper1.paths.catalog_dla}")
            + strCyan(f"\n\t+{bookkeeper2.paths.catalog_dla}\n")
        )

    if bookkeeper1.paths.catalog_bal != bookkeeper2.paths.catalog_bal:
        print(
            "Different BAL catalog:"
            + strRed(f"\n\t-{bookkeeper1.paths.catalog_bal}")
            + strCyan(f"\n\t+{bookkeeper2.paths.catalog_bal}\n")
        )

    tracer_catalogs_1 = (
        bookkeeper1.config.get("correlations", dict())
        .get("tracer catalogs", dict())
        .copy()
    )
    tracer_catalogs_2 = (
        bookkeeper2.config.get("correlations", dict())
        .get("tracer catalogs", dict())
        .copy()
    )

    for tracer in set(tracer_catalogs_1.keys()).union(tracer_catalogs_2.keys()):
        cat_1 = tracer_catalogs_1.get(tracer, bookkeeper1.paths.catalog)
        cat_2 = tracer_catalogs_2.get(tracer, bookkeeper2.paths.catalog)

        if cat_1 != cat_2:
            print(
                "Different tracer catalog:"
                + strRed(f"\n\t-{cat_1}")
                + strCyan(f"\n\t+{cat_2}\n")
            )

    ini_files = list(
        (bookkeeper2.paths.delta_extraction_path / "configs").glob("*.ini")
    )
    ini_files_base = [
        (bookkeeper1.paths.delta_extraction_path / "configs") / x.name
        for x in ini_files
    ]

    script_files = list((bookkeeper2.paths.correlations_path / "scripts").glob("*.sh"))
    script_files_base = [
        (bookkeeper1.paths.correlations_path / "scripts") / x.name for x in script_files
    ]

    fit_files = list((bookkeeper2.paths.fits_path / "configs").glob("*.ini"))
    fit_files_base = [
        (bookkeeper1.paths.fits_path / "configs") / x.name for x in fit_files
    ]

    config_files = ini_files + script_files + fit_files
    config_files_base = ini_files_base + script_files_base + fit_files_base

    logger.info("Number of config files: %s", len(config_files))

    import difflib

    for config_file, config_file_base in zip(config_files, config_files_base):
        if config_file_base.is_file():
            config_text = config_file_base.read_text()
        else:
            config_text = ""

        base_text = replace_strings(
            config_text,
            bookkeeper1,
        )

        config_text = replace_strings(
            config_file.read_text(),
            bookkeeper2,
        )

        # command = rf'diff <(echo "{base_text}") <(echo "{config_text}") --color --palette="ad=1;3;38;5;135:de=1;3;38;5;9"'
        # run(
        #     command,
        #     shell=True,
        #     executable='/bin/bash'
        # )

        diff_lines = list(
            difflib.unified_diff(
                base_text.split("\n"),
                config_text.split("\n"),
                fromfile="bookkeeper1",
                tofile="bookkeeper2",
                lineterm="",
                n=0,
            )
        )

        if len(diff_lines) > 0:
            print(config_file.name)
            for line in diff_lines:
                for prefix in ("---", "+++"):
                    if line.startswith(prefix):
                        break
                else:
                    if line.startswith("-"):
                        print("\t" + strRed(line))
                    elif line.startswith("+"):
                        print("\t" + strCyan(line))
                    else:
                        print("\t" + line)

            print("\n")


def replace_strings(text: str, bookkeeper: Bookkeeper) -> str:
    """Replaces strings for catalogs and bookkeeper paths so they don't
    show up in the diff
    """
    originals = [
        str(bookkeeper.paths.fits_path),
        str(bookkeeper.paths.correlations_path),
        str(bookkeeper.paths.delta_extraction_path),
        str(bookkeeper.paths.catalog),
        str(bookkeeper.paths.catalog_dla),
        str(bookkeeper.paths.catalog_bal),
        r".*ini files =.*",
        r".*zeff =.*",
        "\(",
        "\)",
        "--",
    ]

    replacements = [
        "fits-path",
        "correlations-path",
        "deltas-path",
        "catalog",
        "catalog-dla",
        "catalog-bal",
        # "catalog-tracer",
        "ini files",
        "zeff",
        "",
        "",
        "'\n'",
    ]

    for original, replacement in zip(originals, replacements):
        text = re.sub(original, replacement, text)

    for tracer, path in (
        bookkeeper.config.get("correlations", dict())
        .get("tracer catalogs", dict())
        .values()
    ):
        text = re.sub(path, tracer, text)

    return text


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bookkeeper1_config",
        type=Path,
    )

    parser.add_argument(
        "bookkeeper2_config",
        type=Path,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
