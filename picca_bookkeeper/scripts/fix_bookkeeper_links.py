"""
Script to modify all the links/paths inside the bookkeeper to match
the current path structure.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

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

    old_dirs = set(args.replace_dirs)

    if not args.only_replace_parsed_dirs:
        yaml_files = list(args.source_dir.glob("**/bookkeeper_config.yaml"))
        for file in yaml_files:
            config = yaml.safe_load(file.read_text())

            old_dirs.add(str(Path(config["data"]["bookkeeper dir"]).parent))

    yaml_files = list(args.source_dir.glob("**/*.yaml"))

    file_string = "\n\t".join(map(str, yaml_files))
    print(f"Config files to be modified:\n\t{file_string}")

    slurm_files = list(args.source_dir.glob("**/*.sh"))
    file_string = "\n\t".join(map(str, slurm_files))
    print(f"Slurm files to be modified:\n\t{file_string}")

    ini_files = list(args.source_dir.glob("**/*.ini"))
    file_string = "\n\t".join(map(str, ini_files))
    print(f"Ini files to be modified:\n\t{file_string}")

    links = list(args.source_dir.glob("**/*.fits"))
    links += list(args.source_dir.glob("**/*.fits.gz"))
    links = [x for x in links if x.is_symlink()]

    file_string = "\n\t".join(map(str, links))
    print(f"Symlink files:\n\t{file_string}")

    # Remove ending / because if there are 2 issues might happen
    updated_old_dirs = []
    for old_dir in old_dirs:
        while old_dir[-1] == "/":
            old_dir = old_dir[:-1]
        updated_old_dirs.append(old_dir)
    old_dirs = set(updated_old_dirs)

    new_dir = args.new_dir
    while new_dir[-1] == "/":
        new_dir = new_dir[:-1]
    print(f"New directory:\n\t{new_dir}")

    print("\nThe following substitutions will be made:")
    for dir_ in old_dirs:
        print(strRed(dir_), "->", strCyan(new_dir))

    answer = input(
        "This action cannot be undone, are you really sure? Please write 'Yes, I am' to proceed:\n"
    )

    if answer != "Yes, I am":
        print("Exiting")
        exit()

    logger.info("Rewriting config files.")
    for yaml_file in yaml_files:
        for dir_ in old_dirs:
            with open(yaml_file, "r") as file:
                lines = file.readlines()

            with open(yaml_file, "w") as file:
                for line in lines:
                    file.write(re.sub(dir_, new_dir, line))

    logger.info("Rewriting slurm files.")
    for slurm_file in slurm_files:
        for dir_ in old_dirs:
            with open(slurm_file, "r") as file:
                lines = file.readlines()

            with open(slurm_file, "w") as file:
                for line in lines:
                    file.write(re.sub(dir_, new_dir, line))

    logger.info("Rewriting ini files.")
    for ini_file in ini_files:
        for dir_ in old_dirs:
            with open(ini_file, "r") as file:
                lines = file.readlines()

            with open(ini_file, "w") as file:
                for line in lines:
                    file.write(re.sub(dir_, new_dir, line))

    logger.info("Modifying symbolic links.")
    for link in links:
        for dir_ in old_dirs:
            if dir_ in str(link.readlink().absolute()):
                new_link = Path(re.sub(dir_, new_dir, str(link.readlink().absolute())))

                link.unlink()
                link.symlink_to(new_link)


def strRed(skk: str) -> str:
    return "\033[91m {}\033[00m".format(skk)


def strCyan(skk: str) -> str:
    return "\033[34m {}\033[00m".format(skk)


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Script to modify all the links/paths inside a bookkeeper structure to match the current path structure where the files are located in. The internal structure of the files shouldn't have been modified.

This script will modify the following things under the given bookkeeper-dir:
    - All .yaml files under the given bookkeeper-dir. To substitute the "bookkeeper dir" specified in the files for the real one.
    - All .sl files, from the collected "bookkeeper dir" in the previous step into the real one.
    - All symlinks, again from the collected "bookkeeper dir" into the real one.
""",
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory where to search for bookkeeper_config.yaml files "
        "to start the process.",
    )

    parser.add_argument(
        "--new-dir",
        default=None,
        type=str,
        help="New bookkeeper dir that should be written in all files.",
    )

    parser.add_argument(
        "--replace-dirs",
        default=[],
        type=str,
        nargs="+",
        help="Add extra string to be substituted for the new-dir.",
    )

    parser.add_argument(
        "--only-replace-parsed-dirs",
        action="store_true",
        help="Only replace directories that have been parsed (not the ones in current structure).",
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
