"""Simple script to cancel all jobs in a jobid log file"""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    with open(args.file, "r") as file:
        lines = file.readlines()

    for line in lines:
        jobid = line.split(" ")[-1].split("\n")[0]
        run(["scancel", str(jobid)])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
