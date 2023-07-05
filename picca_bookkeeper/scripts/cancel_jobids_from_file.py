"""Simple script to cancel all jobs in a jobid log file"""

import argparse
from pathlib import Path
from subprocess import run


def main(args=None):
    if args is None:
        args = get_args()

    with open(args.file, "r") as file:
        lines = file.readlines()

    for line in lines:
        jobid = line.split(" ")[-1].split("\n")[0]
        run(["scancel", str(jobid)])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
