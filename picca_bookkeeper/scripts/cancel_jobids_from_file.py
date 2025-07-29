"""
cancel_jobids_from_file.py

Script to batch-cancel SLURM jobs by reading job IDs from a file.

Functionality:
--------------
This script reads a plain text file containing SLURM job IDs (one per line or
as the last field on each line) and cancels each job using the `scancel`
command-line utility.

Usage:
------
Run from the command line as:
    python cancel_jobids_from_file.py <jobid_log_file>

Arguments:
    <jobid_log_file>: Path to a file containing SLURM job IDs, typically generated
                      by other scripts or task runners in the picca_bookkeeper
                      workflow. Each line should contain a job ID as its last
                      whitespace-separated field.

Example:
--------
    python cancel_jobids_from_file.py /path/to/logs/jobids.log

Notes:
------
    - The script does not perform complex error checking on the log file format;
      it assumes job IDs are the last field on each line.
    - It is up to the user to ensure the log file contains valid job IDs.
    - Requires permissions to cancel the listed jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Cancels SLURM jobs listed in a file by calling `scancel` on each job ID.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]): Parsed command-line arguments.
            If None, arguments are obtained by calling get_args().
    """
    if args is None:
        args = get_args()

    with open(args.file, "r") as file:
        lines = file.readlines()

    for line in lines:
        jobid = line.split(" ")[-1].split("\n")[0]
        run(["scancel", str(jobid)])


def get_args() -> argparse.Namespace:
    """
    Parses command-line arguments to get the path to the job ID log file.

    Returns:
    --------
        argparse.Namespace: Namespace containing:
            - file (Path): Path to a text file with SLURM job IDs to cancel.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
