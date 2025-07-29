"""
compute_zeff.py
---------------

Script to easily compute zeff for a given set of files.

This script computes the effective redshift (zeff) for one or more provided
correlation export files produced by picca or related pipelines. It utilizes
the `compute_zeff` utility from the `picca_bookkeeper.utils` module, which
calculates a weighted average of the redshift values in the input files over
specified distance ranges.

Functionality:
--------------
    - Reads one or more export files containing correlation data
      (typically FITS format).
    - For each file, computes zeff as a weighted average of redshifts in a given
      radial range (using the inverse diagonal of the covariance matrix as weights).
    - Logs the files being processed and prints the computed zeff.

Arguments:
----------
    - export_files (positional): Paths to one or more export files (FITS) to
                          process.
    - --rmins (optional): Minimum radial distances (float or list of floats).
                          Defaults to 0.
    - --rmaxs (optional): Maximum radial distances (float or list of floats).
                          Defaults to 300.
    - --log-level (optional): Logging level (default: INFO).

Example usage:
--------------
From the command line:
    python scripts/compute_zeff.py path/to/export1.fits path/to/export2.fits
    --rmins 0 10 --rmaxs 300 200

From within Python:
    from picca_bookkeeper.scripts.compute_zeff import main
    main()

Notes:
------
- Input files must be in the expected FITS format, with "RP", "RT", "Z", and
  "CO" columns in the appropriate extensions.
- For advanced integration (e.g., with SLURM or batch processing), use the
  Bookkeeper API, which wraps and schedules this script as a job.

"""

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
    """
    Main entry point for computing zeff from one or more correlation export files.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]): Parsed command-line arguments.
            If None, arguments are parsed via getArgs().

    Behavior:
    ----------
        - Parses input file paths, radial distance ranges, and log level.
        - Configures logging.
        - Calls `compute_zeff()` using provided or default arguments.
        - Prints the resulting effective redshift (zeff) to stdout.

    Raises:
    --------
        ValueError: If the compute_zeff function encounters invalid input files.
    """
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
    """
    Parses command-line arguments for computing zeff.

    Returns:
    --------
        argparse.Namespace: Namespace with the following attributes:
            - export_files (List[Path]): List of FITS files to compute zeff from.
            - rmins (List[float]): List of minimum radial distances
              (default: [0.0]).
            - rmaxs (List[float]): List of maximum radial distances
              (default: [300.0]).
            - log_level (str): Logging verbosity level (default: "INFO").

    Notes:
    ------
        - Accepts multiple values for `--rmins` and `--rmaxs`, matched to
          input files.
        - If fewer rmin/rmax values are provided than files, they are broadcast
          if possible.
        - Logging level must be one of: CRITICAL, ERROR, WARNING, INFO,
          DEBUG, NOTSET.
    """
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
