"""
show_bookkeeper_path.py
-----------------------

Script to print out the fit output file or main results directory for a
picca_bookkeeper run.


Functionality:
---------------
    - Accepts one or more bookkeeper configuration YAML files as arguments.
    - For each config file:
        - Instantiates a `Bookkeeper` object to interpret the configuration.
        - Checks for the existence of the main fit output file (`fit_output.fits`)
          via `Bookkeeper.paths.fit_out_fname()`.
            - If present, prints this file's path.
            - If not, checks for the existence of the fits directory
              (`Bookkeeper.paths.fits_path`).
            - If neither exists, falls back to printing the main run directory
              (`Bookkeeper.paths.run_path`).

Usage:
------
Example command-line usage:
    python -m picca_bookkeeper.scripts.show_bookkeeper_path <bookkeeper_config1.yaml>
    [<bookkeeper_config2.yaml> ...] [--pretty]

    # Example:
    python -m picca_bookkeeper.scripts.show_bookkeeper_path configs/bookkeeper_config.yaml
    --pretty

Example programmatic usage:
    from picca_bookkeeper.scripts.show_bookkeeper_path import main
    main()  # Uses sys.argv

    # Or with custom arguments:
    import argparse
    args = argparse.Namespace(
        bookkeeper_configs=[Path("my_config.yaml")],
        pretty=True,
    )
    main(args)

See Also:
----------
    - `Bookkeeper` and `PathBuilder` in `picca_bookkeeper.bookkeeper`
    - Documentation in the repo's README for file structure and configuration
      details

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Print the path to the main output of one or more picca_bookkeeper runs.

    For each config file provided, this function:
        1. Instantiates a `Bookkeeper` using the given YAML config.
        2. Checks for the presence of the main fit output file
           (`fit_output.fits` via `bookkeeper.paths.fit_out_fname()`).
        3. If that file does not exist, checks for the `fits_path` directory.
        4. If neither exists, prints the base run directory.

    Output format:
    --------------
        - Plain path list (default).
        - Pretty format showing the config file followed by the output path
          (if `--pretty` is specified).

    Arguments:
    ----------
        args (argparse.Namespace, optional): Parsed command-line arguments.
            If not provided, arguments are parsed from `sys.argv` using `getArgs()`.

    Raises:
    -------
        FileNotFoundError: If a given configuration file cannot be found or read.

    Example:
    --------
        >>> main()  # uses sys.argv
        >>> args = argparse.Namespace(bookkeeper_configs=[Path("my_config.yaml")],
                                      pretty=True)
        >>> main(args)
    """
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
    """
    Parse command-line arguments for the script.

    Recognized arguments:
    ---------------------
        - One or more paths to Bookkeeper YAML config files.
        - Optional `--pretty` flag to format the output more readably.

    Returns:
    --------
        argparse.Namespace: Namespace containing:
            - bookkeeper_configs (List[Path]): Paths to config files.
            - pretty (bool): Whether to print paths with formatting.

    Example:
    --------
        >>> args = getArgs()
        >>> print(args.bookkeeper_configs)
        [PosixPath('config1.yaml'), PosixPath('config2.yaml')]
    """
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
