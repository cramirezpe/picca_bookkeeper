"""
print_bookkeeper_defaults.py
-----------------------------

Prints default configuration values for picca_bookkeeper components.

This script loads and prints the contents of a default YAML configuration file
corresponding to a specified prefix.

Functionality:
--------------
    - Loads a YAML file located in the package's resources/default_configs
      directory, with the filename determined by the given prefix.
    - Parses the YAML file into a dictionary.
    - Prints the dictionary in a readable format using the DictUtils utility class.

Usage:
------
This script can be run from the command line:
    python -m picca_bookkeeper.scripts.print_bookkeeper_defaults <prefix>

Where <prefix> corresponds to the configuration you want to inspect
(e.g., "pipeline", "database", etc.). The script will look for a YAML file
named <prefix>.yaml inside the default_configs subdirectory of the resources package.

Example:
--------
    python -m picca_bookkeeper.scripts.print_bookkeeper_defaults pipeline

Interactions:
-------------
    - Relies on the resources module for locating the configuration files.
    - Uses DictUtils.print_dict to format the output.
    - The available prefixes and their corresponding YAML files are managed
      within the picca_bookkeeper/resources/default_configs directory, so
      updates to those files or additions/removals will directly affect the
      script's output.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import yaml
from importlib_resources import files

from picca_bookkeeper import resources
from picca_bookkeeper.dict_utils import DictUtils

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Load and print the default configuration for a given prefix.

    This function reads a YAML file located in the `resources/default_configs`
    directory, parses it into a dictionary, and prints its contents using
    `DictUtils.print_dict()`.

    Arguments:
    ----------
        args (argparse.Namespace, optional): Parsed command-line arguments. If None,
            `get_args()` is called to parse them from `sys.argv`.

    Raises:
    -------
        - FileNotFoundError: If the YAML file corresponding to the prefix does not exist.
        - yaml.YAMLError: If the YAML file cannot be parsed.
    """
    if args is None:
        args = get_args()

    defaults = yaml.safe_load(
        files(resources).joinpath(
            f"default_configs/{args.prefix}.yaml").read_text()
    )

    print(DictUtils.print_dict(defaults))


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
    --------
        argparse.Namespace: Parsed arguments with the following attribute:
            - prefix (str): Name of the configuration prefix to load and print.

    Example:
    --------
        Namespace(prefix='pipeline')
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str,
                        help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
