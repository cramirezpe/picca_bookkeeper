"""Simple script to print default values"""

from __future__ import annotations

from importlib_resources import files

from picca_bookkeeper import resources


def main() -> None:
    print(files(resources).joinpath("example_config.yaml").read_text())


if __name__ == "__main__":
    main()
