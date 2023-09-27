"""Simple script to print default values"""

from __future__ import annotations

from typing import TYPE_CHECKING

from importlib_resources import files

from picca_bookkeeper import resources

if TYPE_CHECKING:
    from typing import Optional


def main() -> None:
    print(files(resources).joinpath("example_config.yaml").read_text())


if __name__ == "__main__":
    main()
