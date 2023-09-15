"""Simple script to print default values"""

import argparse

import yaml
from importlib_resources import files

from picca_bookkeeper import resources
from picca_bookkeeper.dict_utils import DictUtils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

def main(args: Optional[argparse.Namespace]=None) -> None:
    if args is None:
        args = get_args()

    defaults = yaml.safe_load(
        files(resources).joinpath(f"default_configs/{args.prefix}.yaml").read_text()
    )

    print(DictUtils.print_dict(defaults))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
