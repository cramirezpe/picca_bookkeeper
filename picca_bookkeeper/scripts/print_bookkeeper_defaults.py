"""Simple script to print default values"""

import argparse

import yaml
from importlib_resources import files

from picca_bookkeeper import resources
from picca_bookkeeper.dict_utils import DictUtils


def main(args=None):
    if args is None:
        args = get_args()

    defaults = yaml.safe_load(
        files(resources).joinpath(f"default_configs/{args.prefix}.yaml").read_text()
    )

    print(DictUtils.print_dict(defaults))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Prefix to print defaults of.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
