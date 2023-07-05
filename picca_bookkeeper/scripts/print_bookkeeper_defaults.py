"""Simple script to print default values"""

from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper import resources
from importlib_resources import files

import yaml


def main():
    defaults = yaml.safe_load(
        files(resources).joinpath("defaults.yaml").read_text()
    )

    print(DictUtils.print_dict(defaults))


if __name__ == "__main__":
    main()