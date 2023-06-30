"""Simple script to print default values"""

from picca_bookkeeper import resources
from importlib_resources import files



def main():
    print(files(resources).joinpath("example_config.yaml").read_text())

if __name__ == "__main__":
    main()
