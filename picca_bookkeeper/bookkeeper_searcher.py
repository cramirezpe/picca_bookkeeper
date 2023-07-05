"""Module to find bookkeepers in one base path and print differences"""

from pathlib import Path
from tabulate import tabulate
from typing import *
import yaml
import numpy as np

from picca_bookkeeper.dict_utils import DictUtils


def get_bookkeeper_differences(
    locations: Union[str, Path],
    analysis_type: str = "delta",
    remove_identical: bool = True,
    transpose: bool = False,
    sort_by_value: bool = False,
) -> None:
    assert analysis_type in ("delta", "correlation", "fit")
    if analysis_type == "delta":
        Config = DeltaConfigReader
    elif analysis_type == "correlation":
        Config = CorrelationConfigReader
    else:
        Config = FitConfigReader
    filename = Config.config_name

    if isinstance(locations, Path) or isinstance(locations, str):
        locations = [
            locations,
        ]

    analyses = []
    for config in locations:
        for file in config.glob(f"**/{filename}"):
            analyses.append(
                Config(
                    file,
                )
            )

    show_diffs(
        [analysis.config for analysis in analyses],
        name_keys=Config.first_els,
        name_values=np.array([analysis.first_els_values for analysis in analyses]),
        remove_identical=remove_identical,
        transpose=transpose,
        sort_by_value=sort_by_value,
    )


def show_diffs(
    configs: List[Dict],
    name_keys: List[str],
    name_values: List[str],
    remove_identical: bool = True,
    transpose: bool = False,
    sort_by_value: bool = False,
    depth: int = 1,
    title: str = "",
) -> None:
    keys = []
    dict_keys = []
    remove_keys = []
    for config in configs:
        if not isinstance(config, dict):
            config = dict()

        for key in config.keys():
            if key == "slurm args":
                remove_keys.append(key)
            elif isinstance(config[key], dict):
                # If there is a dict inside, perform the
                # show diff directly, although it has to
                # be performed outside the configs loop.
                dict_keys.append(key)
            else:
                keys.append(key)
    keys = np.unique(keys)
    dict_keys = np.unique(dict_keys)
    remove_keys = np.unique(remove_keys)

    keys = keys[~np.in1d(keys, remove_keys)]
    keys = keys[~np.in1d(keys, dict_keys)]

    for dict_key in dict_keys:
        for config in configs:
            if dict_key not in config:
                config[dict_key] = dict()
        show_diffs(
            configs=[config[dict_key] for config in configs],
            name_keys=name_keys,
            name_values=name_values,
            remove_identical=remove_identical,
            transpose=transpose,
            sort_by_value=sort_by_value,
            depth=depth + 1,
            title=title + "/" + dict_key,
        )

    rows = []
    for config in configs:
        if config is None:
            config = dict()
        row = []
        for key in keys:
            if key in config:
                row.append(config[key])
            else:
                row.append("")
        rows.append(row)
    rows = np.array(rows)

    if remove_identical:
        mask = ~np.all(rows == rows[0], axis=0)
        keys = keys[mask]
        new_rows = []
        for row in rows:
            new_rows.append(row[mask])

        rows = np.array(new_rows)
    else:
        mask = np.ones_like(keys, dtype=bool)
    if rows.size == 0:
        # No data in plot, then skip
        return

    rows = np.concatenate((name_values, rows), axis=1)
    keys = np.concatenate((name_keys, keys))

    ind = np.lexsort([rows[:, i] for i in range(len(name_keys))])
    if sort_by_value:
        ind = np.lexsort([rows[:, i] for i in range(len(name_keys), rows.shape[1])])
    rows = rows[ind]

    if title is not None:
        print(title)

    if not transpose:
        print(tabulate(rows, keys, tablefmt="pretty"))
    else:
        print(
            tabulate(
                np.concatenate((keys.reshape(1, -1), rows), axis=0).T,
                tablefmt="pretty",
            )
        )


class ConfigReader:
    """Class to read bookkeeper configuration files without loading the bookkeeper.
    Multiple bookkeeper configuration files can be then read easily.
    """

    config_name = "bookkeeper_config.yaml"

    def __init__(self, bookkeeper_path: Union[str, Path]):
        """
        Args:
            bookkeeper_path: Path to the bookkeeper or bookkeeper file.
        """
        self.path = Path(bookkeeper_path)
        if self.path.is_dir():
            self.path = self.path / "configs" / self.config_name

        # Read defaults
        self.read_defaults()

        # Read yaml file
        self.read_config()

        # Remove unneeded sections
        self.clean_config()

        # Saving fields with run names
        self.save_key_fields()

        self.data = None

    @property
    def name(self) -> str:
        return self.path.parent.parent.name

    def read_defaults(self):
        """Method to read bookkeeper defaults"""
        if self.defaults_file.is_file():
            with open(self.defaults_file) as file:
                self.defaults = yaml.safe_load(file)
        else:
            self.defaults = dict()

    def read_config(self):
        """Method to read a bookkeeper configuration yaml file"""
        with open(self.path) as file:
            self.config = yaml.safe_load(file)

        self.config = DictUtils.merge_dicts(
            self.defaults,
            DictUtils.remove_empty(self.config),
        )


class DeltaConfigReader(ConfigReader):
    first_els = np.array(["delta run"])

    def clean_config(self):
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["delta extraction"]

        self.config.pop("prefix")
        self.config.pop("calib")
        self.config.pop("calib region")
        self.config.pop("dla")
        self.config.pop("bal")
        self.config.pop("suffix")

    def save_key_fields(self):
        run_name = self.name

        self.first_els_values = np.array([run_name])

    @property
    def defaults_file(self) -> Path:
        return self.path.parent / "defaults.yaml"


class CorrelationConfigReader(ConfigReader):
    config_name = "bookkeeper_config.yaml"
    first_els = np.array(["delta run", "correlation run"])

    def clean_config(self):
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["correlations"]

        self.config.pop("delta extraction")
        self.config.pop("run name")

    def save_key_fields(self):
        run_name = self.name
        delta_extraction = self.path.parent.parent.parent.parent.name

        self.first_els_values = np.array([delta_extraction, run_name])

    @property
    def defaults_file(self) -> Path:
        return self.path.parent.parent.parent.parent / "configs" / "defaults.yaml"


class FitConfigReader(ConfigReader):
    config_name = "bookkeeper_config.yaml"
    first_els = np.array(["delta run", "correlation run", "fit run"])

    def clean_config(self):
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["fits"]

        self.config.pop("delta extraction")
        self.config.pop("correlation run name")
        self.config.pop("run name")

    def save_key_fields(self):
        run_name = self.name
        correlation_run = self.path.parent.parent.parent.parent.name
        delta_extraction = self.path.parent.parent.parent.parent.parent.parent.name

        self.first_els = np.array(
            [
                "delta run",
                "correlation run",
                "fit run",
            ]
        )
        self.first_els_values = np.array([delta_extraction, correlation_run, run_name])

    @property
    def defaults_file(self) -> Path:
        return (
            self.path.parent.parent.parent.parent.parent.parent
            / "configs"
            / "defaults.yaml"
        )
