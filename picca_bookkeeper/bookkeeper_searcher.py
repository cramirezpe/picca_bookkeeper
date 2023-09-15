"""Module to find bookkeepers in one base path and print differences"""
from __future__ import annotations

from pathlib import Path
from tabulate import tabulate
from typing import TYPE_CHECKING
import yaml
import numpy as np

from picca_bookkeeper.dict_utils import DictUtils

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

    T1 = TypeVar("T1", bound=int)


def get_bookkeeper_differences(
    locations: Path | str | List[Path | str],
    analysis_type: str = "delta",
    remove_identical: bool = True,
    transpose: bool = False,
    sort_by_value: bool = False,
) -> None:
    assert analysis_type in ("delta", "correlation", "fit")

    Config: Type[ConfigReader]

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
        config = Path(config)
        for file in config.glob(f"**/{filename}"):
            analyses.append(
                Config(
                    file,
                )
            )

    valid_analyses = []
    for analysis in analyses:
        if analysis.config_type == analysis.config_type_file:
            valid_analyses.append(analysis)

    show_diffs(
        [analysis.config for analysis in valid_analyses],
        name_keys=Config.first_els,
        name_values=np.array(
            [analysis.first_els_values for analysis in valid_analyses]
        ),
        remove_identical=remove_identical,
        transpose=transpose,
        sort_by_value=sort_by_value,
    )


def show_diffs(
    configs: List[Dict],
    name_keys: List[str] | np.ndarray[Tuple[T1], np.dtype[np.str_]],
    name_values: List[str] | np.ndarray[Tuple[T1], np.dtype[np.str_]],
    remove_identical: bool = True,
    transpose: bool = False,
    sort_by_value: bool = False,
    depth: int = 1,
    title: str = "",
) -> None:
    keys_ = []
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
                keys_.append(key)
    keys = np.unique(keys_)
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

    rows_ = []
    for config in configs:
        if config is None:
            config = dict()
        row = []
        for key in keys:
            if key in config:
                row.append(config[key])
            else:
                row.append("")
        rows_.append(row)
    rows = np.array(rows_)

    if rows.size == 0:
        # No data in plot, then skip
        return
    if remove_identical:
        mask = ~np.all(rows == rows[0], axis=0)
        keys = keys[mask]
        new_rows = []
        for row in rows:
            new_rows.append(row[mask])

        rows = np.array(new_rows)
        if rows.size == 0:
            # No data in plot, then skip
            return
    else:
        mask = np.ones_like(keys, dtype=bool)

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
    config_type: str = ""
    first_els = np.array([])
    first_els_values = np.array([])

    def __init__(self, bookkeeper_path: Path | str):
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

        if self.config_type_file == self.config_type:
            # Remove unneeded sections
            self.clean_config()

            # Saving fields with run names
            self.save_key_fields()

            self.data = None

    @property
    def name(self) -> str:
        return self.path.resolve().parent.parent.name

    def read_defaults(self) -> None:
        """Method to read bookkeeper defaults"""
        if self.defaults_file.is_file():
            with open(self.defaults_file) as file:
                self.defaults = yaml.safe_load(file)
        else:
            self.defaults = dict()

    def read_config(self) -> None:
        """Method to read a bookkeeper configuration yaml file"""
        with open(self.path) as file:
            self.config = yaml.safe_load(file)

        if "fits" in self.config:
            self.config_type_file = "fits"
        if "correlations" in self.config:
            self.config_type_file = "correlations"
        if "delta extraction" in self.config:
            self.config_type_file = "deltas"

        self.config = DictUtils.merge_dicts(
            self.defaults,
            DictUtils.remove_empty(self.config),
        )

    @property
    def defaults_file(self) -> Path:
        raise ValueError("Function has to be defined by child class.")

    def save_key_fields(self) -> None:
        raise ValueError("Function has to be defined by child class.")

    def clean_config(self) -> None:
        raise ValueError("Function has to be defined by child class.")


class DeltaConfigReader(ConfigReader):
    first_els = np.array(["delta run"])
    config_type = "deltas"

    def clean_config(self) -> None:
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["delta extraction"]

        self.config.pop("prefix")
        self.config.pop("calib")
        self.config.pop("calib region")
        self.config.pop("dla")
        self.config.pop("bal")
        self.config.pop("suffix")

    def save_key_fields(self) -> None:
        run_name = self.name

        self.first_els_values = np.array([run_name])

    @property
    def defaults_file(self) -> Path:
        return self.path.resolve().parent / "defaults.yaml"


class CorrelationConfigReader(ConfigReader):
    first_els = np.array(["delta run", "correlation run"])
    config_type = "correlations"

    def clean_config(self) -> None:
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["correlations"]

        self.config.pop("delta extraction")
        self.config.pop("run name")

    def save_key_fields(self) -> None:
        run_name = self.name
        delta_extraction = self.path.resolve().parent.parent.parent.parent.name

        self.first_els_values = np.array([delta_extraction, run_name])

    @property
    def defaults_file(self) -> Path:
        return (
            self.path.resolve().parent.parent.parent.parent
            / "configs"
            / "defaults.yaml"
        )


class FitConfigReader(ConfigReader):
    first_els = np.array(["delta run", "correlation run", "fit run"])
    config_type = "fits"

    def clean_config(self) -> None:
        """Method to remove unnecessary sections from config file"""
        self.config = self.config["fits"]

        self.config.pop("delta extraction")
        self.config.pop("correlation run name")
        self.config.pop("run name")

    def save_key_fields(self) -> None:
        run_name = self.name
        correlation_run = self.path.resolve().parent.parent.parent.parent.name
        delta_extraction = (
            self.path.resolve().parent.parent.parent.parent.parent.parent.name
        )

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
            self.path.resolve().parent.parent.parent.parent.parent.parent
            / "configs"
            / "defaults.yaml"
        )
