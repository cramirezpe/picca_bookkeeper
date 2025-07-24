"""
picca_bookkeeper.bookkeeper

This module provides the Bookkeeper and PathBuilder classes, which manage configuration, 
tasking, and filesystem structure for orchestrating picca and vega analysis runs, including 
delta extraction, correlations, and fitting. It includes task scheduling, configuration 
validation, and path management utilities.

Classes:
    - Bookkeeper: High-level interface for managing picca and vega jobs and configurations.
    - PathBuilder: Utility for generating standardized paths for input/output.
"""

from __future__ import annotations

import configparser
import copy
import filecmp
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from importlib_resources import files
from yaml import SafeDumper

from picca_bookkeeper import resources
from picca_bookkeeper.constants import absorber_igm, forest_regions
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.tasker import ChainedTasker, DummyTasker, Tasker, get_Tasker

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# This converts Nones in dict into empty fields in yaml.
SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)

config_file_sorting = ["general", "delta extraction", "correlations", "fits"]


class Bookkeeper:
    """
    Class to generate Tasker objects which can be used to run different picca jobs.

    Handles configuration loading, validation, and management of directories, 
    and constructs Tasker objects to run various stages of the analysis 
    (delta extraction, correlation, fitting, etc.).

    Attributes:
        config (configparser.ConfigParser): Configuration file for the bookkeeper.
        paths (PathBuilder): Helper for constructing standardized file paths.

    Usage:
        bookkeeper = Bookkeeper("config.yaml")
        tasker = bookkeeper.get_delta_extraction_tasker(region="lya")
    """
    
    label: Optional[str]

    def __init__(
        self,
        config_path: str | Path,
        overwrite_config: bool = False,
        read_mode: bool = True,
    ):
        """
        Args:
            config_path (Path or str): Path to configuration file or to an already
                created run path.
            overwrite_config (bool, optional): overwrite bookkeeper config without
                asking if it already exists inside bookkeeper.
            read_modeo (bool, optional): do not try to write or create bookkeeper
                structure.
        """
        # Try to read the file of the folder
        config_path = Path(config_path)
        if not config_path.is_file():
            if (config_path / "configs/bookkeeper_config.yaml").is_file():
                config_path = config_path / "configs/bookkeeper_config.yaml"
            else:
                raise FileNotFoundError("Config file couldn't be found", config_path)

        with open(config_path) as file:
            self.config = yaml.safe_load(file)

        self.read_mode = read_mode
        self.overwrite_config = overwrite_config

        # Needed to retrieve continuum tag
        self.paths = PathBuilder(self.config)

        ## Needed for partially written bookkeepers
        if self.config.get("correlations", None) is None:
            self.config["correlations"] = dict()

        if self.config.get("fits", None) is None:
            self.config["fits"] = dict()

        self.paths.check_run_directories()
        # If delta extraction comes from another bookkeeper
        # we have to properly link it
        # We also check that files in the bookkeeper folder match input config.
        if self.config["delta extraction"].get("use existing", None) is not None:
            original_config = (
                Path(self.config["delta extraction"]["use existing"])
                / "configs/bookkeeper_config.yaml"
            )
            self.config["delta extraction"] = yaml.safe_load(
                original_config.read_text()
            )["delta extraction"]
            self.config["delta extraction"]["original path"] = original_config.parents[
                1
            ]
        elif not self.read_mode:
            self.paths.check_delta_directories()
            self.check_existing_config("delta extraction", self.paths.config_file)

        if self.config["correlations"].get("use existing", None) is not None:
            original_config = (
                Path(self.config["correlations"]["use existing"])
                / "configs/bookkeeper_config.yaml"
            )
            self.config["correlations"] = yaml.safe_load(original_config.read_text())[
                "correlations"
            ]
            self.config["correlations"]["original path"] = original_config.parents[1]
        elif not self.read_mode:
            self.paths.check_correlation_directories()
            self.check_existing_config("correlations", self.paths.config_file)

        self.paths.check_fit_directories()
        self.check_existing_config("fits", self.paths.config_file)

        self.check_bookkeeper_config()

        if self.read_mode:
            # Only need to read defaults if not in
            # read mode.
            return

        if self.config.get("general").get("defaults file", None) is not None:
            # Read defaults and check if they have changed.
            if Path(self.config["general"]["defaults file"]).is_file():
                defaults_file = Path(self.config["general"]["defaults file"])
            else:
                # Read defaults and check if they have changed.
                defaults_file = files(resources).joinpath(
                    "default_configs/"
                    + str(self.config["general"]["defaults file"])
                    + ".yaml"
                )

            if not defaults_file.is_file():
                raise ValueError("Defaults file not found.", defaults_file)

            self.defaults = yaml.safe_load(defaults_file.read_text())

            self.defaults_diff = dict()

            if self.paths.defaults_file.is_file():
                self.defaults_diff = PathBuilder.compare_configs(
                    self.defaults,
                    yaml.safe_load(self.paths.defaults_file.read_text()),
                )
            else:
                self.defaults_diff = {}
                self.write_bookkeeper(
                    self.defaults,
                    self.paths.defaults_file,
                )

            self.config = DictUtils.merge_dicts(
                self.defaults,
                self.config,
            )

        else:
            self.defaults = dict()
            self.defaults_diff = dict()
        # Update paths config
        self.paths.config = self.config

    def check_existing_config(self, section: str, destination: Path) -> None:
        """
        Ensure that the configuration for a section matches the on-disk config file.
    
        If the config differs, this method will raise a ValueError unless overwrite is enabled.
        Writes the config to disk if missing or if overwrite is set.
    
        Args:
            section (str): Which section of the config to check (e.g., 'delta extraction').
            destination (Path): The config file to check/write.
    
        Raises:
            ValueError: If the configs differ and overwrite is False.
        """
        
        config = copy.deepcopy(self.config)

        # for key in list(config.keys()):
        #     if key not in (section, "general", "data"):
        #         config.pop(key, None)

        if not destination.is_file():
            self.write_bookkeeper(config, destination)
        elif self.overwrite_config:
            self.write_bookkeeper(config, destination)
        else:
            comparison = PathBuilder.compare_configs(
                config,
                yaml.safe_load(destination.read_text()),
                section,
            )
            if comparison == dict():
                # They are the same
                self.write_bookkeeper(config, destination)
            else:
                raise ValueError(
                    f"Incompatible configs: {destination}\n"
                    f"{DictUtils.print_dict(comparison)}\n\n"
                )

    @staticmethod
    def write_ini(config: Dict, file: Path | str) -> None:
        """
        Safely save a dictionary into an .ini file
    
        Converts all values to strings and uses ConfigParser to write the result 
        to disk with consistent formatting.
    
        Args:
            config (Dict): Dict to store as ini file.
            file (Path | str): Destination path for the .ini file.
        """
    
        config = DictUtils.convert_to_string(config)

        parser = configparser.ConfigParser()
        parser.optionxform = str  # type: ignore
        parser.read_dict(config)

        with open(file, "w") as write_file:
            parser.write(write_file)

    @staticmethod
    def write_bookkeeper(config: Dict, file: Path | str) -> None:
        """
        Write a validated bookkeeper YAML configuration file to file.
    
        This method transforms the configuration to ensure compatibility with 
        existing workflows by rewriting certain sections (e.g., replacing 
        "original path" with "use existing"). It also enforces a specific key 
        ordering for clarity and consistency.
    
        Args:
            config (Dict): Bookkeeper configuration to store as yaml file.
            file (Path | str): Destination path for the bookkeeper config file.
    
        Raises:
            ValueError: If the configuration contains unexpected keys that violate the expected order.
        """
        
        # I need to recover the use existing and remove the original structure
        # To avoid issues if re-running with stored bookkeeper
        if (
            config.get("delta extraction", dict()).get("original path", None)
            is not None
        ):
            config["delta extraction"] = {
                "use existing": str(config["delta extraction"]["original path"]),
            }

        if config.get("correlations", dict()).get("original path", None) is not None:
            config["correlations"] = {
                "use existing": str(config["correlations"]["original path"]),
            }

        correct_order = {
            "general": [
                "conda environment",
                "system",
                "slurm args",
                "defaults file",
                "raw mocks",
                "true mocks",
            ],
            "data": ["bookkeeper dir", "healpix data", "catalog"],
            "delta extraction": [
                "use existing",
                "calib",
                "calib region",
                "dla",
                "bal",
                "calibration data",
                "computed deltas",
                "mask file",
                "extra args",
                "slurm args",
            ],
            "correlations": [
                "use existing",
                "unblind",
                "unblind y1",
                "tracer catalogs",
                "fast metals",
                "computed correlations",
                "computed exports",
                "computed distortions",
                "computed metals",
                "extra args",
                "slurm args",
            ],
            "fits": [
                "auto correlations",
                "cross correlations",
                "compute covariance",
                "smooth covariance",
                "computed covariances",
                "compute zeff",
                "vega metals",
                "sampler environment",
                "no distortion",
                "no bao",
                "no hcd",
                "no metals",
                "no sky",
                "no qso rad",
                "rmin cf",
                "rmax cf",
                "rmin xcf",
                "rmax xcf",
                "extra args",
                "slurm args",
            ],
        }

        try:
            config = dict(
                sorted(config.items(), key=lambda s: list(correct_order).index(s[0]))
            )

            for key, value in config.items():
                config[key] = dict(
                    sorted(value.items(), key=lambda s: correct_order[key].index(s[0]))
                )
        except ValueError as e:
            raise ValueError(f"Invalid item in config file").with_traceback(
                e.__traceback__
            )

        with open(file, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def check_bookkeeper_config(self) -> None:        
        """
        Validate the bookkeeper config file, raise error if invalid.
    
        Checks for incompatible settings (e.g., smoothing covariance without computing it),
        validates command names for each analysis stage, and ensures region keys are valid.
    
        Raises:
            ValueError: If an invalid configuration is detected, such as:
                - Smoothing covariance without enabling covariance computation.
                - Unrecognized command names in 'slurm args' or 'extra args'.
                - Invalid region names within argument mappings.
        """

        logger.debug("Checking smooth covariance consistency")
        if self.config.get("fits", dict()).get(
            "smooth covariance", False
        ) and not self.config.get("fits", dict()).get("compute covariance", False):
            raise ValueError("Smooth covariance requires compute covariance.")

        logger.debug("Checking command names.")
        delta_extraction_commands = [
            "picca_delta_extraction",
            "picca_convert_transmission",
        ]
        correlation_commands = [
            "picca_cf",
            "picca_xcf",
            "picca_dmat",
            "picca_xdmat",
            "picca_metal_dmat",
            "picca_metal_xdmat",
            "picca_fast_metal_dmat",
            "picca_fast_metal_xdmat",
            "picca_export",
        ]
        fit_commands = [
            "picca_bookkeeper_correct_config_zeff",
            "run_vega",
            "run_vega_mpi",
            "vega_auto",
            "vega_cross",
            "vega_main",
            "write_full_covariance",
            "smooth_covariance",
        ]

        valid_regions = list(forest_regions.keys()) + ["general", "all"]

        for arg_type in (
            "slurm args",
            "extra args",
        ):
            for section, commands in zip(
                ("delta extraction", "correlations", "fits"),
                (delta_extraction_commands, correlation_commands, fit_commands),
            ):
                for key in (
                    self.config.get(section, dict()).get(arg_type, dict()).keys()
                ):
                    if key not in commands:
                        raise ValueError("Invalid command in bookkeeper config: ", key)

                    if self.config[section][arg_type][key] is not None:
                        for region in self.config[section][arg_type][key].keys():
                            if region not in valid_regions:
                                raise ValueError(
                                    f"Invalid region ({region}) in bookkeeper config "
                                    f"section: {section} - {arg_type} - {key}"
                                )

    @staticmethod
    def validate_region(region: str) -> str:        
        """
        Validate that the specified region is present in the forest_regions dictionary.
    
        Args:
            region: Region (should be in forest_regions to pass the validation).
    
        Returns:
            str: The validated region name.
    
        Raises:
            ValueError: If the region string is not recognized / not in forest_regions.
        """
        
        if region not in forest_regions:
            raise ValueError("Invalid region", region)

        return region

    @staticmethod
    def validate_absorber(absorber: str) -> str:
        """
        Validate that the specified absorber is supported in the absorber list.
    
        Allows lowercase 'lya' and 'lyb' as valid exceptions. Raises an error for 
        any unrecognized absorber name.
    
        Args:
            absorber (str): Absorber name to use.
    
        Returns:
            str: The validated absorber name.
    
        Raises:
            ValueError: If the absorber is not found in absorber_igm.
        """

        if absorber not in absorber_igm:
            raise ValueError("Invalid absorber", absorber)

        return absorber

    def generate_slurm_header_extra_args(
        self,
        config: Dict,
        section: str,
        command: str,
        region: Optional[str] = None,
        absorber: Optional[str] = None,
        region2: Optional[str] = None,
        absorber2: Optional[str] = None,
        tracer: Optional[str] = None,
    ) -> Dict:
        """
        Generate and merge extra SLURM header arguments to the run.
    
        This method constructs a subcommand key based on absorber(s), region(s), and tracer,
        then looks up and merges SLURM arguments from the general and section-specific config
        under matching subcommand keys. General arguments are merged first, followed by more
        specific ones, with later entries overriding earlier ones.
    
        Args:
            config (Dict): Bookkeeper config.
            section (str): Section of the config to query (e.g., "correlations", "fits").
            command (str): Command to be run (e.g., "picca_cf", "smooth_covariance").
            region (Optional[str]): Specify region where the command will be run.
            absorber (Optional[str]): First absorber to use for correlations.
            region2 (Optional[str]): Region label for the second absorber (if applicable).
            absorber2 (Optional[str]): Second absorber to use for correlations (if applicable).
            tracer (Optional[str]): Tracer used in cross-correlations (e.g., "qso", "dla").
    
        Returns:
            Dict: Merged SLURM argument dictionary for the specified context.
        """
        
        if "slurm args" in config["general"]:
            args = copy.deepcopy(config["general"]["slurm args"])
        else:
            args = dict()

        config = copy.deepcopy(config[section])

        command_name = command.split(".py")[0]

        tracer = "" if tracer is None else tracer
        absorber = "" if absorber is None else absorber
        absorber2 = "" if absorber2 is None else absorber2
        region = "" if region is None else region
        region2 = "" if region2 is None else region2

        region_subcommand = ""

        if tracer != "":
            region_subcommand += f"{tracer}_"

        if region != "":
            region_subcommand += f"{absorber}{region}"

        if region2 != "":
            region_subcommand += f"_{absorber2}{region2}"

        for subcommand in ("general", region_subcommand, "all"):
            if (
                config.get("slurm args", None) is not None
                and config["slurm args"].get(command_name, None) is not None
                and config["slurm args"][command_name].get(subcommand, None) is not None
            ):
                args = DictUtils.merge_dicts(
                    args,
                    config["slurm args"][command_name][subcommand],
                )

        return DictUtils.remove_dollar(args)

    def generate_extra_args(
        self,
        config: Dict,
        section: str,
        command: str,
        region: Optional[str] = None,
        absorber: Optional[str] = None,
        region2: Optional[str] = None,
        absorber2: Optional[str] = None,
        tracer: Optional[str] = None,
    ) -> Dict:
        """
        Generate and merge additional runtime arguments to the run.
    
        Constructs a subcommand key using absorber(s), region(s), and tracer, and merges any 
        matching 'extra args' entries from the config. General arguments are merged first, followed 
        by context-specific overrides.
    
        Args:
            config (Dict): Section of the bookkeeper config to look into.
            section (str): Section of the config to query (e.g., "correlations", "fits"), 
                            should be prioritized.
            command (str): Command to be run (e.g., "picca_cf", "vega_cross").
            region (Optional[str]): Specify region where the command will be run.
            absorber (Optional[str]): First absorber to use for correlations.
            region2 (Optional[str]): Region label for the second absorber (if applicable).
            absorber2 (Optional[str]): Second absorber to use for correlations (if applicable).
            tracer (Optional[str]): Tracer used in cross-correlations (e.g., "qso", "dla").
    
        Returns:
            Dict: Merged dictionary of extra arguments for the specified command context.
        """
        
        config = copy.deepcopy(config[section])

        command_name = command.split(".py")[0]

        tracer = "" if tracer is None else tracer
        absorber = "" if absorber is None else absorber
        absorber2 = "" if absorber2 is None else absorber2
        region = "" if region is None else region
        region2 = "" if region2 is None else region2

        region_subcommand = ""

        if tracer != "":
            region_subcommand += f"{tracer}_"

        if region != "":
            region_subcommand += f"{absorber}{region}"

        if region2 != "":
            region_subcommand += f"_{absorber2}{region2}"

        args: Dict = dict()
        for subcommand in ("general", region_subcommand, "all"):
            if (
                config.get("extra args", None) is not None
                and config["extra args"].get(command_name, None) is not None
                and config["extra args"][command_name].get(subcommand, None) is not None
            ):
                args = DictUtils.merge_dicts(
                    args,
                    config["extra args"][command_name][subcommand],
                )

        return DictUtils.remove_dollar(args)

    def generate_system_arg(self, system: Optional[str]) -> str:
        """
        Return the specified system or fallback to the default from config.
    
        Args:
            system (Optional[str]): System name to use, or None to use the default.
    
        Returns:
            str: Resolved system name (e.g., "slurm_perlmutter").
        """
        
        if system is None:
            return copy.copy(
                self.config.get("general", dict()).get("system", "slurm_perlmutter")
            )
        else:
            return system

    def add_calibration_options(
        self, extra_args: Dict, calib_step: Optional[int] = None
    ) -> Tuple[List[Path], Dict]:
        """
        Method to add calibration options to extra args. 
    
        Depending on the calibration mode set in the config (`calib` = 1 or 2), this method 
        appends the appropriate correction types and file paths to the argument dictionary. 
        It also verifies that required calibration files and directories exist and raises 
        errors when they don't.
    
        Args:
            extra_args (Dict): Configuration to be used in the run.
            calib_step (Optional[int]): Current calibration step (1 or 2), or None for the main run.
    
        Returns:
            Tuple[List[Path], Dict]: A tuple containing:
                - List of input files required for calibration.
                - Updated `extra_args` dictionary including calibration corrections.
    
        Raises:
            ValueError: If calibration settings are inconsistent with the config.
            FileNotFoundError: If required calibration output files or directories are missing.
        """
        
        # List input files needed
        input_files = []

        # update corrections section
        # here we are dealing with calibration runs
        # If there is no calibration, we should not have calib_steps
        if self.config["delta extraction"]["calib"] not in (0, 10):
            if (
                "CalibrationCorrection" in extra_args["corrections"].values()
                or "IvarCorrection" in extra_args["corrections"].values()
            ):
                raise ValueError(
                    "Calibration corrections added by user with calib option != 10"
                )

        if self.config["delta extraction"]["calib"] == 0 and calib_step is not None:
            raise ValueError("Trying to run calibration with calib = 0 in config file.")

        # Two calibration steps:
        elif self.config["delta extraction"]["calib"] == 2:
            # Second calibration step
            if calib_step is not None:
                if calib_step == 2:
                    num_corrections = (
                        int(
                            extra_args.get("corrections", dict()).get(
                                "num corrections", 0
                            )
                        )
                        + 1
                    )

                    extra_args = DictUtils.merge_dicts(
                        extra_args,
                        {
                            "corrections": {
                                "num corrections": num_corrections,
                                f"type {num_corrections - 1}": "CalibrationCorrection",
                            },
                            f"correction arguments {num_corrections - 1}": {
                                "filename": str(
                                    self.paths.delta_attributes_file(
                                        None, calib_step=1
                                    ).resolve()
                                ),
                            },
                        },
                    )
                    input_files.append(
                        self.paths.delta_attributes_file(None, calib_step=1)
                    )

            # Actual run (no calibration)
            else:
                if not self.paths.deltas_log_path(None, calib_step=2).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration_tasker "
                        "before running deltas."
                    )

                num_corrections = (
                    int(extra_args.get("corrections", dict()).get("num corrections", 0))
                    + 2
                )

                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "corrections": {
                            "num corrections": num_corrections,
                            f"type {num_corrections -2}": "CalibrationCorrection",
                            f"type {num_corrections -1}": "IvarCorrection",
                        },
                        f"correction arguments {num_corrections - 2}": {
                            "filename": str(
                                self.paths.delta_attributes_file(
                                    None, calib_step=1
                                ).resolve()
                            ),
                        },
                        f"correction arguments {num_corrections - 1}": {
                            "filename": str(
                                self.paths.delta_attributes_file(
                                    None, calib_step=2
                                ).resolve()
                            ),
                        },
                    },
                )
                input_files.append(self.paths.delta_attributes_file(None, calib_step=1))
                input_files.append(self.paths.delta_attributes_file(None, calib_step=2))

        elif self.config["delta extraction"]["calib"] == 1:
            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.paths.deltas_log_path(None, calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration_tasker "
                        "before running deltas."
                    )
                num_corrections = (
                    int(extra_args.get("corrections", dict()).get("num corrections", 0))
                    + 1
                )

                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "corrections": {
                            "num corrections": num_corrections,
                            f"type {num_corrections - 1}": "CalibrationCorrection",
                        },
                        f"correction arguments {num_corrections - 1}": {
                            "filename": str(
                                self.paths.delta_attributes_file(
                                    None, calib_step=1
                                ).resolve()
                            ),
                        },
                    },
                )
                input_files.append(self.paths.delta_attributes_file(None, calib_step=1))
        return input_files, extra_args

    def add_mask_options(
        self, extra_args: Dict, calib_step: Optional[int] = None
    ) -> Dict:
        """
        Add masking options to extra args.
    
        Includes user-supplied mask files, and optionally adds DLA or BAL masking based on 
        the bookkeeper config YAML. Ensures that referenced mask files exist and are consistent 
        with stored copies, raising errors if mismatches are detected.
    
        Args:
            extra_args (Dict): Configuration to be used in the run.
            calib_step (Optional[int]): Current calibration step. If None, masking for DLA/BAL is applied.
    
        Returns:
            Dict: Updated `extra_args` dictionary including all applicable mask configurations.
    
        Raises:
            FileNotFoundError: If a user-provided mask file is missing.
            FileExistsError: If a different mask file is already stored in the bookkeeper path.
            ValueError: If a DlaMask or BalMask is already set by the user when the 
                        corresponding config option is active.
        """

        if self.config["delta extraction"].get("mask file", "") not in ("", None):
            # If a mask file is given in the config file
            if not Path(self.config["delta extraction"]["mask file"]).is_file():
                # If the file does not exist
                raise FileNotFoundError("Provided mask file does not exist.")
            else:
                # If the file does exist
                if not self.paths.continuum_fitting_mask.is_file():
                    # If no file in output bookkeeper
                    shutil.copy(
                        self.config["delta extraction"]["mask file"],
                        self.paths.continuum_fitting_mask,
                    )
                # If file in output bookkeeper
                elif not filecmp.cmp(
                    self.paths.continuum_fitting_mask,
                    self.config["delta extraction"]["mask file"],
                ):
                    raise FileExistsError(
                        "different mask file already stored in the bookkeeper",
                        self.paths.continuum_fitting_mask,
                    )

            num_masks = int(extra_args["masks"]["num masks"]) + 1
            extra_args = DictUtils.merge_dicts(
                extra_args,
                {
                    "masks": {
                        "num masks": num_masks,
                        f"type {num_masks - 1}": "LinesMask",
                    },
                    f"mask arguments {num_masks - 1}": {
                        "filename": self.paths.continuum_fitting_mask,
                    },
                },
            )

        # update masks sections if necessary
        if calib_step is None:
            if self.config["delta extraction"].get("dla", None) is not None:
                if "DlaMask" in extra_args["masks"].values():
                    raise ValueError("DlaMask set by user with dla option != None")

                prev_mask_number = int(extra_args["masks"]["num masks"])
                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "masks": {
                            "num masks": prev_mask_number + 1,
                            f"type {prev_mask_number}": "DlaMask",
                        },
                        f"mask arguments {prev_mask_number}": {
                            f"filename": self.paths.catalog_dla.resolve(),
                            "los_id name": "TARGETID",
                        },
                    },
                )

            if self.config["delta extraction"].get("bal", None) not in (None, False):
                if "BalMask" in extra_args["masks"].values():
                    raise ValueError("BalMask set by user with bal option != None")

                prev_mask_number = int(extra_args["masks"]["num masks"])
                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "masks": {
                            "num masks": prev_mask_number + 1,
                            f"type {prev_mask_number}": "BalMask",
                        },
                        f"mask arguments {prev_mask_number}": {
                            f"filename": self.paths.catalog_bal.resolve(),
                            "los_id name": "TARGETID",
                        },
                    },
                )

        return extra_args

    def validate_mock_runs(self, deltas_config_dict: Dict) -> None:
        """
        Validate config file for mock runs.
    
        Performs sanity checks on the bookkeeper config and provided delta config 
        to ensure compatibility with mock run requirements. Raises errors if 
        raw mocks are improperly configured or if required expected flux parameters 
        for true mocks are missing or incomplete.
    
        Args:
            deltas_config_dict (Dict): Dictionary containing config to be used.
    
        Raises:
            ValueError: If raw mocks are set but improperly handled, or if required 
                        expected flux fields are missing for true mocks.
        """
        # if self.config["delta extraction"]["prefix"] not in [
        #     "dMdB20",
        #     "raw",
        #     True,
        #     "custom",
        # ]:
        #     raise ValueError(
        #         f"Unrecognized continuum fitting prefix: "
        #         f"{self.config['delta extraction']['prefix']}"
        #     )

        if self.config["general"].get("raw mocks", False):
            raise ValueError(
                f"raw continuum fitting provided in config file, use "
                "get_raw_deltas_tasker instead"
            )
        if self.config["general"].get("true mocks", False):
            if (
                "expected flux" not in deltas_config_dict
                or "raw statistics file" not in deltas_config_dict["expected flux"]
                or deltas_config_dict["expected flux"]["raw statistics file"]
                in ("", None)
            ):
                raise ValueError(
                    f"Should define expected flux and raw statistics file in extra "
                    "args section in order to run TrueContinuum"
                )
            deltas_config_dict["expected flux"]["type"] = "TrueContinuum"
            if (
                deltas_config_dict.get("expected flux", dict()).get(
                    "input directory", 0
                )
                == 0
            ):
                deltas_config_dict["expected flux"][
                    "input directory"
                ] = self.paths.healpix_data

    def get_raw_deltas_tasker(
        self,
        region: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Create a Tasker object to run raw deltas with picca.
    
        This method sets up all necessary arguments, SLURM header options, and paths 
        for running the delta extraction step on the specified forest region. It 
        validates configurations, checks for existing outputs, and supports debug mode.
    
        Args:
            region (str): Forest region to compute deltas (default: 'lya'). Must be in `forest_regions`.
            system (Optional[str]): Execution system or shell to use (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug (bool): Enable debug mode with reduced runtime and sample size (default: False).
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting 
                (default: None). Could be a  Tasker object or a slurm jobid
            overwrite (bool): Overwrite existing files if True (default: False).
            skip_sent (bool): If True, skip and return a DummyTasker if output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object ready to submit or run the delta extraction job.
    
        Raises:
            ValueError: If the Bookkeeper is initialized in read mode or if default config values have changed 
                since the last run.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        job_name = f"raw_deltas_{region}"
        if self.check_existing_output_file(
            self.paths.delta_attributes_file(region),
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        command = "picca_convert_transmission.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="delta extraction",
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="delta extraction",
            command=command,
            region=region,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.delta_extraction_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.delta_extraction_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "object-cat": str(self.paths.catalog),
            "in-dir": str(self.paths.healpix_data.resolve()),
            "out-dir": str(self.paths.deltas_path(region)),
            "lambda-rest-min": forest_regions[region]["lambda-rest-min"],
            "lambda-rest-max": forest_regions[region]["lambda-rest-max"],
        }
        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args,
                dict(qos="debug", time="00:30:00"),
            )
            args = DictUtils.merge_dicts(args, dict(nspec=1000))

        self.paths.deltas_path(region).mkdir(exist_ok=True, parents=True)
        self.paths.delta_attributes_file(region, None).parent.mkdir(
            exist_ok=True, parents=True
        )
        delta_stats_file = self.paths.deltas_path(region).parent / "Delta-stats.fits.gz"
        self.paths.delta_attributes_file(region, None).symlink_to(delta_stats_file)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.delta_extraction_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.delta_extraction_path / f"logs/jobids.log",
            wait_for=wait_for,
            out_files=[
                delta_stats_file,
            ],
        )

    def get_delta_extraction_tasker(
        self,
        region: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        calib_step: Optional[int] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:   
        """
        Create a Tasker object to run delta extraction with picca.
    
        This method prepares the configuration, SLURM headers, and input/output file handling 
        necessary for running the delta extraction on a specified forest region. It supports 
        optional calibration steps, validates mock run configurations, and handles cases 
        where outputs or calibration data already exist by returning a DummyTasker.
    
        Args:
            region (str): Forest region to compute deltas (default: 'lya'). Must be in `forest_regions`.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug (bool): Enable debug mode with reduced runtime and sample size (default: False).
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting 
                (default: None). Could be a Tasker object or a slurm jobid.
            calib_step (Optional[int]): Calibration step number. If None, runs without calibration (default: None).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): If True, skip and return a DummyTasker if output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object ready to submit or run the delta extraction job.
    
        Raises:
            ValueError: If the Bookkeeper is initialized in read mode or if default config values have changed 
                since the last run (most overwrite or delete existing config file).
        """
        copied_attributes: Path | None

        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        job_name = f"delta_extraction_{region}"
        if calib_step is not None:
            job_name += "_calib_step_" + str(calib_step)

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        if self.check_existing_output_file(
            self.paths.delta_attributes_file(region, calib_step),
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        # Check if calibration data needs to be copied:
        if calib_step is not None:
            copy_deltas_paths = self.paths.copied_deltas_files(
                f"calibration_{calib_step}"
            )
            if copy_deltas_paths is not None:
                copied_attributes = copy_deltas_paths[1]
            else:
                copied_attributes = self.paths.copied_calib_attributes(calib_step)

            if copied_attributes is not None:
                filename = self.paths.delta_attributes_file(None, calib_step)
                filename.unlink(missing_ok=True)
                filename.parent.mkdir(exist_ok=True, parents=True)
                filename.symlink_to(copied_attributes)

                return DummyTasker()

        copy_deltas_paths = self.paths.copied_deltas_files(region)
        if copy_deltas_paths is not None:
            deltas_dir = self.paths.deltas_path(region)
            if deltas_dir.is_dir() or deltas_dir.is_file():
                shutil.rmtree(deltas_dir)
            deltas_dir.parent.mkdir(exist_ok=True, parents=True)
            deltas_dir.symlink_to(copy_deltas_paths[0])

            filename = self.paths.delta_attributes_file(region)
            filename.unlink(missing_ok=True)
            filename.parent.mkdir(exist_ok=True, parents=True)
            filename.symlink_to(copy_deltas_paths[1])

            return DummyTasker()

        command = "picca_delta_extraction.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="delta extraction",
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="delta extraction",
            command=command,
            region=region,
        )

        config_file = self.paths.delta_extraction_path / f"configs/{job_name}.ini"

        deltas_config_dict = DictUtils.merge_dicts(
            {
                "general": {
                    "overwrite": True,
                    "out dir": str(
                        self.paths.deltas_path(region, calib_step).parent.resolve()
                    )
                    + "/",
                },
                "data": {
                    "type": "DesiHealpixFast",
                    "catalogue": str(self.paths.catalog),
                    "input directory": str(self.paths.healpix_data.resolve()),
                    "lambda min rest frame": forest_regions[region]["lambda-rest-min"],
                    "lambda max rest frame": forest_regions[region]["lambda-rest-max"],
                },
                "corrections": {
                    "num corrections": 0,
                },
                "masks": {
                    "num masks": 0,
                },
                "expected flux": {},
            },
            updated_extra_args,
        )

        input_files, deltas_config_dict = self.add_calibration_options(
            deltas_config_dict,
            calib_step,
        )

        deltas_config_dict = self.add_mask_options(
            deltas_config_dict,
            calib_step,
        )

        self.validate_mock_runs(deltas_config_dict)

        # create config for delta_extraction options
        self.paths.deltas_path(region, calib_step).mkdir(parents=True, exist_ok=True)
        self.paths.deltas_log_path(region, calib_step).mkdir(
            parents=True, exist_ok=True
        )
        self.write_ini(deltas_config_dict, config_file)

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.delta_extraction_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.delta_extraction_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            deltas_config_dict.get("data", dict()).update({"max num spec": 1000})

        return get_Tasker(updated_system)(
            command=command,
            command_args={"": str(config_file.resolve())},
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.delta_extraction_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.delta_extraction_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=input_files,
            out_files=[
                self.paths.delta_attributes_file(region, calib_step),
            ],
        )

    def get_calibration_extraction_tasker(
        self,
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> ChainedTasker:
        """
        Method to get a Tasker object to run calibration with picca delta
        extraction method.
    
        This method determines the appropriate calibration steps based on the config,
        validates the calibration region, and returns a chained sequence of Taskers
        for each calibration step. It handles calibration modes 1, 2, and 10 explicitly,
        raising errors if the configuration is invalid or incomplete.
    
        Args:
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug (bool): Enable debug mode with reduced runtime and sample size (default: False).
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting 
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): If True, skip and return a DummyTasker if outputs already exist (default: False).
    
        Returns:
            ChainedTasker: A chained sequence of Taskers that run calibration delta extraction steps in order.
    
        Raises:
            ValueError: If the bookkeeper is in read mode, the calibration region is undefined or invalid,
                or the calibration mode is unsupported.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        steps = []
        region = self.config["delta extraction"].get("calib region", 0)
        if region in (0, "", None):
            raise ValueError("Calibration region not defined in config file.")
        else:
            region = self.validate_region(region)

        if self.config["delta extraction"]["calib"] not in [0, 1, 2, 10]:
            raise ValueError(
                "Invalid calib value in config file. (Valid values are 0 1 2 3 10)"
            )
        elif self.config["delta extraction"]["calib"] in (2,):
            steps.append(
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    calib_step=1,
                    overwrite=overwrite,
                    skip_sent=skip_sent,
                )
            )
            steps.append(
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=steps[0],
                    calib_step=2,
                    overwrite=overwrite,
                    skip_sent=skip_sent,
                )
            )
        elif self.config["delta extraction"]["calib"] in (1, 10):
            steps = [
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    calib_step=1,
                    overwrite=overwrite,
                    skip_sent=skip_sent,
                ),
            ]
        else:
            raise ValueError("Trying to run calibration with calib=0 in config file.")

        return ChainedTasker(taskers=steps)

    def get_cf_tasker(
        self,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-forest correlations with picca.    
        
        Args:
            region (str): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (Optional[str]): Secondary forest region for cross-correlation.
                Defaults to None, which means auto-correlation with the primary region.
            absorber (str): First absorber to use for correlations (e.g., 'lya').
            absorber2 (Optional[str]): Second absorber to use for correlations.
                Defaults to None, which uses the same absorber as `absorber`.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug (bool): Enable debug mode with reduced runtime and sample size (default: False).
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting 
                (default: None). Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: A Tasker object configured to run the forest correlation function.
    
        Raises:
            ValueError: If bookkeeper is initialized in read mode or if default config values have changed
                since the last run.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        job_name = f"cf_{absorber}{region}_{absorber2}{region2}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.cf_fname(absorber, region, absorber2, region2)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_cf_file = self.paths.copied_correlation_file(
            "computed correlations",
            absorber,
            region,
            absorber2,
            region2,
            output_filename.name,
        )
        if copy_cf_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_cf_file)

            return DummyTasker()

        command = "picca_cf.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "in-dir": str(self.paths.deltas_path(region).resolve()),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        if absorber2 != absorber:
            args["lambda-abs2"] = absorber_igm[absorber2]

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2).resolve())

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_dmat_tasker(
        self,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-forest distortion matrix
        measurements with picca.
    
        Args:
            region (str): Primary forest region. Options include 'lya', 'lyb'. Default is 'lya'.
            region2 (Optional[str]): Region to use for cross-correlations.
                Default: None, auto-correlation.
            absorber (str): First absorber to use for correlations (e.g., 'lya').
            absorber2 (Optional[str]): Second absorber to use for correlations.
                Defaults to None, which uses the same absorber as `absorber`.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug (bool): Enable debug mode with reduced runtime and sample size (default: False).
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting 
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: A Tasker object configured to run the distortion matrix measurement.
    
        Raises:
            ValueError: If bookkeeper is initialized in read mode or if default config values
                have changed since the last run.
        """        
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        job_name = f"dmat_{absorber}{region}_{absorber2}{region2}"
        job_name = job_name.replace("(", "").replace(")", "")

        output_filename = self.paths.dmat_fname(absorber, region, absorber2, region2)
        # Check if output already there
        updated_system = self.generate_system_arg(system)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_dmat_file = self.paths.copied_correlation_file(
            "computed distortions",
            absorber,
            region,
            absorber2,
            region2,
            output_filename.name,
        )
        if copy_dmat_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_dmat_file)

            return DummyTasker()

        command = "picca_dmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "in-dir": str(self.paths.deltas_path(region).resolve()),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        if absorber2 != absorber:
            args["lambda-abs2"] = absorber_igm[absorber2]

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2).resolve())

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args,
                dict(qos="debug", time="00:30:00"),
            )
            args = DictUtils.merge_dicts(args, dict(nspec=1000))

        output_filename.parent.mkdir(
            exist_ok=True,
            parents=True,
        )

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_cf_exp_tasker(
        self,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-forest correlation export with
         picca.
    
        Args:
            region (str): Primary forest region. Options include 'lya', 'lyb'. Default: 'lya'.
            region2 (Optional[str]): Secondary forest region for cross-correlation.
                Defaults to None for auto-correlation with `region`.
            absorber (str): First absorber to use for correlations (e.g., 'lya').
            absorber2 (Optional[str]): cond absorber to use for cross-correlations.
                Defaults to None, meaning same absorber as `absorber`.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to run the correlation export.
    
        Raises:
            ValueError: If bookkeeper is initialized in read mode or if default config values
                have changed since the last run.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        job_name = f"cf_exp_{absorber}{region}_{absorber2}{region2}"
        job_name = job_name.replace("(", "").replace(")", "")

        output_filename = self.paths.exp_cf_fname(absorber, region, absorber2, region2)
        # Check if output already there
        updated_system = self.generate_system_arg(system)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_cf_exp_file = self.paths.copied_correlation_file(
            "computed exports",
            absorber,
            region,
            absorber2,
            region2,
            output_filename.name,
        )
        if copy_cf_exp_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_cf_exp_file)

            return DummyTasker()

        command = "picca_export.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "data": str(
                self.paths.cf_fname(absorber, region, absorber2, region2).resolve()
            ),
            "out": str(output_filename),
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        in_files = [
            self.paths.cf_fname(absorber, region, absorber2, region2).resolve(),
        ]

        precommand = ""
        if self.config["correlations"].get("unblind", False):
            logger.warn("CORRELATIONS WILL BE UNBLINDED, BE CAREFUL.")
            precommand = f"picca_bookkeeper_unblind_correlations"

            cf_file = self.paths.cf_fname(
                absorber, region, absorber2, region2
            ).resolve()
            precommand += f" --cf {str(cf_file)}"
            if not self.config["fits"].get("no distortion", False):
                dmat_file = self.paths.dmat_fname(
                    absorber, region, absorber2, region2
                ).resolve()
                precommand += f" --dmat {str(dmat_file)}"
                in_files.append(dmat_file)
            if (not self.config["fits"].get("no metals", False)) and not self.config[
                "fits"
            ].get("vega metals", False):
                metal_file = self.paths.metal_fname(
                    absorber, region, absorber2, region2
                ).resolve()
                precommand += f" --metal-dmat {str(metal_file)}"
                in_files.append(metal_file)
        elif self.config["correlations"].get("unblind y1", False):
            logger.warn("Applying Y1 unblind.")
            args["unblind-y1"] = ""
            args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            precommand=precommand,
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=in_files,
            out_files=[
                output_filename,
            ],
        )

    def get_metal_tasker(
        self,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-forest metal distortion matrix
        measurements with picca.
    
        Args:
            region (str): Primary forest region. Options include 'lya', 'lyb'. Default: 'lya'.
            region2 (Optional[str]): Secondary forest region for cross-correlations.
                Defaults to None for auto-correlation with `region`.
            absorber (str): First absorber to use for correlations. (e.g., 'lya').
            absorber2 (Optional[str]): Second absorber to use for correlations.
                Defaults to None, meaning same absorber as `absorber`.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug: Whether to use debug options.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to run metal distortion matrix measurements.
    
        Raises:
            ValueError: If initialized in read mode or if config defaults have changed since last run.
        """        
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        job_name = f"metal_{absorber}{region}_{absorber2}{region2}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.metal_fname(absorber, region, absorber2, region2)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        # If metal matrices are provided, we just copy them into the bookkeeper
        # as if they were computed.
        copy_metal_matrix = self.paths.copied_correlation_file(
            "computed metals",
            absorber,
            region,
            absorber2,
            region2,
            output_filename.name,
        )
        if copy_metal_matrix is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_metal_matrix)

            return DummyTasker()

        if self.config["correlations"].get("fast metals", False):
            fast_metal = True
            command = "picca_fast_metal_dmat.py"
        else:
            fast_metal = False
            command = "picca_metal_dmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {}

        if fast_metal:
            args["in-attributes"] = str(
                self.paths.delta_attributes_file(region).resolve()
            )
            args["delta-dir"] = str(self.paths.deltas_path(region).resolve())

            if region2 != region:
                args["in-attributes2"] = str(
                    self.paths.delta_attributes_file(region2).resolve()
                )
        else:
            args["in-dir"] = str(self.paths.deltas_path(region).resolve())

            if region2 != region:
                args["in-dir2"] = str(self.paths.deltas_path(region2).resolve())

        args["out"] = str(output_filename)
        args["lambda-abs"] = absorber_igm[absorber]

        if absorber2 != absorber:
            args["lambda-abs2"] = absorber_igm[absorber2]

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args,
                dict(qos="debug", time="00:30:00"),
            )
            args = DictUtils.merge_dicts(args, dict(nspec=1000))

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_xcf_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        tracer: str = "qso",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-quasar correlations with picca.
    
        Args:
            region (str): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            absorber (str):  First absorber to use for correlations. Default: 'lya'.
            tracer (str): Tracer type for cross-correlation (e.g., 'qso'). Default: 'qso'.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug: Whether to use debug options.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to run forest-quasar cross-correlation.
    
        Raises:
            ValueError: If initialized in read mode or if config defaults have changed since last run.
        """     
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        job_name = f"xcf_{tracer}_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xcf_fname(absorber, region, tracer)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xcf_file = self.paths.copied_correlation_file(
            "computed correlations",
            absorber,
            region,
            None,
            None,
            output_filename.name,
            tracer=tracer,
        )
        if copy_xcf_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_xcf_file)

            return DummyTasker()

        command = "picca_xcf.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.get_tracer_catalog(tracer)

        args = {
            "in-dir": str(self.paths.deltas_path(region).resolve()),
            "drq": str(drq),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_xdmat_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        tracer: str = "qso",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-quasar distortion matrix
        measurements with picca.
    
        Args:
            region (str, optional): Absorption region to use ('lya', 'lyb'). Default is 'lya'.
            absorber (str): irst absorber to use for correlations. Default is 'lya'.
            tracer (str, optional): Tracer type for cross-correlation (e.g., 'qso'). Default is 'qso'.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug: Whether to use debug options.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to run the forest-quasar distortion matrix measurement.
    
        Raises:
            ValueError: If initialized in read mode or if config defaults have changed since last run.
        """        
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        job_name = f"xdmat_{tracer}_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xdmat_fname(absorber, region, tracer)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xdmat_file = self.paths.copied_correlation_file(
            "computed distortions",
            absorber,
            region,
            None,
            None,
            output_filename.name,
            tracer=tracer,
        )
        if copy_xdmat_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_xdmat_file)

            return DummyTasker()

        command = "picca_xdmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.get_tracer_catalog(tracer)

        args = {
            "in-dir": str(self.paths.deltas_path(region).resolve()),
            "drq": str(drq),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )
        output_filename.parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_xcf_exp_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        tracer: str = "qso",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        """
        Method to get a Tasker object to run forest-quasar correlation export with
        picca.
    
        Args:
            region (str, optional): Absorption region to use ('lya', 'lyb'). Default is 'lya'.
            absorber (str): First absorber to use for correlations. Default is 'lya'.
            tracer (str, optional): Tracer type for the cross-correlation. Default is 'qso'.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug: Whether to use debug options.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to export forest-quasar correlation data.
    
        Raises:
            ValueError: If initialized in read mode or if config defaults have changed since last run.
        """        
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)
        job_name = f"xcf_exp_{tracer}_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.exp_xcf_fname(absorber, region, tracer)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xcf_exp_file = self.paths.copied_correlation_file(
            "computed exports",
            absorber,
            region,
            None,
            None,
            output_filename.name,
            tracer=tracer,
        )
        if copy_xcf_exp_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_xcf_exp_file)

            return DummyTasker()

        command = "picca_export.py"

        command = "picca_export.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "data": str(self.paths.xcf_fname(absorber, region, tracer).resolve()),
            "out": str(output_filename),
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        in_files = [
            self.paths.xcf_fname(absorber, region, tracer),
        ]

        precommand = ""
        if self.config["correlations"].get("unblind", False):
            logger.warn("CORRELATIONS WILL BE UNBLINDED, BE CAREFUL.")
            precommand = f"picca_bookkeeper_unblind_correlations"

            xcf_file = self.paths.xcf_fname(absorber, region, tracer).resolve()
            precommand += f" --cf {str(xcf_file)}"
            if not self.config["fits"].get("no distortion", False):
                xdmat_file = self.paths.xdmat_fname(absorber, region).resolve()
                precommand += f" --dmat {str(xdmat_file)}"
                in_files.append(xdmat_file)
            if (not self.config["fits"].get("no metals", False)) and not self.config[
                "fits"
            ].get("vega metals", False):
                xmetal_file = self.paths.metal_fname(absorber, region).resolve()
                precommand += f" --metal-dmat {str(xmetal_file)}"
                in_files.append(xmetal_file)
        elif self.config["correlations"].get("unblind y1", False):
            logger.warn("Applying Y1 unblind.")
            args["unblind-y1"] = ""
            args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            precommand=precommand,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=in_files,
            out_files=[
                output_filename,
            ],
        )

    def get_xmetal_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        tracer: str = "qso",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run forest-quasar metal distortion matrix
                measurements with picca.    
                
        Args:
            region (str, optional): Absorption region to use ('lya', 'lyb'). Default is 'lya'.
            absorber (str): First absorber to use for correlations. Default is 'lya'.
            tracer (str, optional): Tracer type for the cross-correlation. Default is 'qso'.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            debug: Whether to use debug options.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker object to run forest-quasar metal distortion matrix computation.
    
        Raises:
            ValueError: If initialized in read mode or if config defaults have changed since last run.
        """ 
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)
        job_name = f"xmetal_{tracer}_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xmetal_fname(absorber, region, tracer)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_metal_matrix = self.paths.copied_correlation_file(
            "computed metals",
            absorber,
            region,
            None,
            None,
            output_filename.name,
            tracer=tracer,
        )
        if copy_metal_matrix is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            output_filename.symlink_to(copy_metal_matrix)

            return DummyTasker()

        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        if self.config["correlations"].get("fast metals", False):
            fast_metal = True
            command = "picca_fast_metal_xdmat.py"
        else:
            fast_metal = False
            command = "picca_metal_xdmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            command=command,
            region=region,
            absorber=absorber,
            tracer=tracer,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.get_tracer_catalog(tracer)

        args = {}
        if fast_metal:
            args["in-attributes"] = str(
                self.paths.delta_attributes_file(region).resolve()
            )
            args["delta-dir"] = str(self.paths.deltas_path(region).resolve())
        else:
            args["in-dir"] = str(self.paths.deltas_path(region).resolve())

        args["drq"] = str(drq)
        args["out"] = str(output_filename)
        args["lambda-abs"] = absorber_igm[absorber]

        args = DictUtils.merge_dicts(args, updated_extra_args)

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_files=[
                output_filename,
            ],
        )

    def get_compute_zeff_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """
        Create a Tasker to compute the effective redshift using 
        `picca_bookkeeper_correct_config_zeff`.
    
        Args:
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker | DummyTasker: Tasker object to submit the job, or 
                    DummyTasker if the task is skipped.
        """        
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        job_name = "correct_config_zeff"

        command = "picca_bookkeeper_correct_config_zeff"

        updated_system = self.generate_system_arg(system)
        if self.check_existing_output_file(
            self.paths.fit_computed_params_out(),
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        input_files = self.write_fit_configuration()
        # Remove dependency on itself
        if self.paths.fit_computed_params_out() in input_files:
            input_files.remove(self.paths.fit_computed_params_out())

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.fits_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.fits_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args,
            updated_slurm_header_extra_args,
        )

        args = {
            "": self.paths.config_file,
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_files=[
                self.paths.fit_computed_params_out(),
            ],
        )

    def get_fit_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """
        Method to get a Tasker object to run vega with correlation data.
    
        Args:
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker | DummyTasker: Tasker object to submit the job, or 
            DummyTasker if the task is skipped.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        job_name = "vega_fit"
        if self.check_existing_output_file(
            self.paths.fit_out_fname(),
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        input_files = self.write_fit_configuration()

        # Now slurm args
        command = "run_vega.py"
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.fits_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.fits_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args,
            updated_slurm_header_extra_args,
        )

        return get_Tasker(updated_system)(
            command=command,
            command_args={"": str(self.paths.fit_main_fname().resolve())},
            packages=["vega"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_files=[
                self.paths.fit_out_fname(),
            ],
        )

    def get_sampler_tasker(
        self,
        auto_correlations: List[str] = [],
        cross_correlations: List[str] = [],
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """
        Method to get a Tasker object to run vega sampler with correlation data.    
        
        Args:
            auto_correlations (List[str]): List of auto-correlation names to include.
            cross_correlations (List[str]): List of cross-correlation names to include.
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker | DummyTasker: A Tasker object to submit the job, or a DummyTasker if
            the task is skipped due to existing output or config issues.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values\\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        job_name = "vega_sampler"

        updated_system = self.generate_system_arg(system)
        if self.check_existing_output_file(
            self.paths.sampler_out_path() / "jobidfile",
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        input_files = self.write_fit_configuration()

        command = "run_vega_mpi.py"
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.fits_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.fits_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args,
            updated_slurm_header_extra_args,
        )

        if self.config["fits"].get("sampler environment", None) is None:
            environment = self.config["general"]["conda environment"]
        else:
            environment = self.config["fits"]["sampler environment"]

        return get_Tasker(updated_system)(
            command=command,
            command_args={"": str(self.paths.fit_main_fname().resolve())},
            packages=["vega"],
            slurm_header_args=slurm_header_args,
            environment=environment,
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_files=[
                self.paths.sampler_out_path() / "jobidfile",
            ],
        )

    def generate_fit_configuration(self) -> Dict:
        """
        Method to generate fit configuration args to be combined with input YAML file.
    
        This method constructs and validates a nested dictionary of fit options to be passed to
        vega for correlation function fitting. It checks types for key fit flags and cut ranges,
        and conditionally sets parameters and sampling flags based on booleans in the config
        (e.g., disabling BAO, HCD, metals, sky systematics, or QSO radiation effects).
    
        It also applies rmin/rmax cuts for auto- and cross-correlations if defined.
    
        Returns:
            Dict: A complete dictionary with validated and modified fit configuration,
                  stored under `config["fits"]["extra args"]`.
                  
        Raises:
            ValueError: If required fields in the config are not of the expected type.
        """
        
        config = DictUtils.merge_dicts(
            {
                "fits": {
                    "extra args": {},
                },
            },
            self.config,
        )

        for field in "no bao", "no hcd", "no metals", "no sky", "no qso rad":
            if not isinstance(config["fits"][field], bool):
                raise ValueError(f"Fit config {field} should be boolean.")

        for field in "rmin cf", "rmax cf", "rmin xcf", "rmax xcf":
            if not isinstance(config["fits"][field], (np.integer, type(None), int)):
                raise ValueError(
                    f"Fit config {field} should be integer. "
                    f"Type: {type(config['fits'][field])}"
                )

        args = DictUtils.merge_dicts(
            {
                "vega_auto": {
                    "general": {
                        "data": {},
                        "cuts": {},
                        "model": {},
                        "metals": {},
                    }
                },
                "vega_cross": {
                    "general": {
                        "data": {},
                        "cuts": {},
                        "model": {},
                        "metals": {},
                    }
                },
                "vega_main": {
                    "general": {
                        "data sets": {},
                        "cosmo-fit type": {},
                        "fiducial": {},
                        "control": {},
                        "output": {},
                        "Polychord": {},
                        "sample": {},
                        "parameters": {},
                    }
                },
            },
            config["fits"]["extra args"],
        )

        if config["fits"]["no bao"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_main": {
                        "all": {
                            "sample": {
                                "ap": "$",
                                "at": "$",
                            },
                            "parameters": {
                                "bao_amp": 0,
                            },
                        }
                    }
                },
            )

        if config["fits"]["no hcd"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_main": {
                        "all": {
                            "sample": {
                                "bias_hcd": "$",
                                "beta_hcd": "$",
                            }
                        }
                    },
                    "vega_auto": {
                        "all": {
                            "model": {
                                "model-hcd": "$",
                            }
                        }
                    },
                    "vega_cross": {
                        "all": {
                            "model": {
                                "model-hcd": "$",
                            }
                        }
                    },
                },
            )

        if config["fits"]["no metals"]:
            metal_dict = {
                metal: "$"
                for metal in (
                    "SiII(1190)",
                    "SiII(1193)",
                    "SiII(1260)",
                    "SiIII(1207)",
                    "CIV(eff)",
                )
            }

            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_main": {"all": {"sample": metal_dict}},
                    "vega_auto": {
                        "all": {
                            "metals": "$",
                        }
                    },
                    "vega_cross": {
                        "all": {
                            "metals": "$",
                        }
                    },
                },
            )

        if config["fits"]["no sky"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_main": {
                        "all": {
                            "sample": {
                                "desi_inst_sys_amp": "$",
                            },
                            "parameters": {
                                "desi_inst_sys_amp": 0,
                            },
                        }
                    }
                },
            )

        if config["fits"]["no qso rad"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_main": {
                        "all": {
                            "sample": {
                                "qso_rad_strength": "$",
                            },
                            "parameters": {
                                "qso_rad_strength": 0,
                            },
                        }
                    }
                },
            )

        if config["fits"]["rmin cf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_auto": {
                        "all": {
                            "cuts": {
                                "r-min": str(config["fits"]["rmin cf"]),
                            }
                        }
                    }
                },
            )
        if config["fits"]["rmax cf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_auto": {
                        "all": {
                            "cuts": {
                                "r-max": str(config["fits"]["rmax cf"]),
                            }
                        }
                    }
                },
            )
        if config["fits"]["rmin xcf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_cross": {
                        "all": {
                            "cuts": {
                                "r-min": str(config["fits"]["rmin xcf"]),
                            }
                        }
                    }
                },
            )
        if config["fits"]["rmax xcf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_cross": {
                        "all": {
                            "cuts": {
                                "r-max": str(config["fits"]["rmax xcf"]),
                            }
                        }
                    }
                },
            )

        config["fits"]["extra args"] = args
        return config

    def write_fit_configuration(self) -> List[Path]:
        """
        Generate and write full Vega configuration `.ini` files for auto, cross, and main fits, 
        using the main bookkeeper config YAML.
    
        This method uses the current bookkeeper config to:
        - Generate correlation-specific configuration arguments (via `generate_fit_configuration`)
        - Build `.ini` files for each auto- and cross-correlation, including optional metadata such as:
            - Metal models (from precomputed files or Vega internal models)
            - Distortion matrices
            - Unblinding flags
        - Assemble the final `vega_main.ini` file with paths to all sub-configs and output locations
        - Add dependencies for sampler input, covariance matrix, or derived parameters if required
    
        Returns:
            List[Path]: All input files (e.g., exports, distortions, metal files) used by the Tasker 
            instance to correctly set up dependencies.
    
        Raises:
            ValueError: If any required format or validation fails during file generation.
        """

        ini_files = []
        input_files = []

        config = self.generate_fit_configuration()

        if config["fits"].get("auto correlations", None) not in (None, ""):
            auto_correlations = config["fits"]["auto correlations"].split(" ")
        else:
            auto_correlations = []
        if config["fits"].get("cross correlations", None) not in (None, ""):
            cross_correlations = config["fits"]["cross correlations"].split(" ")
        else:
            cross_correlations = []

        export_files_auto = []

        # Set because there can be repeated values.
        for auto_correlation in auto_correlations:
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            export_file = self.paths.exp_cf_fname(
                absorber, region, absorber2, region2
            ).resolve()
            export_files_auto.append(export_file)

            metals_file = self.paths.metal_fname(
                absorber, region, absorber2, region2
            ).resolve()
            distortion_file = self.paths.dmat_fname(
                absorber, region, absorber2, region2
            ).resolve()

            args = DictUtils.merge_dicts(
                config,
                {
                    "fits": {
                        "extra args": {
                            "vega_auto": {
                                "general": {
                                    "data": {
                                        "name": f"{absorber}{region}x{absorber2}{region2}",
                                        "filename": export_file,
                                        "tracer1": absorber_igm[absorber],
                                        "tracer2": absorber_igm[absorber2],
                                        "tracer1-type": "continuous",
                                        "tracer2-type": "continuous",
                                    },
                                    # "metals": {},
                                }
                            }
                        }
                    }
                },
            )
            if not config["fits"].get("no metals", False):
                if config["fits"].get("vega metals", False):
                    args = DictUtils.merge_dicts(
                        args,
                        {
                            "fits": {
                                "extra args": {
                                    "vega_auto": {
                                        "general": {
                                            "data": {
                                                "weights-tracer1": self.paths.delta_attributes_file(
                                                    region=region
                                                ).resolve(),
                                                "weights-tracer2": self.paths.delta_attributes_file(
                                                    region=region2
                                                ).resolve(),
                                            },
                                            "model": {
                                                "new_metals": True,
                                            },
                                            "metals": {},
                                        }
                                    }
                                }
                            }
                        },
                    )
                else:  # Use metals from results
                    args = DictUtils.merge_dicts(
                        args,
                        {
                            "fits": {
                                "extra args": {
                                    "vega_auto": {
                                        "general": {
                                            "metals": {
                                                "filename": metals_file,
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    )
            input_files.append(export_file)

            if not config["fits"].get("no distortion", False):
                args["fits"]["extra args"]["vega_auto"]["general"]["data"][
                    "distortion-file"
                ] = distortion_file
                input_files.append(distortion_file)

            # Check if we need to unblind
            if self.config["correlations"].get("unblind y1", False):
                args["fits"]["extra args"]["vega_auto"]["general"]["data"][
                    "unblind-y1"
                ] = True

            vega_args = self.generate_extra_args(
                config=args,
                section="fits",
                command="vega_auto.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
                region2=region2,
                absorber2=absorber2,
            )

            filename = self.paths.fit_auto_fname(absorber, region, absorber2, region2)
            self.write_ini(vega_args, filename)
            ini_files.append(str(filename))

            if (not vega_args.get("no metals", True)) and not args["fits"].get(
                "vega metals", False
            ):
                input_files.append(metals_file)

        export_files_cross = []

        # Set because there can be repeated values.
        for cross_correlation in cross_correlations:
            tracer, absorber, region = cross_correlation.replace("-", ".").split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            export_file = self.paths.exp_xcf_fname(absorber, region, tracer).resolve()
            export_files_cross.append(export_file)

            metals_file = self.paths.xmetal_fname(absorber, region, tracer).resolve()
            distortion_file = self.paths.xdmat_fname(absorber, region, tracer).resolve()

            args = DictUtils.merge_dicts(
                config,
                {
                    "fits": {
                        "extra args": {
                            "vega_cross": {
                                "general": {
                                    "data": {
                                        "name": f"{absorber}{region}x{tracer}",
                                        "tracer1": absorber_igm[absorber],
                                        "tracer2": tracer.upper(),
                                        "tracer1-type": "continuous",
                                        "tracer2-type": "discrete",
                                        "filename": export_file,
                                    },
                                }
                            }
                        }
                    }
                },
            )

            if not config["fits"].get("no metals", False):
                if config["fits"].get("vega metals", False):
                    args = DictUtils.merge_dicts(
                        args,
                        {
                            "fits": {
                                "extra args": {
                                    "vega_cross": {
                                        "general": {
                                            "data": {
                                                "weights-tracer1": self.paths.delta_attributes_file(
                                                    region=region
                                                ).resolve(),
                                                "weights-tracer2": self.paths.get_tracer_catalog(
                                                    tracer
                                                ).resolve(),
                                            },
                                            "model": {
                                                "new_metals": True,
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    )
                else:
                    args = DictUtils.merge_dicts(
                        args,
                        {
                            "fits": {
                                "extra args": {
                                    "vega_cross": {
                                        "general": {
                                            "metals": {
                                                "filename": metals_file,
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    )
            input_files.append(export_file)

            if not self.config["fits"].get("no distortion", False):
                args["fits"]["extra args"]["vega_cross"]["general"]["data"][
                    "distortion-file"
                ] = distortion_file
                input_files.append(distortion_file)

            # Check if we need to unblind
            if self.config["correlations"].get("unblind y1", False):
                args["fits"]["extra args"]["vega_cross"]["general"]["data"][
                    "unblind-y1"
                ] = True

            vega_args = self.generate_extra_args(
                config=args,
                section="fits",
                command="vega_cross.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
                tracer=tracer,
            )

            filename = self.paths.fit_cross_fname(absorber, region, tracer)
            self.write_ini(vega_args, filename)
            ini_files.append(str(filename))

            if (not vega_args.get("no metals", True)) and not args["fits"].get(
                "vega metals", False
            ):
                input_files.append(metals_file)

        # Now the main file
        args = DictUtils.merge_dicts(
            config,
            {
                "fits": {
                    "extra args": {
                        "vega_main": {
                            "general": {
                                "data sets": {
                                    "ini files": " ".join(ini_files),
                                },
                                "Polychord": {
                                    "path": self.paths.sampler_out_path(),
                                },
                                "output": {"filename": self.paths.fit_out_fname()},
                            }
                        }
                    }
                }
            },
        )

        # Check if covariance is needed, and add it
        if self.config["fits"].get("compute covariance", False):
            args["fits"]["extra args"]["vega_main"]["general"]["data sets"][
                "global-cov-file"
            ] = self.paths.covariance_file().resolve()
            input_files.append(self.paths.covariance_file().resolve())

        # Check precomputed zeff and others.
        if self.config["fits"].get("compute zeff", False):
            if self.paths.fit_computed_params_out().is_file():
                if self.paths.fit_computed_params_out().stat().st_size > 19:
                    # If the output file is filled, then I update config
                    args = DictUtils.merge_dicts(
                        args,
                        yaml.safe_load(
                            self.paths.fit_computed_params_out().read_text()
                        ),
                    )
                else:
                    # Otherwise it generates a dependency
                    input_files.append(self.paths.fit_computed_params_out())

        vega_args = self.generate_extra_args(
            config=args,
            section="fits",
            command="vega_main.py",  # The .py needed to make use of same function
        )

        filename = self.paths.fit_main_fname()
        self.write_ini(vega_args, filename)

        return input_files

    def get_covariance_matrix_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run covariance matrix.

        Args:
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            overwrite: Overwrite files in destination.
            skip_sent: Skip the run if fit output already present.

        Returns:
            Tasker: Tasker object to run covariance matrix.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values)."
            )

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        job_name = "write_full_covariance"

        output_filename = self.paths.covariance_file_unsmoothed()
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_covariance_file = self.paths.copied_covariance_file(smoothed=False)
        if copy_covariance_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            output_filename.symlink_to(copy_covariance_file)

            return DummyTasker()

        input_files = []
        if self.config["fits"].get("auto correlations", None) not in (None, ""):
            auto_correlations = self.config["fits"]["auto correlations"].split(" ")
        else:
            auto_correlations = []
        if self.config["fits"].get("cross correlations", None) not in (None, ""):
            cross_correlations = self.config["fits"]["cross correlations"].split(" ")
        else:
            cross_correlations = []

        args = {}
        for auto_correlation in auto_correlations:
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            input_files.append(
                self.paths.cf_fname(absorber, region, absorber2, region2).resolve()
            )

            args[f"{region}-{region2}"] = str(
                self.paths.cf_fname("lya", region, "lya", region2).resolve()
            )

        for cross_correlation in cross_correlations:
            (
                tracer,
                absorber,
                region,
            ) = cross_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            input_files.append(self.paths.xcf_fname(absorber, region, tracer).resolve())

            args[f"{region}-qso"] = str(
                self.paths.xcf_fname("lya", region, tracer).resolve()
            )

        # Now slurm args
        command = "write_full_covariance.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.fits_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.fits_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args,
            updated_slurm_header_extra_args,
        )

        args["output"] = str(self.paths.covariance_file_unsmoothed())

        args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command="/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-tests/correlations/scripts/write_full_covariance_matrix_flex_size.py",
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_files=[
                self.paths.covariance_file_unsmoothed(),
            ],
        )

    def get_smooth_covariance_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """
        Method to get a Tasker object to run smooth covariance.
    
        This task reads the unsmoothed covariance matrix and outputs a smoothed version,
        using configuration from the first available auto or cross-correlation. If the 
        smoothed file already exists or has been copied, a DummyTasker is returned.
    
        Args:
            system (Optional[str]): Execution system or shell to use  (e.g., 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or computer shell). 
                Defaults to config value if None.
            wait_for (Optional[Tasker | ChainedTasker | int | List[int]]): Job(s) to wait for before starting
                (default: None). Could be a Tasker object or a slurm jobid (int).
            overwrite (bool): Overwrite existing output files if True (default: False).
            skip_sent (bool): Skip and return DummyTasker if the output already exists (default: False).
    
        Returns:
            Tasker: Configured Tasker to execute the smoothing job, or DummyTasker
            if no execution is needed.
    
        Raises:
            ValueError: If bookkeeper is in read-only mode or config defaults differ.
        """
        
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                "values)."
            )

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        job_name = "smooth_covariance"
        output_filename = self.paths.covariance_file_smoothed()
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_covariance_file = self.paths.copied_covariance_file(smoothed=True)
        if copy_covariance_file is not None:
            output_filename.unlink(missing_ok=True)
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            output_filename.symlink_to(copy_covariance_file)

            return DummyTasker()

        input_files = [
            self.paths.covariance_file_unsmoothed().resolve(),
        ]

        # Now slurm args
        command = "smooth_covariance.py"

        input_files = []
        if self.config["fits"].get("auto correlations", None) not in (None, ""):
            auto_correlations = self.config["fits"]["auto correlations"].split(" ")
        else:
            auto_correlations = []
        if self.config["fits"].get("cross correlations", None) not in (None, ""):
            cross_correlations = self.config["fits"]["cross correlations"].split(" ")
        else:
            cross_correlations = []

        args = {}
        for auto_correlation in auto_correlations:
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            cf_args = self.generate_extra_args(
                config=self.config,
                section="correlations",
                command="picca_cf.py",
                region=region,
                absorber=absorber,
                region2=region2,
                absorber2=absorber2,
            )

            for range_ in ["rt-min", "rt-max", "rp-min", "rp-max", "np", "nt"]:
                if range_ in cf_args:
                    args[f"{range_}-auto"] = cf_args[range_]

            # This ugly break indicates that we cannot pass multiple ranges, only 
            # the first one...
            break

        for cross_correlation in cross_correlations:
            (
                tracer,
                absorber,
                region,
            ) = cross_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            xcf_args = self.generate_extra_args(
                config=self.config,
                section="correlations",
                command="picca_xcf.py",
                region=region,
                absorber=absorber,
                tracer=tracer,
            )

            for range_ in ["rt-min", "rt-max", "rp-min", "rp-max", "np", "nt"]:
                if range_ in xcf_args:
                    args[f"{range_}-cross"] = xcf_args[range_]

            # This ugly break indicates that we cannot pass multiple ranges, only 
            # the first one...
            break

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            command=command,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.fits_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.fits_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args,
            updated_slurm_header_extra_args,
        )

        
        args["input-cov"] = str(self.paths.covariance_file_unsmoothed())
        args["output-cov"] = str(self.paths.covariance_file_smoothed())
        
        args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command="/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-tests/correlations/scripts/write_smooth_covariance_flex_size.py",
            command_args=args,
            packages=["picca"],
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_files=[
                self.paths.covariance_file_smoothed(),
            ],
        )

    def check_existing_output_file(
        self, file: Path, job_name: str, skip_sent: bool, overwrite: bool, system: str
    ) -> bool:
        """
        Method to check the status of an output file.
    
        This is primarily used to avoid re-running tasks that have already completed 
        or are in progress, based on the presence and contents of the output file.
    
        Args:
            file (Path): Path to the output file to check.
            job_name (str): Name of the job (used for logging).
            skip_sent (bool): If True, skip jobs that were already sent or are still running.
            overwrite (bool): Whether existing runs should be overwritten.
            system (str): System where jobs are being run (used to query job status).
    
        Returns:
            bool: True if the job should be skipped (file exists and skip_sent is True).
                  False if the job should proceed (file does not exits, or run to be overwritten).
    
        Raises:
            FileExistsError: If the file exists, overwrite is False, and skip_sent is False.
    
        Notes:
            - If the file exists but is very small (<40 bytes), it is assumed to contain a job ID.
              The job status is then checked via the system's scheduler.
            - If the job is no longer active, it will not be skipped.
        """      
        if not file.is_file():
            return False
        else:
            if overwrite:
                return False
            elif skip_sent:
                size = file.stat().st_size
                if size < 40:
                    jobid = int(file.read_text().splitlines()[0])
                    status = get_Tasker(system).get_jobid_status(jobid)
                    logger.debug(f"{job_name} status: {status}")
                    if status not in (
                        "COMPLETED",
                        "RUNNING",
                        "PENDING",
                        "REQUEUED",
                        "SUSPENDED",
                    ):
                        return False
                logger.info(f"{job_name}: skipping already run:\n\t{str(file)}")
                return True
            else:
                raise FileExistsError(
                    "Destination file already exists, run with overwrite option or "
                    "skipping completed runs to continue",
                    file,
                )


class PathBuilder:
    """
    Class to define and manage filesystem paths according to bookkeeper conventions.

    Handles the construction and validation of standardized paths for all input and output
    files required by the picca_bookkeeper workflow, including locations for data, deltas,
    correlations, fits, and configuration files. Automates creation and checking of directories,
    provides utilities to resolve paths for specific analysis products, and implements logic for
    linking or copying results from previous runs or external sources as specified in the config.

    Attributes:
        config (dict / configparser.ConfigParser): Bookkeeper configuration used to build all relevant paths.

    Usage:
        paths = PathBuilder(config)
        deltas_dir = paths.deltas_path(region="lya")
        correlation_file = paths.cf_fname("lya", "lya", "lya", "lya")
        paths.check_run_directories()
        catalog_path = paths.catalog

    Main Features:
        - Constructs paths for all data products based on config.
        - Validates existence of critical files and directories; raises errors if misconfigured.
        - Supports both absolute and relative paths, expanding and checking as needed.
        - Allows access to specialized outputs (deltas, correlations, fits, masks, metals, covariance, etc.).
        - Handles linking/copying of external results into the workflow as configured.
        - Provides static comparison utilities for config validation.
        - Automates directory creation for each analysis stage.
    """

    def __init__(self, config: Dict):
        """
        Initialize the PathBuilder with a given configuration.

        Args:
            config (dict): Configuration dictionary used to resolve paths.
        """
        self.config = config

    @property
    def healpix_data(self) -> Path:
        """
        Get the path to the healpix data directory.

        Returns:
            Path: Path to the healpix data.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        healpix_data = Path(self.config["data"]["healpix data"])

        if not healpix_data.is_dir():
            raise FileNotFoundError("Invalid healpix data in config", str(healpix_data))
        else:
            return healpix_data

    @property
    def run_path(self) -> Path:
        """
        Get the root path for the bookkeeper run.

        Returns:
            Path: Root path where all outputs and intermediate files are stored.

        Raises:
            ValueError: If the path is not specified in the config.
        """
        bookkeeper_dir = self.config.get("data", dict()).get("bookkeeper dir", None)

        if bookkeeper_dir is None:
            raise ValueError("Invalid bookkeeper dir in config")
        else:
            return Path(bookkeeper_dir)

    @property
    def delta_extraction_path(self) -> Path:
        """
        Get the path to the delta extraction directory.

        Returns:
            Path: Path to the delta results, either from original path or run directory.
        """
        if self.config.get("delta extraction", dict()).get("original path") is not None:
            return Path(self.config["delta extraction"]["original path"]) / "deltas"
        else:
            return self.run_path / "deltas"

    @property
    def continuum_fitting_mask(self) -> Path:
        """
        Get the path to the continuum fitting mask file.

        Returns:
            Path: Path to `continuum_fitting_mask.txt`, used in continuum fitting.
        """        return self.delta_extraction_path / "configs" / "continuum_fitting_mask.txt"

    @property
    def correlations_path(self) -> Path:
        """
        Get the path to the correlation output directory.

        Returns:
            Path: Path to correlation results, either from original path or run directory.
        """
        if self.config.get("correlations", dict()).get("original path") is not None:
            return Path(self.config["correlations"]["original path"]) / "correlations"
        else:
            return self.run_path / "correlations"

    @property
    def fits_path(self) -> Path:
        """
        Get the path to the fits output directory.

        Returns:
            Path: Path to where fit results are stored.
        """
        return self.run_path / "fits"

    @property
    def config_file(self) -> Path:
       """
        Get the default path to the bookkeeper configuration file inside bookkeeper.

        Returns:
            Path: Path to `bookkeeper_config.yaml` inside the run directory.
        """
        return self.run_path / "configs" / "bookkeeper_config.yaml"

    @property
    def defaults_file(self) -> Path:
        """
        Get the path to the defaults configuration file inside bookkeeper.

        Returns:
            Path: Path to `defaults.yaml` in the configs directory.
        """
        return self.config_file.parent / "defaults.yaml"

    def get_catalog_from_field(self, field: str) -> Path:
        """
        Get the path to a catalog file from a catalog name in a given config field.

        Supports resolution of catalogs for standard quasars, DLAs, or BALs. Validates
        existence of the specified catalog file, expands the full path if only the filename
        is given.

        Args:
            field (str): Catalog field to resolve. Options are:
                - "catalog": main QSO catalog
                - "dla": Damped Lyman-Alpha catalog
                - "bal": BAL masking catalog

        Returns:
            Path: Path to the resolved catalog file to be used.

        Raises:
            FileNotFoundError: If the specified catalog does not exist.
        """
        if field == "dla":
            dla = self.config["delta extraction"].get("dla", None)
            if dla is None:
                return Path("None")
            elif Path(dla).is_file():
                catalog = Path(dla)
            else:
                raise FileNotFoundError("Couldn't find valid DLA catalog")
        elif field == "bal":
            bal = self.config["delta extraction"].get("bal", None)
            if bal not in (True, None) and Path(bal).is_file():
                catalog = Path(bal)
            else:
                catalog = self.get_catalog_from_field("catalog")
        else:
            if (self.config["data"].get("catalog", None) not in ("", None)) and Path(
                self.config["data"].get("catalog", "")
            ).is_file():
                catalog = Path(self.config["data"].get("catalog", ""))
            else:
                raise FileNotFoundError(
                    f"Couldn't find valid catalog for field {field}",
                    self.config["data"].get("catalog", ""),
                )

        return catalog

    @property
    def catalog(self) -> Path:
        """
        Get the main QSO catalog used for delta extraction.

        Returns:
            Path: Path to the main catalog file.
        """        
        return self.get_catalog_from_field("catalog")

    @property
    def catalog_dla(self) -> Path:
        """
        Get the catalog used for masking DLAs in delta extraction.

        Returns:
            Path: Path to the DLA catalog file.
        """
        return self.get_catalog_from_field("dla")

    @property
    def catalog_bal(self) -> Path:
        """
        Get the catalog used for BAL masking in delta extraction.

        Returns:
            Path: Path to the BAL catalog file.
        """
        return self.get_catalog_from_field("bal")

    def get_tracer_catalog(self, tracer: str = "qso") -> Path:
        """
        Get a tracer catalog used in cross-correlation.

        If the tracer-specific catalog exists, it is returned; otherwise, the default
        QSO catalog is used as a fallback.

        Args:
            tracer (str): Name of the tracer, e.g., "qso", "lya", etc.

        Returns:
            Path: Path to the resolved tracer catalog file.
        """        
        candidate = (
            self.config["correlations"].get("tracer catalogs", dict()).get(tracer, None)
        )

        if candidate not in ("", None) and Path(candidate).is_file():
            return Path(candidate)
        else:
            return self.get_catalog_from_field("catalog")

    @staticmethod
    def compare_configs(
        config1: Dict,
        config2: Dict,
        section: Optional[str] = None,
        ignore_fields: List[str] = [],
    ) -> Dict:
        """
        Compare two configurations and return their differences.

        Optionally limits comparison to a specific section and ignores given fields.

        Args:
            config1 (Dict): First config dictionary.
            config2 (Dict): Second config dictionary.
            section (Optional[str]): Config section to compare.
            ignore_fields (List[str]): List of fields to ignore during comparison.

        Returns:
            Dict: Dictionary of differences between configs.
        """
        if section is not None:
            config1 = config1[section]
            config2 = config2[section]

        for field in ignore_fields:
            if field in config1:
                config1.pop(field)
            if field in config2:
                config2.pop(field)

        return DictUtils.remove_empty(
            DictUtils.diff_dicts(
                config1,
                config2,
            )
        )

    def check_run_directories(self) -> None:
        """
        Ensure base directories exist in the run path.

        Currently ensures the existence of the `configs/` directory.
        """
        for folder in ("configs",):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)

    def check_delta_directories(self) -> None:
        """
        Ensure necessary subdirectories exist in the delta extraction path.

        Creates `scripts`, `logs`, `results`, and `configs` if they do not exist.
        """
        for folder in ("scripts", "logs", "results", "configs"):
            (self.delta_extraction_path / folder).mkdir(exist_ok=True, parents=True)

    def check_correlation_directories(self) -> None:
        """
        Ensure necessary subdirectories exist in the correlations path.

        Creates `scripts`, `results`, `logs`, and `configs` if they do not exist.
        """
        for folder in ("scripts", "results", "logs", "configs"):
            (self.correlations_path / folder).mkdir(exist_ok=True, parents=True)

    def check_fit_directories(self) -> None:
        """
        Ensure necessary subdirectories exist in the fits path.

        Creates `scripts`, `results`, `logs`, `configs`, and nested `results/sampler` if missing.
        """
        for folder in ("scripts", "results", "logs", "configs", "results/sampler"):
            (self.fits_path / folder).mkdir(exist_ok=True, parents=True)

    def deltas_path(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:       
        """
        Get the path to the deltas output directory.

        Args:
            region (Optional[str]): Sky region label used (e.g., 'lya', 'lyb'). 
                                    Required if `calib_step` is not given.
            calib_step (Optional[int]): Calibration step (1 or 2). If provided, region is ignored.

        Returns:
            Path: Path to the directory containing delta files.

        Raises:
            ValueError: If neither `calib_step` nor valid `region` is provided.
        """        
        if calib_step is not None:
            return (
                self.delta_extraction_path
                / "results"
                / f"calibration_{calib_step}"
                / "Delta"
            )
        else:
            if region is None:
                raise ValueError("Invalid region provided: ", region)
            region = Bookkeeper.validate_region(region)
            return self.delta_extraction_path / "results" / region / "Delta"

    def deltas_log_path(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """
        Get the path to the log directory for delta outputs.

        Args:
            region (Optional[str]): Sky region used (required if `calib_step` is not set).
            calib_step (Optional[int]): Calibration step (1 or 2).

        Returns:
            Path: Path to the log directory associated with deltas.
        """
        deltas_path = self.deltas_path(region, calib_step)
        return deltas_path.parent / "Log"

    def delta_attributes_file(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """
        Get the path to the delta attributes FITS file.

        Args:
            region (Optional[str]): Sky region (required if `calib_step` is not set).
            calib_step (Optional[int]): Calibration step.

        Returns:
            Path: Path to `delta_attributes.fits.gz`.
        """
        return (
            self.deltas_log_path(region=region, calib_step=calib_step)
            / "delta_attributes.fits.gz"
        )

    def copied_calib_attributes(self, step: int) -> Path | None:
        """
        Get the path to a precomputed calibration attributes file.

        Reads from the bookkeeper config and checks that the file exists.
        Logs whether the file is used or computed.

        Args:
            step (int): Current calibration step (1 or 2).

        Returns:
            Path | None: Path to attributes file if found, else None.

        Raises:
            FileNotFoundError: If a file is listed but does not exist.
        """
        calibration_data = self.config["delta extraction"].get("calibration data", None)

        if calibration_data is None:
            attributes = None
        else:
            attributes = calibration_data.get(f"step {step}", None)

        if attributes is not None:
            if not Path(attributes).is_file():
                raise FileNotFoundError(
                    f"calibration step: {step}: Invalid attributes file provided."
                )
            logger.info(
                f"calibration step {step}: "
                f"Using attributes from file:\n\t{str(attributes)}"
            )
        else:
            logger.info(
                f"calibration step {step}: "
                f"No attributes file provided, it will be computed."
            )
        return attributes

    def copied_deltas_files(self, region: str) -> List[Path] | None:
        """
        Get paths to precomputed deltas and attributes for a region.

        Checks for exiting delta files if specified in bookkeeper config file. 
        Looks for region-specific or general configuration, checks file existence,
        and logs whether files will be used or recomputed.

        Args:
            region (str): Sky region name to look for in config file (e.g., 'lya').

        Returns:
            List[Path] | None: Paths to [Delta/, delta_attributes.fits.gz], or None if not found.

        Raises:
            FileNotFoundError: If provided paths exist in config but files are missing.
        """
        if (
            self.config["delta extraction"]
            .get("computed deltas", dict())
            .get(region, None)
            is not None
        ):
            parent = Path(self.config["delta extraction"]["computed deltas"][region])
            deltas = parent / "Delta"
            attributes = parent / "Log/delta_attributes.fits.gz"

            if not deltas.is_dir():
                raise FileNotFoundError(
                    f"{region}: Invalid deltas directory provided", deltas
                )
            if not attributes.is_file():
                raise FileNotFoundError(
                    f"{region}: Invalid attributes file provided", attributes
                )

        elif (
            self.config["delta extraction"]
            .get("computed deltas", dict())
            .get("general", None)
            is not None
        ):
            parent = Path(self.config["delta extraction"]["computed deltas"]["general"])
            deltas = parent / region / "Delta"
            attributes = parent / region / "Log/delta_attributes.fits.gz"

            if not deltas.is_dir() or not attributes.is_file():
                logger.info(
                    f"{region}: no files provided to use, deltas will be computed"
                )

                return None
        else:
            logger.info(f"{region}: no files provided to use, deltas will be computed")

            return None

        logger.info(f"{region}: Using deltas from file:\n\t{str(deltas)}")
        logger.info(f"{region}: Using attributes from file:\n\t{str(attributes)}")
        return [deltas, attributes]

    def copied_correlation_file(
        self,
        subsection: str,
        absorber: str,
        region: str,
        absorber2: Optional[str],
        region2: Optional[str],
        filename: Optional[str],
        tracer: Optional[str] = "qso",
    ) -> Path | None:
        """
        Get the path to a precomputed correlation file if configured.

        Tries to retrieve correlation from bookkeeper config using absorber/region keys,
        optionally falling back to a general file + filename.

        Args:
            subsection (str): Subsection of the config to search in (e.g., "computed correlations").
            absorber (str): First absorber name (e.g., "lya").
            region (str): First region where the correlation is computed.
            absorber2 (Optional[str]): Second absorber name (for cross-correlations) if applicable.
            region2 (Optional[str]): Second region where the correlation is computed, if applicable.
            filename (Optional[str]): Name of the file to retrieve under general path.
            tracer (Optional[str]): Tracer name (e.g., "qso").

        Returns:
            Path | None: Path to the correlation file if found; otherwise, None.

        Raises:
            FileNotFoundError: If a configured path exists but the file is missing.
        """
        if absorber2 is None and (tracer is not None):
            name = f"{absorber}{region}"
            tracer += "_"
        else:
            name = f"{absorber}{region}_{absorber2}{region2}"
            tracer = ""

        if (
            self.config["correlations"].get(subsection, dict()).get(tracer + name, None)
            is not None
        ):
            file = Path(
                self.config["correlations"].get(subsection, dict()).get(tracer + name)
            )

            if not file.is_file():
                raise FileNotFoundError(
                    f"{tracer + name}: Invalid correlation file provided", file
                )

        elif (
            self.config["correlations"].get(subsection, dict()).get("general", None)
            is not None
            and filename is not None
        ):
            parent = Path(
                self.config["correlations"].get(subsection, dict()).get("general")
            )
            file = parent / (tracer + name) / filename

            if not file.is_file():
                logger.info(
                    f"{tracer + name}: no file provided to use, correlation will be computed"
                )

                return None

        else:
            logger.info(
                f"{tracer + name}: no file provided to use, correlation will be computed"
            )

            return None

        logger.info(f"{tracer + name}: Using correlation from file:\n\t{str(file)}")
        return file

    def copied_covariance_file(self, smoothed: bool = False) -> Path | None:
        """
        Return the path to a precomputed covariance file, if specified in the bookkeeper config.
    
        Args:
            smoothed (bool, optional): If True, returns the smoothed covariance file.
                                       Defaults to False.
    
        Returns:
            Path | None: Path to the covariance file, or None if it should be computed.
        """
        name = "full-covariance"

        if smoothed:
            name += "-smoothed"

        if (
            self.config["fits"].get("computed covariances", dict()).get(name, None)
            is not None
        ):
            file = self.config["fits"].get("computed covariances", dict()).get(name)

            if not file.is_file():
                raise FileNotFoundError(
                    f"{name}: Invalid covariance file provided", file
                )

        elif (
            self.config["fits"].get("computed covariances", dict()).get("general", None)
            is not None
        ):
            file = (
                Path(
                    self.config["fits"]
                    .get("computed covariances", dict())
                    .get("general")
                )
                / f"{name}.fits"
            )

            if not file.is_file():
                logger.info(
                    f"{name}: no file provided to use, covariance will be computed"
                )

                return None

        else:
            logger.info(f"{name}: no file provided to use, covariance will be computed")

            return None

        logger.info(f"{name}: Using covariance from file:\n\t{str(file)}")
        return file

    def cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Return the path to the forest-forest correlation function file.
    
        Args:
            absorber (str): First absorber.
            region (str): Region where the correlation is computed.
            absorber2 (Optional[str]): Second absorber (for cross-correlations), if applicable.
            region2 (Optional[str]): Second region (for cross-correlations), if applicable.
    
        Returns:
            Path: Path to the correlation file (cf.fits.gz).
        """
        region2 = region if region2 is None else region2
        absorber2 = absorber if absorber2 is None else absorber2
        return (
            self.correlations_path
            / "results"
            / f"{absorber.replace('(', '').replace(')', '')}{region}_"
            f"{absorber2.replace('(', '').replace(')', '')}{region2}" / f"cf.fits.gz"
        )

    def dmat_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Return the path to the distortion matrix file for forest-forest correlations.
    
        Args:
            absorber (str): First absorber.
            region (str): Region where the correlation is computed.
            absorber2 (Optional[str]): Second absorber (for cross-correlations), if applicable.
            region2 (Optional[str]): Second region (for cross-correlations), if applicable.
    
        Returns:
            Path: Path to the distortion matrix file (dmat.fits.gz).
        """
        return (
            self.cf_fname(absorber, region, absorber2, region2).parent / f"dmat.fits.gz"
        )

    def metal_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Return the path to the metal distortion matrix file, for forest-forest correlations,
        if it exists.
    
        Args:
            absorber (str): First absorber.
            region (str): Region where the correlation is computed.
            absorber2 (Optional[str]): Second absorber (for cross-correlations), if applicable.
            region2 (Optional[str]): Second region (for cross-correlations), if applicable.
    
        Returns:
            Path: Path to the metal distortion file (metal.fits or metal.fits.gz).
        """
        parent = self.cf_fname(absorber, region, absorber2, region2).parent
        if (parent / "metal.fits.gz").is_file():
            return parent / "metal.fits.gz"
        else:
            return parent / f"metal.fits"

    def exp_cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Return the path to the exported forest-forest correlation function export file.
    
        Args:
            absorber (str): First absorber.
            region (str): Region where the correlation is computed.
            absorber2 (Optional[str]): Second absorber (if cross-correlation), if applicable.
            region2 (Optional[str]): Second region (if cross-correlation), if applicable.
    
        Returns:
            Path: Path to the exported correlation file (cf_exp.fits.gz).
        """        
        cor_file = self.cf_fname(absorber, region, absorber2, region2)
        return cor_file.parent / f"cf_exp.fits.gz"

    def xcf_fname(self, absorber: str, region: str, tracer: str = "qso") -> Path:
        """
        Return the path to the forest-quasar correlation function export file.
    
        Args:
            absorber (str): Absorber to use (e.g., 'lya').
            region (str): Region of the forest used.
            tracer (str): Tracer to use (default: 'qso').
    
        Returns:
            Path: Path to the correlation file (xcf.fits.gz).
        """
        return (
            self.correlations_path
            / "results"
            / f"{tracer}_{absorber.replace('(', '').replace(')', '')}{region}"
            / f"xcf.fits.gz"
        )

    def xdmat_fname(self, absorber: str, region: str, tracer: str = "qso") -> Path:
        """
        Return the path to the distortion matrix file for forest-quasar correlations.
    
        Args:
            absorber (str): Absorber to use (e.g., 'lya').
            region (str): Region of the forest used.
            tracer (str): Tracer to use (default: 'qso').
    
        Returns:
            Path: Path to the distortion matrix file (xdmat.fits.gz).
        """    
        return self.xcf_fname(absorber, region, tracer).parent / f"xdmat.fits.gz"

    def xmetal_fname(self, absorber: str, region: str, tracer: str = "qso") -> Path:
        """
        Return the path to the metal distortion matrix file for forest-quasar correlations.
    
        Args:
            absorber (str): Absorber to use (e.g., 'lya').
            region (str): Region of the forest used.
            tracer (str): Tracer to use (default: 'qso').
    
        Returns:
            Path: Path to the metal distortion matrix file (xmetal.fits[.gz]).
        """
        parent = self.xcf_fname(absorber, region, tracer).parent
        if (parent / "xmetal.fits.gz").is_file():
            return parent / "xmetal.fits.gz"
        else:
            return parent / "xmetal.fits"

    def exp_xcf_fname(self, absorber: str, region: str, tracer: str = "qso") -> Path:
        """
        Return the path to the exported forest-quasar correlation function export file.
    
        Args:
            absorber (str): Absorber to use (e.g., 'lya').
            region (str): Region of the forest used.
            tracer (str): Tracer to use (default: 'qso').
    
        Returns:
            Path: Path to the exported correlation file (xcf_exp.fits.gz).
        """    
        cor_file = self.xcf_fname(absorber, region, tracer)
        return cor_file.parent / f"xcf_exp.fits.gz"

    def fit_auto_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Return the path to the fit config file for forest-forest auto- correlations.
    
        Args:
            absorber (str): First absorber.
            region (str): First region.
            absorber2 (Optional[str]): Second absorber (defaults to first if None), if applicable.
            region2 (Optional[str]): Second region (defaults to first if None), if applicable.
    
        Returns:
            Path: Path to the fit config file (.ini).
        """
        region2 = region if region2 is None else region2
        absorber2 = absorber if absorber2 is None else absorber2

        name = (
            self.fits_path / "configs" / f"{absorber}{region}x{absorber2}{region2}.ini"
        )
        return name.parent / name.name.replace("(", "").replace(")", "")

    def fit_cross_fname(self, absorber: str, region: str, tracer: str) -> Path:
        """
        Return the path to the fit config file for forest-quasar cross-correlations.
    
        Args:
            absorber (str): Absorber used (e.g., 'lya').
            region (str): Region used.
            tracer (str): Tracer used (e.g., 'qso').
    
        Returns:
            Path: Path to the fit config file (.ini).
        """
        name = self.fits_path / "configs" / f"{tracer}x{absorber}{region}.ini"
        return name.parent / name.name.replace("(", "").replace(")", "")

    def fit_main_fname(self) -> Path:
        """
        Return the path to the main fit configuration file.
    
        Returns:
            Path: Path to main fit config file (main.ini).
        """
        return self.fits_path / "configs" / "main.ini"

    def fit_out_fname(self) -> Path:
         """
        Return the path to the output file from the fit.
    
        Returns:
            Path: Path to fit output file (fit_output.fits).
        """
        return self.fits_path / "results" / "fit_output.fits"

    def covariance_file_unsmoothed(self) -> Path:
        """
        Return the path to the unsmoothed covariance file.
    
        Returns:
            Path: The path to the unsmoothed covariance file (full-covariance.fits).
        """
        return self.fits_path / "results" / "full-covariance.fits"

    def covariance_file_smoothed(self) -> Path:
        """
        Return the path to the smoothed covariance file.
    
        Returns:
            Path: The path to the smoothed covariance file (full-covariance-smoothed.fits).
        """
        return self.fits_path / "results" / "full-covariance-smoothed.fits"

    def covariance_file(self) -> Path:
        """
        Return the path to the covariance file, selecting smoothed or unsmoothed based on config.
    
        Returns:
            Path: Path to the selected covariance file.
        """
        if self.config["fits"].get("smooth covariance", False):
            return self.covariance_file_smoothed()
        else:
            return self.covariance_file_unsmoothed()

    def sampler_out_path(self) -> Path:
        """
        Return the path to the sampler output directory.
    
        Returns:
            Path: Path to sampler output folder.
        """
        return self.fits_path / "results" / "sampler"

    def fit_computed_params_out(self) -> Path:
        """
        Return the path to the file containing computed fit parameters (e.g., zeff).
    
        Returns:
            Path: Path to file (computed_parameters.yaml).
        """
        return self.fits_path / "results" / "computed_parameters.yaml"
