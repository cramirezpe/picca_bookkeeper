from __future__ import annotations

import configparser
import copy
import filecmp
import logging
import shutil
import sys
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
from picca_bookkeeper.utils import compute_zeff

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# This converts Nones in dict into empty fields in yaml.
SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)

config_file_sorting = ["general", "delta extraction", "correlations", "fits"]


def get_quasar_catalog(
    release: str, survey: str, catalog: str, bal: bool = False
) -> Path:  # pragma: no cover
    """Function to obtain a quasar catalog given different options

    Attributes:
        release (str): everest, fuji, guadalupe, fugu, ...
        survey (str): sv1, sv3, sv13, main, all
        catalog (str): redrock_v0, afterburn_v0, afterburn_v1, ...
        bal (bool): whether to search for catalogs with BALs included.
    """
    catalogues_file = files(resources).joinpath("catalogues/quasar.yaml")
    catalogues = yaml.safe_load(catalogues_file.read_text())

    if bal:
        catalog += "-bal"

    if catalogues[release][survey].get(catalog, None) is not None:
        return Path(catalogues[release][survey][catalog])
    elif (
        not bal and catalogues[release][survey].get(catalog + "-bal", None) is not None
    ):
        return Path(catalogues[release][survey][catalog + "-bal"])
    else:
        raise FileNotFoundError(
            "Could not find a compatible quasar catalogue inside the bookkeeper."
            f"\tRelease: {release}\n\tSurvey: {survey}\n\tCatalog: {catalog}"
        )


def get_dla_catalog(release: str, survey: str, version: int = 1) -> Path:
    """Function to obtain a DLA catalog.

    Arguments:
        release (str): everest, fuji, guadalupe, fugu,...
        survey (str): sv1, sv3, sv13, main, all
        version (float): version of the catalog
    """
    catalogues_file = files(resources).joinpath("catalogues/dla.yaml")
    catalogues = yaml.safe_load(catalogues_file.read_text())

    if catalogues[release][survey].get(f"v{version}", None) is not None:
        return Path(catalogues[release][survey][f"v{version}"])
    else:
        raise FileNotFoundError(
            "Could not find a compatible DLA catalogue inside the bookkeeper."
            f"\tRelease: {release}\n\tSurvey: {survey}\n\tVersion: {version}"
        )


class Bookkeeper:
    """Class to generate Tasker objects which can be used to run different picca jobs.

    Attributes:
        config (configparser.ConfigParser): Configuration file for the bookkeeper.
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

        self.paths = PathBuilder(self.config)

        self.read_mode = read_mode

        # Potentially could add fits things here
        # Defining dicts containing all information.
        self.fits = None
        self.correlations = None
        self.delta_extraction = None

        if self.config.get("fits") is not None:
            self.fits = self.config.get("fits")
            config_type = "fits"
        if self.config.get("correlations") is not None:
            self.correlations = self.config.get("correlations")
            config_type = "correlations"
        if self.config.get("delta extraction") is not None:
            self.delta_extraction = self.config.get("delta extraction")
            config_type = "deltas"

        if config_type == "fits":
            # In this case, correlations is not defined in the config file
            # and therefore, we should search for it.
            with open(self.paths.correlation_config_file, "r") as f:
                correlation_config = yaml.safe_load(f)
            self.correlations = correlation_config["correlations"]
            self.config["correlations"] = self.correlations

        if config_type in ("fits", "correlations"):
            # In this case, delta extraction is not defined in the config file
            # and therefore, we should search for it.
            with open(self.paths.delta_config_file, "r") as f:
                delta_config = yaml.safe_load(f)
            self.delta_extraction = delta_config["delta extraction"]
            self.config["delta extraction"] = self.delta_extraction

        self.paths = PathBuilder(self.config)

        if self.read_mode:
            # Next steps imply writting on bookkeeper destination
            # for read_mode we can finish here.
            return

        self.paths.check_delta_directories()

        if self.correlations is not None:
            self.paths.check_correlation_directories()

        if self.fits is not None:
            self.paths.check_fit_directories()

        # Potentially could add fits things here.
        # Copy bookkeeper configuration into destination
        # If bookkeeper included delta
        if config_type == "deltas":
            config_delta = copy.deepcopy(self.config)
            config_delta.pop("correlations", None)
            config_delta.pop("fits", None)

            if not self.paths.delta_config_file.is_file():
                self.write_bookkeeper(config_delta, self.paths.delta_config_file)
            elif filecmp.cmp(self.paths.delta_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                # If we want to directly overwrite the config file in destination
                self.write_bookkeeper(config_delta, self.paths.delta_config_file)
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path, self.paths.delta_config_file, "delta extraction"
                )
                if comparison == dict():
                    # They are the same
                    self.write_bookkeeper(config_delta, self.paths.delta_config_file)
                else:
                    raise ValueError(
                        "delta extraction section of config file should match delta "
                        "extraction section from file already in the bookkeeper. "
                        "Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.delta_config_file}"
                    )

        if self.correlations is not None and config_type != "fits":
            config_corr = copy.deepcopy(self.config)

            config_corr["correlations"]["delta extraction"] = self.paths.continuum_tag

            config_corr.pop("delta extraction")
            config_corr.pop("fits", None)

            if not self.paths.correlation_config_file.is_file():
                self.write_bookkeeper(config_corr, self.paths.correlation_config_file)
            elif filecmp.cmp(self.paths.correlation_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                self.write_bookkeeper(config_corr, self.paths.correlation_config_file)
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path,
                    self.paths.correlation_config_file,
                    "correlations",
                    ["delta extraction"],
                )
                if comparison == dict():
                    # They are the same
                    self.write_bookkeeper(
                        config_corr, self.paths.correlation_config_file
                    )
                else:
                    raise ValueError(
                        "correlations section of config file should match correlation section "
                        "from file already in the bookkeeper. Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.correlation_config_file}"
                    )
        if self.fits is not None:
            config_fit = copy.deepcopy(self.config)

            config_fit["fits"]["delta extraction"] = self.paths.continuum_tag

            if self.config["fits"].get("correlation run name", None) is None:
                config_fit["fits"]["correlation run name"] = self.config[
                    "correlations"
                ]["run name"]

            config_fit.pop("delta extraction")
            config_fit.pop("correlations", None)

            if not self.paths.fit_config_file.is_file():
                self.write_bookkeeper(config_fit, self.paths.fit_config_file)
            elif filecmp.cmp(self.paths.fit_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                self.write_bookkeeper(config_fit, self.paths.fit_config_file)
            else:
                comparison = PathBuilder.compare_config_files(
                    config_path,
                    self.paths.fit_config_file,
                    "fits",
                    ["delta extraction", "correlation run name"],
                )
                if comparison == dict():
                    self.write_bookkeeper(config_fit, self.paths.fit_config_file)
                else:
                    raise ValueError(
                        "fits section of config file should match fits section "
                        "from file already in the bookkeeper. Unmatching items:\n\n"
                        f"{DictUtils.print_dict(comparison)}\n\n"
                        f"Remove config file to overwrite {self.paths.fit_config_file}"
                    )

        # Read defaults and check if they have changed.
        defaults_file = files(resources).joinpath(
            "default_configs/"
            + str(self.config["delta extraction"]["prefix"])
            + ".yaml"
        )
        if not defaults_file.is_file():
            raise ValueError("Invalid prefix, no defaults file found.", defaults_file)

        self.defaults = yaml.safe_load(defaults_file.read_text())

        self.defaults_diff = dict()

        if self.paths.defaults_file.is_file():
            self.defaults_diff = PathBuilder.compare_config_files(
                self.paths.defaults_file,
                defaults_file,
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

    @staticmethod
    def write_ini(config: Dict, file: Path | str) -> None:
        """Safely save a dictionary into an .ini file

        Args
            config: Dict to store as ini file.
            file: path where to store the ini.
        """
        config = DictUtils.convert_to_string(config)

        parser = configparser.ConfigParser()
        parser.optionxform = str  # type: ignore
        parser.read_dict(config)

        with open(file, "w") as write_file:
            parser.write(write_file)

    @staticmethod
    def write_bookkeeper(config: Dict, file: Path | str) -> None:
        """Method to write bookkeeper yaml file to file

        Args:
            config: Dict to store as yaml file.
            file: path where to store the bookkeeper.
        """
        correct_order = {
            "general": ["conda environment", "system", "slurm args"],
            "data": ["bookkeeper dir", "healpix data", "release", "survey", "catalog"],
            "delta extraction": [
                "prefix",
                "calib",
                "calib region",
                "dla",
                "bal",
                "suffix",
                "calibration data",
                "deltas",
                "link deltas",
                "mask file",
                "dla catalog",
                "bal catalog",
                "extra args",
                "slurm args",
            ],
            "correlations": [
                "delta extraction",
                "run name",
                "unblind",
                "catalog tracer",
                "fast metals",
                "link correlations",
                "link metals",
                "cf files",
                "cf exp files",
                "xcf files",
                "xcf exp files",
                "distortion matrices",
                "xdistortion matrices",
                "metal matrices",
                "xmetal matrices",
                "extra args",
                "slurm args",
            ],
            "fits": [
                "delta extraction",
                "correlation run name",
                "run name",
                "auto correlations",
                "cross correlations",
                "compute zeff",
                "sampler environment",
                "bao",
                "hcd",
                "distortion",
                "metals",
                "sky",
                "qso rad",
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

    @staticmethod
    def validate_region(region: str) -> str:
        """Method to check if a region string is valid.

        Will raise value error if the region is not in forest_regions.

        Args:
            region: Region (should be in forest_regions to pass the validation).
        """
        if region not in forest_regions:
            raise ValueError("Invalid region", region)

        return region

    @staticmethod
    def validate_absorber(absorber: str) -> str:
        """Method to check if a absorber is valid.

        Will raise value error if the absorber not in picca.absorbers.ABSORBER_IGM

        lya and lyb are exceptions in lowercase.

        Args:
            absorber: Absorber to be used
        """
        if absorber not in absorber_igm:
            raise ValueError("Invalid absorber", absorber)

        return absorber

    def generate_slurm_header_extra_args(
        self,
        config: Dict,
        section: str,
        slurm_args: Dict,
        command: str,
        region: Optional[str] = None,
        absorber: Optional[str] = None,
        region2: Optional[str] = None,
        absorber2: Optional[str] = None,
    ) -> Dict:
        """Add extra slurm header args to the run.

        Args:
            config: bookkeeper config to look into.
            section: Section name to look into.
            slurm_args: Slurm args passed through the get_tasker method. They
                should be prioritized.
            command: Picca command to be run.
            region: Specify region where the command will be run.
            absorber: First absorber to use for correlations.
            region2: For scripts where two regions are needed.
            absorber2: Second absorber to use for correlations.
        """
        if "slurm args" in config["general"]:
            args = copy.deepcopy(config["general"]["slurm args"])
        else:
            args = dict()

        config = copy.deepcopy(config[section])

        sections = ["general", command.split(".py")[0]]

        absorber = "" if absorber is None else absorber
        absorber2 = "" if absorber2 is None else absorber2
        region = "" if region is None else region
        region2 = "" if region2 is None else region2

        if region != "":
            sections.append(command.split(".py")[0] + f"_{absorber}{region}")

        if region2 != "":
            sections[-1] += f"_{absorber2}{region2}"

        if "slurm args" in config.keys() and isinstance(config["slurm args"], dict):
            for section in sections:
                if section in config["slurm args"] and isinstance(
                    config["slurm args"][section], dict
                ):
                    args = DictUtils.merge_dicts(args, config["slurm args"][section])

            # remove args marked as remove_
            for section in sections:
                if "remove_" + section in config["slurm args"] and isinstance(
                    config["slurm args"]["remove_" + section], dict
                ):
                    args = DictUtils.remove_matching(
                        args, config["slurm args"]["remove_" + section]
                    )

        for section in sections:
            if section in slurm_args and isinstance(slurm_args[section], dict):
                args = DictUtils.merge_dicts(args, slurm_args[section])

            if "remove_" + section in slurm_args and isinstance(
                slurm_args["remove_" + section], dict
            ):
                args = DictUtils.remove_matching(args, slurm_args["remove_" + section])

        return args

    def generate_extra_args(
        self,
        config: Dict,
        section: str,
        extra_args: Dict,
        command: str,
        region: Optional[str] = None,
        absorber: Optional[str] = None,
        region2: Optional[str] = None,
        absorber2: Optional[str] = None,
    ) -> Dict:
        """Add extra extra args to the run.

        Args:
            config: Section of the bookkeeper config to look into.
            section: Section name to look into.
            extra_args: extra args passed through the get_tasker method. They
                should be prioritized.
            command: picca command to be run.
            region: specify region where the command will be run.
            absorber: First absorber to use for correlations.
            region2: For scripts where two regions are needed.
            absorber2: Second absorber to use for correlations.
        """
        config = copy.deepcopy(config[section])

        sections = [command.split(".py")[0]]

        absorber = "" if absorber is None else absorber
        absorber2 = "" if absorber2 is None else absorber2
        region = "" if region is None else region
        region2 = "" if region2 is None else region2

        if region != "":
            sections.append(command.split(".py")[0] + f"_{absorber}{region}")

        if region2 != "":
            sections[-1] += f"_{absorber2}{region2}"

        args: Dict = dict()
        if "extra args" in config.keys() and isinstance(config["extra args"], dict):
            for section in sections:
                if section in config["extra args"] and isinstance(
                    config["extra args"][section], dict
                ):
                    args = DictUtils.merge_dicts(args, config["extra args"][section])

            # remove args marked as remove_
            for section in sections:
                if "remove_" + section in config["extra args"] and isinstance(
                    config["extra args"]["remove_" + section], dict
                ):
                    args = DictUtils.remove_matching(
                        args, config["extra args"]["remove_" + section]
                    )

        for section in sections:
            if section in extra_args and isinstance(extra_args[section], dict):
                args = DictUtils.merge_dicts(args, extra_args[section])

            if "remove_" + section in extra_args and isinstance(
                extra_args["remove_" + section], dict
            ):
                args = DictUtils.remove_matching(args, extra_args["remove_" + section])

        return args

    def generate_system_arg(self, system: Optional[str]) -> str:
        if system is None:
            return copy.copy(
                self.config.get("general", dict()).get("system", "slurm_perlmutter")
            )
        else:
            return system

    def add_calibration_options(
        self, extra_args: Dict, calib_step: Optional[int] = None
    ) -> Tuple[List[Path], Dict]:
        """Method to add calibration options to extra args

        Args:
            extra_args: Configuration to be used in the run.
            calib_step: Current calibration step, None if main run.

        Retursn:
            updated extra_args
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
                                    self.paths.delta_attributes_file(None, calib_step=1)
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
                                self.paths.delta_attributes_file(None, calib_step=1)
                            ),
                        },
                        f"correction arguments {num_corrections - 1}": {
                            "filename": str(
                                self.paths.delta_attributes_file(None, calib_step=2)
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
                                self.paths.delta_attributes_file(None, calib_step=1)
                            ),
                        },
                    },
                )
                input_files.append(self.paths.delta_attributes_file(None, calib_step=1))
        return input_files, extra_args

    def add_mask_options(
        self, extra_args: Dict, calib_step: Optional[int] = None
    ) -> Dict:
        """Method to add mask options to extra args

        Args:
            extra_args: Configuration to be used in the run.
            calib_step: Current calibration step, None if main run.

        Retursn:
            updated extra_args
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
            if self.config["delta extraction"]["dla"] != 0:
                if "DlaMask" in extra_args["masks"].values():
                    raise ValueError("DlaMask set by user with dla option != 0")

                prev_mask_number = int(extra_args["masks"]["num masks"])
                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "masks": {
                            "num masks": prev_mask_number + 1,
                            f"type {prev_mask_number}": "DlaMask",
                        },
                        f"mask arguments {prev_mask_number}": {
                            f"filename": self.paths.catalog_dla,
                            "los_id name": "TARGETID",
                        },
                    },
                )

            if self.config["delta extraction"]["bal"] != 0:
                if self.config["delta extraction"]["bal"] == 2:
                    if "BalMask" in extra_args["masks"].values():
                        raise ValueError("BalMask set by user with bal option !=0")

                prev_mask_number = int(extra_args["masks"]["num masks"])
                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "masks": {
                            "num masks": prev_mask_number + 1,
                            f"type {prev_mask_number}": "BalMask",
                        },
                        f"mask arguments {prev_mask_number}": {
                            "filename": self.paths.catalog_bal,
                            "los_id name": "TARGETID",
                        },
                    },
                )

        return extra_args

    def validate_mock_runs(self, deltas_config_dict: Dict) -> None:
        """Method to validate config file for mock runs

        Args:
            delta_config_dict: Dict containing config to be used
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
        if self.config["delta extraction"]["prefix"] == "raw":
            raise ValueError(
                f"raw continuum fitting provided in config file, use "
                "get_raw_deltas_tasker instead"
            )
        elif self.config["delta extraction"]["prefix"] == True:
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
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run raw deltas with picca.

        Args:
            region: Region where to compute deltas. Options: forest_regions.
                Default: 'lya'
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run was already
                sent before.

        Returns:
            Tasker: Tasker object to run delta extraction.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            extra_args=extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="delta extraction",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "object-cat": str(self.paths.catalog),
            "in-dir": str(self.paths.transmission_data),
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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.run_path / f"logs/jobids.log",
            wait_for=wait_for,
            out_file=delta_stats_file,
        )

    def get_delta_extraction_tasker(
        self,
        region: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        calib_step: Optional[int] = None,
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run delta extraction with picca.

        Args:
            region: Region where to compute deltas. Options: forest_regions.
                Default: 'lya'
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.
            extra_args: Set extra options for picca delta extraction.
                The format should be a dict of dicts: wanting to change
                "num masks" in "masks" section one should pass
                {'num masks': {'masks': value}}.
            calib_step: Calibration step. Default: None, no calibration
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run delta extraction.
        """
        copied_attributes: Path | None

        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            extra_args=extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="delta extraction",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )

        config_file = self.paths.run_path / f"configs/{job_name}.ini"

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
                    "type": "DesiHealpix",
                    "catalogue": str(self.paths.catalog),
                    "input directory": str(self.paths.healpix_data),
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
            "output": str(self.paths.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.run_path / f"logs/{job_name}-%j.err"),
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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.run_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=input_files,
            out_file=self.paths.delta_attributes_file(region, calib_step),
        )

    def get_calibration_extraction_tasker(
        self,
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> ChainedTasker:
        """Method to get a Tasker object to run calibration with picca delta
        extraction method.

        Args:
            system (str, optional): Shell to use for job. 'slurm_cori' to use
                slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts
                on perlmutter, 'bash' to run it in login nodes or computer
                shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job
                to finish before running the current one. Could be a  Tasker object
                or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a dictionary
                with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run delta extraction for calibration.
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
                    slurm_header_extra_args=slurm_header_extra_args,
                    extra_args=extra_args,
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
                    slurm_header_extra_args=slurm_header_extra_args,
                    extra_args=extra_args,
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
                    slurm_header_extra_args=slurm_header_extra_args,
                    extra_args=extra_args,
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
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-forest correlations with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2: Region to use for cross-correlations.
                Default: None, auto-correlation.
            absorber: First absorber to use for correlations.
            absorber2: Second absorber to use for correlations.
                Default: Same as absorber.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            "cf files",
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
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
            "in-dir": str(self.paths.deltas_path(region)),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        if absorber2 != absorber:
            args["lambda-abs2"] = absorber_igm[absorber2]

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_file=output_filename,
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
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-forest distortion matrix
        measurements with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2: Region to use for cross-correlations.
                Default: None, auto-correlation.
            absorber: First absorber to use for correlations.
            absorber2: Second absorber to use for correlations.
                Default: Same as absorber.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            "distortion matrices",
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
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
            "in-dir": str(self.paths.deltas_path(region)),
            "out": str(output_filename),
            "lambda-abs": absorber_igm[absorber],
        }

        if absorber2 != absorber:
            args["lambda-abs2"] = absorber_igm[absorber2]

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_file=output_filename,
        )

    def get_cf_exp_tasker(
        self,
        region: str = "lya",
        region2: Optional[str] = None,
        absorber: str = "lya",
        absorber2: Optional[str] = None,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-forest correlation export with
         picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2: Region to use for cross-correlations.
                Default: None, auto-correlation.
            absorber: First absorber to use for correlations.
            absorber2: Second absorber to use for correlations.
                Default: Same as absorber.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            "cf exp files", absorber, region, absorber2, region2, output_filename.name
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
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
            "data": str(self.paths.cf_fname(absorber, region, absorber2, region2)),
            "out": str(output_filename),
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if (
            self.paths.config["data"]["survey"] == "main"
            or self.paths.config["data"]["survey"] == "all"
        ):
            args["blind-corr-type"] = "lyaxlya"

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        in_files = [
            self.paths.cf_fname(absorber, region, absorber2, region2),
        ]

        if self.config["correlations"].get("unblind", False):
            logger.warn("CORRELATIONS WILL BE UNBLINDED, BE CAREFUL.")
            precommand = f"picca_bookkeeper_unblind_correlations"
            
            cf_file = str(self.paths.cf_fname(absorber, region, absorber2, region2))
            precommand += f" --cf {cf_file}"
            if self.config["fits"].get("distortion", True):
                dmat_file = str(self.paths.dmat_fname(absorber, region, absorber2, region2))
                precommand += f" --dmat {dmat_file}"
                in_files.append(dmat_file)
            if self.config["fits"].get("metals", True):
                metal_file = str(self.paths.metal_fname(absorber, region, absorber2, region2))
                precommand += f" --metal-dmat {metal_file}"
                in_files.append(metal_file)
        else:
            precommand = ""

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            precommand=precommand,
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=in_files,
            out_file=output_filename,
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
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-forest metal distortion matrix
        measurements with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2: Region to use for cross-correlations.
                Default: None, auto-correlation.
            absorber: First absorber to use for correlations.
            absorber2: Second absorber to use for correlations.
                Default: Same as absorber.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            "metal matrices", absorber, region, absorber2, region2, output_filename.name
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
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
            args["in-attributes"] = str(self.paths.delta_attributes_file(region))
            args["delta-dir"] = str(self.paths.deltas_path(region))

            if region2 != region:
                args["in-attributes2"] = str(self.paths.delta_attributes_file(region2))
        else:
            args["in-dir"] = str(self.paths.deltas_path(region))

            if region2 != region:
                args["in-dir2"] = str(self.paths.deltas_path(region2))

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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region_)
                for region_ in (region, region2)
            ],
            out_file=output_filename,
        )

    def get_xcf_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-quasar correlations with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            absorber: First absorber to use for correlations.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        job_name = f"xcf_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xcf_fname(absorber, region)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xcf_file = self.paths.copied_correlation_file(
            "xcf files",
            absorber,
            region,
            None,
            None,
            output_filename.name,
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_file=output_filename,
        )

    def get_xdmat_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-quasar distortion matrix
        measurements with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb').
                Default: 'lya'.
            absorber: First absorber to use for correlations.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm
                scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell. Default: None, read
                from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to
                finish before running the current one. Could be a  Tasker object
                or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default
                options if needed (time, qos, etc...). Use a dictionary with the
                format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to picca_deltas.py script.
                Use a dictionary with the format {'argument_name', 'argument_value'}.
                Use {'argument_name': ''} if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-quasar distortion matrix.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        job_name = f"xdmat_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xdmat_fname(absorber, region)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xdmat_file = self.paths.copied_correlation_file(
            "xdistortion matrices",
            absorber,
            region,
            None,
            None,
            output_filename.name,
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_file=output_filename,
        )

    def get_xcf_exp_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        """Method to get a Tasker object to run forest-quasar correlation export with
        picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            absorber: First absorber to use for correlations.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)
        job_name = f"xcf_exp_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.exp_xcf_fname(absorber, region)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_xcf_exp_file = self.paths.copied_correlation_file(
            "xcf exp files",
            absorber,
            region,
            None,
            None,
            output_filename.name,
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
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
            "data": str(self.paths.xcf_fname(absorber, region)),
            "out": str(output_filename),
            "blind-corr-type": "qsoxlya",
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if (
            self.paths.config["data"]["survey"] == "main"
            or self.paths.config["data"]["survey"] == "all"
        ):
            args["blind-corr-type"] = "qsoxlya"

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        output_filename.parent.mkdir(exist_ok=True, parents=True)

        in_files = [
            self.paths.xcf_fname(absorber, region),
        ]

        if self.config["correlations"].get("unblind", False):
            logger.warn("CORRELATIONS WILL BE UNBLINDED, BE CAREFUL.")
            precommand = f"picca_bookkeeper_unblind_correlations"
            
            xcf_file = str(self.paths.xcf_fname(absorber, region))
            precommand += f" --cf {xcf_file}"
            if self.config["fits"].get("distortion", True):
                xdmat_file = str(self.paths.xdmat_fname(absorber, region))
                precommand += f" --dmat {xdmat_file}"
                in_files.append(xdmat_file)
            if self.config["fits"].get("metals", True):
                xmetal_file = str(self.paths.metal_fname(absorber, region))
                precommand += f" --metal-dmat {xmetal_file}"
                in_files.append(xmetal_file)
        else:
            precommand = ""

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            precommand=precommand,
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=in_files,
            out_file=output_filename,
        )

    def get_xmetal_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: Optional[str] = None,
        debug: bool = False,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker:
        """Method to get a Tasker object to run forest-quasar metal distortion matrix
        measurements with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            absorber: First absorber to use for correlations.
            system: Shell to use for job. 'slurm_perlmutter' to use slurm
                scripts on perlmutter, 'bash' to  run it in login nodes or
                computer shell. Default: None, read from config file.
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            overwrite: Overwrite files in destination.
            skip_sent: Skip this and return a DummyTasker if the run
                was already sent before.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)
        job_name = f"xmetal_{absorber}{region}"
        job_name = job_name.replace("(", "").replace(")", "")

        # Check if output already there
        updated_system = self.generate_system_arg(system)
        output_filename = self.paths.xmetal_fname(absorber, region)
        if self.check_existing_output_file(
            output_filename,
            job_name,
            skip_sent,
            overwrite,
            updated_system,
        ):
            return DummyTasker()

        copy_metal_matrix = self.paths.copied_correlation_file(
            "xmetal matrices",
            absorber,
            region,
            None,
            None,
            output_filename.name,
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
                f"values\). Defaults diff:\n\n"
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
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )

        slurm_header_args = {
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = DictUtils.merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {}
        if fast_metal:
            args["in-attributes"] = str(self.paths.delta_attributes_file(region))
            args["delta-dir"] = str(self.paths.deltas_path(region))
        else:
            args["in-dir"] = str(self.paths.deltas_path(region))

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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
            in_files=[
                self.paths.delta_attributes_file(region),
            ],
            out_file=output_filename,
        )

    def get_compute_zeff_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """Method to get a Tasker object to run correct_config_zeff

        Args:
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.
            overwrite: Overwrite files in destination.
            skip_sent: Skip the run if fit output already present.

        Returns:
            Tasker: Tasker object to run correct_config_zeff.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            extra_args=dict(),
            section="fits",
            command=command,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            section="fits",
            slurm_args=slurm_header_extra_args,
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
            "": self.paths.fit_config_file,
        }

        args = DictUtils.merge_dicts(args, updated_extra_args)

        return get_Tasker(updated_system)(
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_file=self.paths.fit_computed_params_out(),
        )

    def get_fit_tasker(
        self,
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        # vega_extra_args: Dict = dict(),
        slurm_header_extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """Method to get a Tasker object to run vega with correlation data.

        Args:
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.
            overwrite: Overwrite files in destination.
            skip_sent: Skip the run if fit output already present.

        Returns:
            Tasker: Tasker object to run vega.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            slurm_args=slurm_header_extra_args,
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
            slurm_header_args=slurm_header_args,
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_file=self.paths.fit_out_fname(),
        )

    def get_sampler_tasker(
        self,
        auto_correlations: List[str] = [],
        cross_correlations: List[str] = [],
        system: Optional[str] = None,
        wait_for: Optional[Tasker | ChainedTasker | int | List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        overwrite: bool = False,
        skip_sent: bool = False,
    ) -> Tasker | DummyTasker:
        """Method to get a Tasker object to run vega sampler with correlation data.

        Args:
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.
            overwrite: Overwrite files in destination.
            skip_sent: Skip the run if sampler output already present.

        Returns:
            Tasker: Tasker object to run vega sampler.
        """
        if self.read_mode:
            raise ValueError("Initialize bookkeeper without read_mode to run jobs.")
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
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
            slurm_args=slurm_header_extra_args,
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
            slurm_header_args=slurm_header_args,
            environment=environment,
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            in_files=input_files,
            wait_for=wait_for,
            out_file=self.paths.sampler_out_path() / "jobidfile",
            force_OMP_threads=1,
        )

    def generate_fit_configuration(self) -> Dict:
        """
        Method to generate fit configuration args to be combined with input
        yaml file
        """
        config = DictUtils.merge_dicts(
            {
                "fits": {
                    "extra args": {},
                },
            },
            self.config,
        )

        for field in "bao", "hcd", "metals", "sky", "qso rad":
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
                    "data": {},
                    "cuts": {},
                    "model": {},
                    "metals": {},
                },
                "vega_cross": {
                    "data": {},
                    "cuts": {},
                    "model": {},
                    "metals": {},
                },
                "vega_main": {
                    "data sets": {},
                    "cosmo-fit type": {},
                    "fiducial": {},
                    "control": {},
                    "output": {},
                    "Polychord": {},
                    "sample": {},
                    "parameters": {},
                },
            },
            config["fits"]["extra args"],
        )

        if not config["fits"]["bao"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "remove_vega_main": {  # This removes fields
                        "sample": {
                            "ap": "",
                            "at": "",
                        }
                    },
                    "vega_main": {  # This adds/modifies fields
                        "parameters": {
                            "bao_amp": 0,
                        }
                    },
                },
            )

        if not config["fits"]["hcd"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "remove_vega_main": {
                        "sample": {
                            "bias_hcd": "",
                            "beta_hcd": "",
                        }
                    },
                    "remove_vega_auto": {
                        "model": {
                            "model-hcd": "",
                        }
                    },
                    "remove_vega_cross": {
                        "model": {
                            "model-hcd": "",
                        }
                    },
                },
            )

        if not config["fits"]["metals"]:
            remove_from_sampled = dict()
            for metal in (
                "SiII(1190)",
                "SiII(1193)",
                "SiII(1260)",
                "SiIII(1207)",
                "CIV(eff)",
            ):
                if f"bias_eta_{metal}" in config["fits"].get("extra args", dict()).get(
                    "vega_main", dict()
                ).get("sample", dict()):
                    remove_from_sampled[metal] = ""

            args = DictUtils.merge_dicts(
                args,
                {
                    "remove_vega_auto": {
                        "metals": "",
                    },
                    "remove_vega_cross": {
                        "metals": "",
                    },
                    "remove_vega_main": {
                        "sample": remove_from_sampled,
                    },
                },
            )

        if not config["fits"]["sky"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "remove_vega_main": {
                        "sample": {
                            "desi_inst_sys_amp": "",
                        }
                    },
                    "vega_main": {
                        "parameters": {
                            "desi_inst_sys_amp": "0",
                        }
                    },
                },
            )

        if not config["fits"]["qso rad"]:
            args = DictUtils.merge_dicts(
                args,
                {
                    "remove_vega_main": {
                        "sample": {
                            "qso_rad_strength": "",
                        }
                    },
                    "vega_main": {
                        "parameters": {
                            "qso_rad_strength": "0",
                        }
                    },
                },
            )

        if config["fits"]["rmin cf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_auto": {
                        "cuts": {
                            "r-min": str(config["fits"]["rmin cf"]),
                        }
                    }
                },
            )
        if config["fits"]["rmax cf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_auto": {
                        "cuts": {
                            "r-max": str(config["fits"]["rmax cf"]),
                        }
                    }
                },
            )
        if config["fits"]["rmin xcf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_cross": {
                        "cuts": {
                            "r-min": str(config["fits"]["rmin xcf"]),
                        }
                    }
                },
            )
        if config["fits"]["rmax xcf"] is not None:
            args = DictUtils.merge_dicts(
                args,
                {
                    "vega_cross": {
                        "cuts": {
                            "r-max": str(config["fits"]["rmax xcf"]),
                        }
                    }
                },
            )

        config["fits"]["extra args"] = args
        return config

    def write_fit_configuration(self) -> List[Path]:
        """
        Method to generate and write vega configuration files from bookkeeper config.

        Returns:
            List of input files, needed by the Tasker instance to correctly
            set up dependencies.
        """
        ini_files = []
        input_files = []

        config = self.generate_fit_configuration()

        if config["fits"].get("auto correlations", None) is not None:
            auto_correlations = config["fits"]["auto correlations"].split(" ")
        else:
            auto_correlations = []
        if config["fits"].get("cross correlations", None) is not None:
            cross_correlations = config["fits"]["cross correlations"].split(" ")
        else:
            cross_correlations = []

        export_files_auto = []

        # Set because there can be repeated values.
        for auto_correlation in set(auto_correlations):
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            export_file = self.paths.exp_cf_fname(absorber, region, absorber2, region2)
            export_files_auto.append(export_file)

            metals_file = self.paths.metal_fname(absorber, region, absorber2, region2)
            distortion_file = self.paths.dmat_fname(
                absorber, region, absorber2, region2
            )

            args = DictUtils.merge_dicts(
                config,
                {
                    "fits": {
                        "extra args": {
                            "vega_auto": {
                                "data": {
                                    "name": f"{absorber}{region}x{absorber2}{region2}",
                                    "filename": export_file,
                                    "tracer1": absorber_igm[absorber],
                                    "tracer2": absorber_igm[absorber2],
                                    "tracer1-type": "continuous",
                                    "tracer2-type": "continuous",
                                },
                                "metals": {
                                    "filename": metals_file,
                                },
                            }
                        }
                    }
                },
            )
            input_files.append(export_file)

            if config["fits"].get("distortion", True):
                args["fits"]["extra args"]["vega_auto"]["data"][
                    "distortion-file"
                ] = distortion_file
                input_files.append(distortion_file)

            vega_args = self.generate_extra_args(
                config=args,
                section="fits",
                extra_args=dict(),
                command="vega_auto.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
                region2=region2,
                absorber2=absorber2,
            )

            filename = self.paths.fit_auto_fname(absorber, region, absorber2, region2)
            self.write_ini(vega_args, filename)
            ini_files.append(str(filename))

            if vega_args.get("metals", None) is not None:
                input_files.append(metals_file)

        export_files_cross = []

        # Set because there can be repeated values.
        for cross_correlation in set(cross_correlations):
            absorber, region = cross_correlation.split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            export_file = self.paths.exp_xcf_fname(absorber, region)
            export_files_cross.append(export_file)

            metals_file = self.paths.xmetal_fname(absorber, region)
            distortion_file = self.paths.xdmat_fname(absorber, region)

            args = DictUtils.merge_dicts(
                config,
                {
                    "fits": {
                        "extra args": {
                            "vega_cross": {
                                "data": {
                                    "name": f"qsox{absorber}{region}",
                                    "tracer1": "QSO",
                                    "tracer2": absorber_igm[absorber],
                                    "tracer1-type": "discrete",
                                    "tracer2-type": "continuous",
                                    "filename": export_file,
                                },
                                "metals": {
                                    "filename": metals_file,
                                },
                            }
                        }
                    }
                },
            )
            input_files.append(export_file)

            if self.config["fits"].get("distortion", True):
                args["fits"]["extra args"]["vega_cross"]["data"][
                    "distortion-file"
                ] = distortion_file
                input_files.append(distortion_file)

            vega_args = self.generate_extra_args(
                config=args,
                section="fits",
                extra_args=dict(),
                command="vega_cross.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
            )

            filename = self.paths.fit_cross_fname(absorber, region)
            self.write_ini(vega_args, filename)
            ini_files.append(str(filename))

            if vega_args.get("metals", None) is not None:
                input_files.append(metals_file)

        # Now the main file
        args = DictUtils.merge_dicts(
            config,
            {
                "fits": {
                    "extra args": {
                        "vega_main": {
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
            },
        )

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
            extra_args=dict(),
            command="vega_main.py",  # The .py needed to make use of same function
        )

        if (
            "fiducial" not in vega_args
            or vega_args["fiducial"].get("filename", None) is None
        ):
            vega_args["fiducial"]["filename"] = "PlanckDR16/PlanckDR16.fits"

        filename = self.paths.fit_main_fname()
        self.write_ini(vega_args, filename)

        return input_files

    def check_existing_output_file(
        self, file: Path, job_name: str, skip_sent: bool, overwrite: bool, system: str
    ) -> bool:
        """
        Method to check the status of an output file.

        Args:
            file: Path to the file we want to check.
            job_name: Job name used for logging.
            skip_sent: Whether sent jobs should be skipped.
            overwrite: Whether existing runs should be overwritten.
            system: System where the jobs are being run.

        Returns:
            True if the file exists and the run has to be skipped. False if the file does
                not exist or the run has to be overwritten. It will raise a
                FileExistsError if the overwrite option is disabled, a file exists and
                skip_sent is disabled.
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
    """Class to define paths following the bookkeeper convention.

    Attributes:
        config (configparser.ConfigParser): Configuration used to build paths.

    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration to be used.
        """
        self.config = config

    @property
    def healpix_data(self) -> Path:
        """Path: Location of healpix data."""
        if self.config["data"].get("healpix data", "") in (None, ""):
            if "everest" in self.config["data"]["release"]:
                return Path("/global/cfs/cdirs/desi/spectro/redux/everest/healpix/")
            elif "fuji" in self.config["data"]["release"]:
                return Path("/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/")
            elif "guadalupe" in self.config["data"]["release"]:
                return Path("/global/cfs/cdirs/desi/spectro/redux/guadalupe/healpix/")
            elif (
                "fugu" in self.config["data"]["release"]
                or "fujilupe" in self.config["data"]["release"]
            ):
                return Path("/global/cfs/cdirs/desi/science/lya/fugu_healpix/healpix")
            elif "himalayas" in self.config["data"]["release"]:
                return Path("/global/cfs/cdirs/desi/spectro/redux/himalayas/healpix/")
            elif "iron" in self.config["data"]["release"]:
                return Path("/global/cfs/cdirs/desi/spectro/redux/iron/healpix/")
            elif "v9." in self.config["data"]["release"]:
                version = self.config["data"]["release"].split(".")[1]
                if self.config["data"]["survey"] == "raw":
                    raise ValueError(
                        "Not healpix data for raw realisations with "
                        "no assigned survey (quickquasars realisation)"
                    )
                else:
                    return (
                        Path(
                            f"/global/cfs/cdirs/desi/mocks/lya_forest/develop/london/"
                            f"qq_desi/v9.{version}/"
                        )
                        / self.config["data"]["release"]
                        / self.config["data"]["survey"]
                        / "spectra-16"
                    )
            else:
                raise ValueError(
                    "Could not set healpix data location for provided release: ",
                    self.config["data"]["release"],
                )
        else:
            _path = Path(self.config["data"]["healpix data"])
            if not _path.is_dir():
                raise FileNotFoundError("Directory does not exist ", _path)
            return _path

    @property
    def transmission_data(self) -> Path:
        """Path: Location of transmission LyaCoLoRe mock data."""
        if (
            self.config["data"]["healpix data"] not in ("", None)
            and Path(self.config["data"]["healpix data"]).is_dir()
        ):
            return Path(self.config["data"]["healpix data"])
        elif "v9." not in self.config["data"]["release"]:
            raise ValueError("Trying to access transmission data for non-mock release")
        version = self.config["data"]["release"].split(".")[1]
        return (
            Path(f"/global/cfs/cdirs/desi/mocks/lya_forest/london/v9.{version}/")
            / self.config["data"]["release"]
        )

    @property
    def survey_path(self) -> Path:
        """Path: Survey path following bookkeeper convention."""
        return (
            Path(self.config["data"]["bookkeeper dir"])
            / self.config["data"]["release"]
            / self.config["data"]["survey"]
        )

    @property
    def run_path(self) -> Path:
        """Give full path to the bookkeeper run.

        Returns:
            Path
        """
        # Potentially could add fits things here
        # Start from the bottom (correlations)
        if self.config.get("correlations", dict()).get("delta extraction", "") not in (
            "",
            None,
        ):
            delta_name = self.config["correlations"]["delta extraction"]
        elif self.config.get("fits", dict()).get("delta extraction", "") not in (
            "",
            None,
        ):
            delta_name = self.config["fits"]["delta extraction"]
        else:
            try:
                delta_name = self.continuum_tag
            except ValueError:
                raise ValueError("Error reading delta extraction section.")

        return self.survey_path / self.catalog_name / delta_name

    @property
    def continuum_fitting_mask(self) -> Path:
        """Path: file with masking used in continuum fitting."""
        return self.run_path / "configs" / "continuum_fitting_mask.txt"

    @property
    def correlations_path(self) -> Path:
        """Give full path to the correlation runs.

        Returns:
            Path
        """
        if self.config.get("fits", dict()).get("correlation run name", "") not in (
            "",
            None,
        ):
            correlation_name = self.config["fits"]["correlation run name"]
        else:
            correlation_name = self.config["correlations"]["run name"]

        return self.run_path / "correlations" / correlation_name

    @property
    def fits_path(self) -> Path:
        """Give full path to the fits runs.

        Returns:
            Path
        """
        fit_name = self.config["fits"]["run name"]

        return self.correlations_path / "fits" / fit_name

    @property
    def delta_config_file(self) -> Path:
        """Default path to the deltas config file inside bookkeeper.

        Returns
            Path
        """
        return self.run_path / "configs" / "bookkeeper_config.yaml"

    @property
    def defaults_file(self) -> Path:
        """Location of the defaults file inside the bookkeeper.

        Returns
            Path
        """
        return self.delta_config_file.parent / "defaults.yaml"

    @property
    def correlation_config_file(self) -> Path:
        """Default path to the correlation config file inside bookkeeper.

        Returns
            Path
        """
        return self.correlations_path / "configs" / "bookkeeper_config.yaml"

    @property
    def fit_config_file(self) -> Path:
        """Default path to the fit config file inside bookkeeper

        Returns
            Path
        """
        return self.fits_path / "configs" / "bookkeeper_config.yaml"

    def get_catalog_from_field(self, field: str) -> Path:
        """Method to obtain catalogs given a catalog name in config file.

        It will check if the file exists, expand the full path if only the filename
        is given and raise a ValueError if the file does not exist.

        Args:
            field (str): whether to use catalog, catalog tracer fields, dla fields or
                bal fields. (Options: ["catalog", "catalog_tracer", "dla", "bal"])

        Returns:
            Path: catalog to be used.
        """
        if field == "dla":
            if (
                self.config["delta extraction"].get("dla catalog", None)
                not in ("", None)
            ) and Path(
                self.config["delta extraction"].get("dla catalog", "")
            ).is_file():
                catalog = Path(self.config["delta extraction"]["dla catalog"])
            else:
                field_value = self.config["delta extraction"]["dla"]
                catalog = get_dla_catalog(
                    self.config["data"]["release"],
                    self.config["data"]["survey"],
                    version=field_value,
                )
        elif field == "bal":
            if (
                self.config["delta extraction"].get("bal catalog", None)
                not in ("", None)
            ) and Path(
                self.config["delta extraction"].get("bal catalog", "")
            ).is_file():
                catalog = Path(self.config["delta extraction"]["bal catalog"])
            else:
                catalog = self.get_catalog_from_field("catalog")
        elif field == "catalog_tracer":
            if (
                self.config["correlations"].get("catalog tracer", None)
                not in ("", None)
            ) and Path(self.config["correlations"].get("catalog tracer", "")).is_file():
                catalog = Path(self.config["correlations"]["catalog tracer"])
            else:
                catalog = self.get_catalog_from_field("catalog")
        else:
            # Here is the normal catalog
            if (self.config["data"][field] is None) or Path(
                self.config["data"][field]
            ).is_file():
                catalog = Path(self.config["data"][field])
            elif "/" in self.config["data"][field]:
                raise ValueError("Invalid catalog name", self.config["data"][field])
            elif "LyaCoLoRe" in self.config["data"][field]:
                if self.config["data"]["survey"] == "raw":
                    catalog = self.transmission_data / "master.fits"
                else:
                    catalog = self.healpix_data.parent / "zcat.fits"
            else:
                catalog = get_quasar_catalog(
                    self.config["data"]["release"],
                    self.config["data"]["survey"],
                    self.config["data"]["catalog"],
                    bal=self.config.get("delta extraction", dict()).get("bal", 0) != 0,
                )

        if catalog.is_file():
            return catalog
        else:
            raise FileNotFoundError("catalog not found in path: ", str(catalog))

    @property
    def catalog(self) -> Path:
        """catalog to be used for deltas."""
        return self.get_catalog_from_field("catalog")

    @property
    def catalog_dla(self) -> Path:
        """DLA catalog to be used."""
        return self.get_catalog_from_field("dla")

    @property
    def catalog_bal(self) -> Path:
        """catalog to be used for BAL masking."""
        return self.get_catalog_from_field("bal")

    @property
    def catalog_tracer(self) -> Path:
        """catalog to be used for cross-correlations with quasars"""
        return self.get_catalog_from_field("catalog_tracer")

    @property
    def catalog_name(self) -> str:
        """Returns catalog standardize name."""
        name = Path(self.config["data"]["catalog"]).name
        if Path(self.config["data"]["catalog"]).is_file():
            return self.get_fits_file_name(Path(self.config["data"]["catalog"]))
        else:
            return name

    @staticmethod
    def get_fits_file_name(file: Path | str) -> str:
        name = Path(file).name
        if name[-8:] == ".fits.gz":
            return name[:-8]
        elif name[-5:] == ".fits":
            return name[:-5]
        else:
            raise ValueError("Unrecognized fits catalog filename", name)

    @property
    def continuum_tag(self) -> str:
        """str: tag defining the continuum fitting parameters used."""
        if self.config.get("delta extraction") is None:
            raise ValueError(
                "To get continuum tag delta extraction section should be defined."
            )

        prefix = self.config["delta extraction"]["prefix"]
        calib = self.config["delta extraction"]["calib"]
        calib_region = self.config["delta extraction"].get("calib region", 0)
        if calib == 0:
            calib_region = 0
        suffix = self.config["delta extraction"]["suffix"]
        bal = self.config["delta extraction"]["bal"]
        dla = self.config["delta extraction"]["dla"]

        return "{}_{}.{}.{}.{}_{}".format(prefix, calib, calib_region, dla, bal, suffix)

    @staticmethod
    def compare_config_files(
        file1: str | Path,
        file2: str | Path,
        section: Optional[str] = None,
        ignore_fields: List[str] = [],
    ) -> Dict:
        """Compare two config files to determine if they are the same.

        Args:
            file1
            file2
            section: Section of the yaml file to compare.
            ignore_fields: Fields to ignore in the comparison
        """
        with open(file1, "r") as f:
            config1 = yaml.safe_load(f)

        with open(file2, "r") as f:
            config2 = yaml.safe_load(f)

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

    def check_delta_directories(self) -> None:
        """Method to create basic directories in run directory."""
        for folder in ("scripts", "correlations", "logs", "results", "configs"):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)

    def check_correlation_directories(self) -> None:
        """Method to create basic directories in correlations directory."""
        for folder in ("scripts", "results", "fits", "logs", "configs"):
            (self.correlations_path / folder).mkdir(exist_ok=True, parents=True)

    def check_fit_directories(self) -> None:
        """Method to create basic directories in fits directory."""
        for folder in ("scripts", "results", "logs", "configs", "results/sampler"):
            (self.fits_path / folder).mkdir(exist_ok=True, parents=True)

    def deltas_path(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """Method to obtain the path to deltas output.

        Args:
            region: Region used (in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas directory.
        """
        if calib_step is not None:
            return self.run_path / "results" / f"calibration_{calib_step}" / "Delta"
        else:
            if region is None:
                raise ValueError("Invalid region provided: ", region)
            region = Bookkeeper.validate_region(region)
            return self.run_path / "results" / region / "Delta"

    def deltas_log_path(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """Method to get the path to deltas log.

        Args:
            region: Region used (in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas direct
        """
        deltas_path = self.deltas_path(region, calib_step)
        return deltas_path.parent / "Log"

    def delta_attributes_file(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """Method to get the path to deltas attributes file.

        Args:
            region: Region used (should be in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to delta attributes file
        """
        return (
            self.deltas_log_path(region=region, calib_step=calib_step)
            / "delta_attributes.fits.gz"
        )

    def copied_calib_attributes(self, step: int) -> Path | None:
        """Method to get path to delta attributes for calibration if it
        appears as the one to be read in the bookkeeper (instead of computing
        it).

        Args:
            step: Current calibration step.
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
        """Method to get path to deltas files and delta attributes if it
        appears in the bookkeeper config file (instead of computing it)

        Args:
            region: Region to look in the configuration file.
        """
        files = self.config["delta extraction"].get("deltas", dict()).get(region, None)

        if files is None:
            parent = self.config["delta extraction"].get("link deltas", None)
            if parent is not None:
                parent = Path(parent)

                deltas = parent / region / "Delta"
                attributes = parent / region / "Log/delta_attributes.fits.gz"

                if deltas.is_dir() and attributes.is_file():
                    files = (
                        str(parent / region / "Delta")
                        + " "
                        + str(parent / region / "Log/delta_attributes.fits.gz")
                    )

        if files is not None:
            files = files.split(" ")
            if len(files) != 2:
                raise ValueError(
                    f"{region}: Invalid format for deltas provided. Format should be "
                    "two spaced-separated paths deltas_directory delta_attributes_file"
                )
            if not Path(files[0]).is_dir():
                raise FileNotFoundError(
                    f"{region}: Invalid deltas directory provided", files[0]
                )
            if not Path(files[1]).is_file():
                raise FileNotFoundError(
                    f"{region}: Invalid attributes file provided", files[1]
                )
            logger.info(f"{region}: Using deltas from file:\n\t{str(files[0])}")
            logger.info(
                f"{region}: Using delta attributes from file:\n\t{str(files[1])}"
            )
        else:
            logger.info(
                f"{region}: No files provided to copy, deltas will be computed."
            )
        return files

    def copied_correlation_file(
        self,
        subsection: str,
        absorber: str,
        region: str,
        absorber2: Optional[str],
        region2: Optional[str],
        filename: Optional[str],
    ) -> Path:
        """Method to get a correlation file to copy given in the bookkeeper config

        Args:
            subsection: Subsection of config file to read
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber
            filename: Filename to tried being read from linked correlations.

        Returns:
            Path: Path to file
        """
        if absorber2 is None:
            name = f"{absorber}{region}"
            qso = "qso_"
        else:
            name = f"{absorber}{region}_{absorber2}{region2}"
            qso = ""

        file = self.config["correlations"].get(subsection, dict()).get(name, None)

        if file is None:
            parent = self.config["correlations"].get("link correlations", None)
            if parent is not None:
                parent = Path(parent)
                folder_name = (qso + name).replace("(", "").replace(")", "")
                corr = parent / folder_name / filename

                if corr.is_file():
                    file = corr
        
        if file is None and subsection in ("metal matrices", "xmetal matrices"):
            parent = self.config["correlations"].get("link metals", None)
            if parent is not None:
                parent = Path(parent)
                folder_name = (qso + name).replace("(", "").replace(")", "")
                corr = parent / folder_name / filename

                if corr.is_file():
                    file = corr
        
        if file is None and subsection in ("distortion matrices", "xdistortion matrices"):
            parent = self.config["correlations"].get("link metals", None)
            if parent is not None:
                parent = Path(parent)
                folder_name = (qso + name).replace("(", "").replace(")", "")
                corr = parent / folder_name / filename

                if corr.is_file():
                    file = corr


        if file is not None:
            if not Path(file).is_file():
                raise FileNotFoundError(
                    f"{subsection} {name}: Invalid file provided in config", file
                )
            logger.info(f"{subsection} {name}: Using from file:\n\t{str(file)}")
        else:
            logger.info(
                f"{subsection} {name}: No file provided to copy, it will be computed."
            )

        return file

    def cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """Method to get the path to a forest-forest correlation file.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to correlation file.
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
        """Method to get the path to a distortion matrix file for forest-forest
        correlations.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to correlation file.
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
        """Method to get the path to a metal distortion matrix file for forest-forest
        correlations.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to correlation file.
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
        """Method to get the path to a forest-forest correlation export file.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.cf_fname(absorber, region, absorber2, region2)
        return cor_file.parent / f"cf_exp.fits.gz"

    def xcf_fname(self, absorber: str, region: str) -> Path:
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to correlation file.
        """
        return (
            self.correlations_path
            / "results"
            / f"qso_{absorber.replace('(', '').replace(')', '')}{region}"
            / f"xcf.fits.gz"
        )

    def xdmat_fname(self, absorber: str, region: str) -> Path:
        """Method to get the path to a distortion matrix file for forest-quasar
        correlations.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        return self.xcf_fname(absorber, region).parent / f"xdmat.fits.gz"

    def xmetal_fname(self, absorber: str, region: str) -> Path:
        """Method to get the path to a metal distortion matrix file for forest-quasar
        correlations.

        Args:
            region (str): Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        parent = self.xcf_fname(absorber, region).parent
        if (parent / "xmetal.fits.gz").is_file():
            return parent / "xmetal.fits.gz"
        else:
            return parent / "xmetal.fits"

    def exp_xcf_fname(self, absorber: str, region: str) -> Path:
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.xcf_fname(absorber, region)
        return cor_file.parent / f"xcf_exp.fits.gz"

    def fit_auto_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """Method to get te path to a given fit auto config file.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to fit config file.
        """
        region2 = region if region2 is None else region2
        absorber2 = absorber if absorber2 is None else absorber2

        name = (
            self.fits_path / "configs" / f"{absorber}{region}x{absorber2}{region2}.ini"
        )
        return name.parent / name.name.replace("(", "").replace(")", "")

    def fit_cross_fname(self, absorber: str, region: str) -> Path:
        """Method to get te path to a given fit cross config file.

        Args:
            region: Region where the correlation is computed.
            absorber: First absorber

        Returns:
            Path: Path to fit config file.
        """
        name = self.fits_path / "configs" / f"qsox{absorber}{region}.ini"
        return name.parent / name.name.replace("(", "").replace(")", "")

    def fit_main_fname(self) -> Path:
        """Method to get the path to the main fit config file.

        Returns:
            Path: Path to main fit config file.
        """
        return self.fits_path / "configs" / "main.ini"

    def fit_out_fname(self) -> Path:
        """Method to get the path to the fit output file.

        Returns:
            Path: Path to fit output file.
        """
        return self.fits_path / "results" / "fit_output.fits"

    def sampler_out_path(self) -> Path:
        """Method to get the path to the sampler output foler

        Returns:
            Path to sampler output folder.
        """
        return self.fits_path / "results" / "sampler"

    def fit_computed_params_out(self) -> Path:
        """Method to get the path of the computed zeff output file"""
        return self.fits_path / "results" / "computed_parameters.yaml"
