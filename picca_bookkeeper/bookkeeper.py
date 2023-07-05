import configparser
import copy
import filecmp
import logging
import shutil
import sys
from pathlib import Path
from typing import *

import yaml
from importlib_resources import files
from picca.constants import ABSORBER_IGM
from yaml import SafeDumper

from picca_bookkeeper import resources
from picca_bookkeeper.dict_utils import DictUtils
from picca_bookkeeper.tasker import ChainedTasker, Tasker, get_Tasker

logger = logging.getLogger(__name__)

# This converts Nones in dict into empty fields in yaml.
SafeDumper.add_representer(
    type(None),
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)

forest_regions = {
    "lya": {
        "lambda-rest-min": 1040.0,
        "lambda-rest-max": 1200.0,
    },
    "lyb": {
        "lambda-rest-min": 920.0,
        "lambda-rest-max": 1020.0,
    },
    "mgii_r": {
        "lambda-rest-min": 2900.0,
        "lambda-rest-max": 3120.0,
    },
    "ciii": {
        "lambda-rest-min": 1600.0,
        "lambda-rest-max": 1850.0,
    },
    "civ": {
        "lambda-rest-min": 1410.0,
        "lambda-rest-max": 1520.0,
    },
    "siv": {
        "lambda-rest-min": 1260.0,
        "lambda-rest-max": 1375.0,
    },
    "mgii_11": {
        "lambda-rest-min": 2600.0,
        "lambda-rest-max": 2760.0,
    },
    "mgii_h": {
        "lambda-rest-min": 2100.0,
        "lambda-rest-max": 2760.0,
    },
}

# Get absorbers in lowercase.
absorber_igm = dict((absorber.lower(), absorber) for absorber in ABSORBER_IGM)

config_file_sorting = ["general", "delta extraction", "correlations", "fits"]


def get_quasar_catalog(release, survey, catalog, bal=False) -> Path:  # pragma: no cover
    """Function to obtain a quasar catalog given different options

    Attributes:
        release (str): everest, fuji, guadalupe, fugu, ...
        survey (str): sv1, sv3, sv13, main, all
        catalog (str): redrock_v0, afterburn_v0, afterburn_v1, ...
        bal (bool): whether to search for catalogs with BALs included.
    """
    if release in ("everest", "fuji", "guadalupe", "fugu"):
        basedir = Path("/global/cfs/cdirs/desi/science/lya/early-3d/catalogs/qso")
    else:
        basedir = Path(
            "/global/cfs/cdirs/desi/users/cramirez/Continuum_fitting_Y1/catalogs/qso"
        )

    if bal:
        catalog += "-bal"

    catalog = basedir / release / survey / catalog
    for suffix in (".fits", ".fits.gz", "-bal.fits", "-bal.fits.gz"):
        if catalog.with_name(catalog.name + suffix).is_file():
            return catalog.with_name(catalog.name + suffix)
    else:
        raise FileNotFoundError(
            f"Could not find a compatible catalog inside the bookkeeper. "
            f"(Path: {catalog})"
        )


def get_dla_catalog(release, survey, version=1) -> Path:
    """Function to obtain a DLA catalog.

    Arguments:
        release (str): everest, fuji, guadalupe, fugu,...
        survey (str): sv1, sv3, sv13, main, all
        version (float): version of the catalog
    """
    if release in ("everest", "fuji", "guadalupe", "fugu"):
        basedir = Path("/global/cfs/cdirs/desi/science/lya/early-3d/catalogs/dla")
    else:
        basedir = Path(
            "/global/cfs/cdirs/desi/users/cramirez/Continuum_fitting_Y1/catalogs/dla"
        )

    catalog = basedir / release / survey / f"dla_catalog_v{version}"

    for suffix in (".fits", ".fits.gz"):
        if catalog.with_name(catalog.name + suffix):
            return catalog.with_name(catalog.name + suffix)
    else:
        raise FileNotFoundError(
            f"Could not find a compatible catalog in the bookkeeper. (Path: {catalog})"
        )


class Bookkeeper:
    """Class to generate Tasker objects which can be used to run different picca jobs.

    Attributes:
        config (configparser.ConfigParser): Configuration file for the bookkeeper.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        overwrite_config: bool = False,
        read_mode: bool = False,
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

        # If the calibration is taken from another place, we should
        # also load this other bookkeeper
        if config_type == "deltas" and self.config["delta extraction"].get(
            "calibration data", ""
        ) not in ("", None):
            try:
                self.calibration = Bookkeeper(
                    self.paths.run_path.parent
                    / self.config["delta extraction"].get("calibration data")
                )
            except Exception as e:
                raise Exception("Error loading calibration bookkeeper").with_traceback(
                    e.__traceback__
                )
        else:
            self.calibration = self

        if read_mode:
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
                    print(DictUtils.print_dict(comparison))
                    raise ValueError(
                        "delta extraction section of config file should match delta "
                        "extraction section from file already in the bookkeeper. "
                        "Unmatching items above"
                    )
            # Copy full bookkeeper.
            shutil.copyfile(
                config_path,
                self.paths.delta_config_file.parent / "bookkeeper_config_full.yaml",
            )

        if self.correlations is not None and config_type != "fits":
            config_corr = copy.deepcopy(self.config)

            config_corr["correlations"]["delta extraction"] = self.paths.continuum_tag

            config_corr.pop("delta extraction")

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
                    print(DictUtils.print_dict(comparison))
                    raise ValueError(
                        "correlations section of config file should match correlation section "
                        "from file already in the bookkeeper. Unmatching items above"
                    )
        if self.fits is not None:
            config_fit = copy.deepcopy(self.config)

            config_fit["fits"]["delta extraction"] = self.paths.continuum_tag
            config_fit["fits"]["correlation run name"] = self.config["correlations"][
                "run name"
            ]

            config_fit.pop("delta extraction")
            config_fit.pop("correlations")

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
                    print(DictUtils.print_dict(comparison))
                    raise ValueError(
                        "fits section of config file should match fits section "
                        "from file already in the bookkeeper. Unmatching items above."
                    )

        # Read defaults and check if they have changed.
        defaults_file = files(resources).joinpath(
            str(self.config["delta extraction"]["prefix"]) + ".yaml"
        )
        if not defaults_file.is_file():
            raise ValueError("Invalid prefix, no defaults file found.", defaults_file)

        self.defaults = yaml.safe_load(defaults_file.read_text())

        self.defaults_diff = dict()

        if self.paths.defaults_file.is_file():
            self.defaults_diff = PathBuilder.compare_config_files(
                self.paths.defaults_file, defaults_file,
            )
        else:
            self.defaults_diff = {}
            self.write_bookkeeper(
                self.defaults,
                self.paths.defaults_file,
            )

    @staticmethod
    def write_ini(config: Dict, file: Union[Path, str]) -> None:
        """Safely save a dictionary into an .ini file

        Args
            config: Dict to store as ini file.
            file: path where to store the ini.
        """
        config = DictUtils.convert_to_string(config)

        parser = configparser.ConfigParser()
        parser.read_dict(config)

        with open(file, "w") as file:
            parser.write(file)

    @staticmethod
    def write_bookkeeper(config: Dict, file: Union[Path, str]) -> None:
        """Method to write bookkeeper yaml file to file

        Args:
            config: Dict to store as yaml file.
            file: path where to store the bookkeeper.
        """
        correct_order = {
            "general": ["conda environment", "system", "slurm args"],
            "data": ["early dir", "healpix data", "release", "survey", "catalog"],
            "delta extraction": [
                "prefix",
                "calib",
                "calib region",
                "dla",
                "bal",
                "suffix",
                "calibration data",
                "mask file",
                "dla catalog",
                "bal catalog",
                "extra args",
                "slurm args",
            ],
            "correlations": [
                "delta extraction",
                "run name",
                "catalog tracer",
                "metal matrices",
                "xmetal matrices",
                "extra args",
                "slurm args",
            ],
            "fits": [
                "delta extraction",
                "correlation run name",
                "run name",
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

    @property
    def is_mock(self) -> bool:
        if "v9." in self.config["data"]["release"]:
            return True
        else:
            return False

    @staticmethod
    def validate_region(region: str) -> str:
        """Method to check if a region string is valid. Also converts it into lowercase.

        Will raise value error if the region is not in forest_regions.

        Args:
            region: Region (should be in forest_regions to pass the validation).
        """
        if region.lower() not in forest_regions:
            raise ValueError("Invalid region", region)

        return region.lower()

    @staticmethod
    def validate_absorber(absorber: str) -> str:
        """Method to check if a absorber is valid.

        Will raise value error if the absorber not in picca.absorbers.ABSORBER_IGM

        Args:
            absorber: Absorber to be used
        """
        if absorber.lower() not in absorber_igm:
            raise ValueError("Invalid absorber", absorber)

        return absorber.lower()

    def generate_slurm_header_extra_args(
        self,
        config: Dict,
        default_config: Dict,
        section: str,
        slurm_args: Dict,
        command: str,
        region: str = None,
        absorber: str = None,
        region2: str = None,
        absorber2: str = None,
    ) -> Dict:
        """Add extra slurm header args to the run.

        Args:
            config: bookkeeper config to look into.
            default_config: deafults config to look into.
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
        copied_args = copy.deepcopy(slurm_args)
        config = copy.deepcopy(config[section])
        defaults = copy.deepcopy(default_config[section])

        sections = ["general", command.split(".py")[0]]

        if absorber is not None:
            sections.append(command.split(".py")[0] + f"_{absorber}")
            if region is not None:
                sections[-1] = sections[-1] + region

        if absorber2 is not None:
            sections[-1] = sections[-1] + f"_{absorber2}"
            if region2 is not None:
                sections[-1] = sections[-1] + region2

        # We iterate over the sections from low to high priority
        # overriding the previous set values if there is a coincidence
        if "slurm args" in defaults.keys() and isinstance(defaults["slurm args"], dict):
            for section in sections:
                if section in defaults["slurm args"] and isinstance(
                    defaults["slurm args"][section], dict
                ):
                    args = DictUtils.merge_dicts(args, defaults["slurm args"][section])

        # We iterate over the sections from low to high priority
        # overriding the previous set values if there is a coincidence
        # Now with the values set by user
        if "slurm args" in config.keys() and isinstance(config["slurm args"], dict):
            for section in sections:
                if section in config["slurm args"] and isinstance(
                    config["slurm args"][section], dict
                ):
                    args = DictUtils.merge_dicts(args, config["slurm args"][section])

        # Copied args is the highest priority
        return DictUtils.merge_dicts(args, copied_args)

    def generate_extra_args(
        self,
        config: Dict,
        default_config: Dict,
        section: str,
        extra_args: Dict,
        command: str,
        region: str = None,
        absorber: str = None,
        region2: str = None,
        absorber2: str = None,
    ) -> Dict:
        """Add extra extra args to the run.

        Args:
            config: Section of the bookkeeper config to look into.
            default_config: Section of the deafults config to look into.
            section: Section name to look into.
            extra_args: extra args passed through the get_tasker method. They
                should be prioritized.
            command: picca command to be run.
            region: specify region where the command will be run.
            absorber: First absorber to use for correlations.
            region2: For scripts where two regions are needed.
            absorber2: Second absorber to use for correlations.
        """
        copied_args = copy.deepcopy(extra_args)
        config = copy.deepcopy(config[section])
        defaults = copy.deepcopy(default_config[section])

        sections = [command.split(".py")[0]]

        if absorber is not None:
            sections.append(command.split(".py")[0] + f"_{absorber}")
            if region is not None:
                sections[-1] = sections[-1] + region

        if absorber2 is not None:
            sections[-1] = sections[-1] + f"_{absorber2}"
            if region2 is not None:
                sections[-1] = sections[-1] + region2

        args = dict()
        if "extra args" in defaults.keys() and isinstance(defaults["extra args"], dict):
            for section in sections:
                if section in defaults["extra args"] and isinstance(
                    defaults["extra args"][section], dict
                ):
                    args = DictUtils.merge_dicts(args, defaults["extra args"][section])

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

        # Copied args is the highest priority
        return DictUtils.merge_dicts(args, copied_args)

    def generate_system_arg(self, system) -> str:
        if system is None:
            return copy.copy(
                self.config.get("general", dict()).get("system", "slurm_perlmutter")
            )
        else:
            return system

    def add_calibration_options(self, extra_args: Dict, calib_step: int = None) -> Dict:
        """Method to add calibration options to extra args

        Args:
            extra_args: Configuration to be used in the run.
            calib_step: Current calibration step, None if main run.

        Retursn:
            updated extra_args
        """
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
        elif self.config["delta extraction"]["calib"] == 1:
            # Second calibration step
            if calib_step is not None:
                if calib_step == 2:
                    num_corrections = (
                        int(extra_args.get("corrections").get("num corrections", 0)) + 1
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
                                    self.calibration.paths.delta_attributes_file(
                                        None, calib_step=1
                                    )
                                ),
                            },
                        },
                    )

            # Actual run (no calibration)
            else:
                if not self.calibration.paths.deltas_path(calib_step=2).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration_tasker "
                        "before running deltas."
                    )

                num_corrections = (
                    int(extra_args.get("corrections").get("num corrections", 0)) + 2
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
                                self.calibration.paths.delta_attributes_file(
                                    None, calib_step=1
                                )
                            ),
                        },
                        f"correction arguments {num_corrections - 1}": {
                            "filename": str(
                                self.calibration.paths.delta_attributes_file(
                                    None, calib_step=2
                                )
                            ),
                        },
                    },
                )

        elif self.config["delta extraction"]["calib"] == 2:
            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.paths.deltas_path(calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration tasker "
                        "before running deltas."
                    )
                num_corrections = (
                    int(extra_args.get("corrections").get("num corrections", 0)) + 1
                )

                extra_args = DictUtils.merge_dicts(
                    extra_args,
                    {
                        "corrections": {
                            "num corrections": num_corrections,
                            f"type {num_corrections - 1}": "CalibrationCorrection",
                        },
                        f"corrections arguments {num_corrections - 1}": {
                            "filename": str(
                                self.calibration.paths.delta_attributes_file(
                                    None, calib_step=1
                                )
                            ),
                        },
                    },
                )
        return extra_args

    def add_mask_options(self, extra_args: Dict, calib_step: int = None) -> Dict:
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
            extra_args = DictUtils(
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
        if (
            self.config["delta extraction"]["dla"] != 0
            or self.config["delta extraction"]["bal"] != 0
        ) and calib_step is None:
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
            else:
                raise ValueError(
                    "Invalid value for bal: ",
                    self.config["delta extraction"]["bal"],
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
            if deltas_config_dict.get("expected flux").get("input directory", 0) == 0:
                deltas_config_dict["expected flux"][
                    "input directory"
                ] = self.paths.healpix_data

    def get_raw_deltas_tasker(
        self,
        region: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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

        Returns:
            Tasker: Tasker object to run delta extraction.
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

        command = "picca_convert_transmission.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="delta extraction",
            extra_args=extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="delta extraction",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"raw_deltas_{region}"

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

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.run_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_delta_extraction_tasker(
        self,
        region: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        calib_step: int = None,
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

        Returns:
            Tasker: Tasker object to run delta extraction.
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

        command = "picca_delta_extraction.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="delta extraction",
            extra_args=extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="delta extraction",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"delta_extraction_{region}"
        if calib_step is not None:
            job_name += "_calib_step_" + str(calib_step)
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
                    "type": "DesisimMocks"
                    if "v9." in self.config["data"]["release"]
                    else "DesiHealpix",
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

        deltas_config_dict = self.add_calibration_options(
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
            deltas_config_dict.get("data").update({"max num spec": 1000})

        return get_Tasker(
            updated_system,
            command=command,
            command_args={"": str(config_file.resolve())},
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.run_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_calibration_extraction_tasker(
        self,
        system: str = None,
        debug: str = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
    ) -> Tasker:
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

        Returns:
            Tasker: Tasker object to run delta extraction for calibration.
        """
        steps = []
        region = self.config["delta extraction"].get("calib region", 0)
        if region in (0, "", None):
            raise ValueError("Calibration region not defined in config file.")
        else:
            region = self.validate_region(region)

        if self.config["delta extraction"]["calib"] not in [0, 1, 2, 3, 10]:
            raise ValueError(
                "Invalid calib value in config file. (Valid values are 0 1 2 3 10)"
            )
        elif self.config["delta extraction"]["calib"] in (1,):
            steps.append(
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    slurm_header_extra_args=slurm_header_extra_args,
                    extra_args=extra_args,
                    calib_step=1,
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
                )
            )
        elif self.config["delta extraction"]["calib"] in (2, 3, 10):
            steps = (
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    slurm_header_extra_args=slurm_header_extra_args,
                    extra_args=extra_args,
                    calib_step=1,
                ),
            )
        else:
            raise ValueError("Trying to run calibration with calib=0 in config file.")

        return ChainedTasker(taskers=steps)

    def get_cf_tasker(
        self,
        region: str = "lya",
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_cf.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
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
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"cf_{absorber}{region}_{absorber2}{region2}"

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
            "out": str(self.paths.cf_fname(absorber, region, absorber2, region2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

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

        self.paths.cf_fname(absorber, region, absorber2, region2).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_dmat_tasker(
        self,
        region: str = "lya",
        region2: str = None,
        absorber: str = "LYA",
        absorber2: str = None,
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_dmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
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
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"dmat_{absorber}{region}_{absorber2}{region2}"

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
            "out": str(self.paths.dmat_fname(absorber, region, absorber2, region2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args,
                dict(qos="debug", time="00:30:00"),
            )
            args = DictUtils.merge_dicts(args, dict(nspec=1000))

        self.paths.dmat_fname(absorber, region, absorber2, region2).parent.mkdir(
            exist_ok=True,
            parents=True,
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_cf_exp_tasker(
        self,
        region: str = "lya",
        region2: str = None,
        absorber: str = "LYA",
        absorber2: str = None,
        system: str = None,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        no_dmat: bool = False,
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
            no_dmat: Do not use distortion matrix.

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
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_export.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
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
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"cf_exp_{absorber}{region}_{absorber2}{region2}"

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
            "out": str(self.paths.exp_cf_fname(absorber, region, absorber2, region2)),
        }
        if not no_dmat:
            args["dmat"] = str(
                self.paths.dmat_fname(absorber, region, absorber2, region2)
            )

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if (
            self.paths.config["data"]["survey"] == "main"
            or self.paths.config["data"]["survey"] == "all"
        ):
            args["blind-corr-type"] = "lyaxlya"

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_metal_tasker(
        self,
        region: str = "lya",
        region2: str = None,
        absorber: str = "LYA",
        absorber2: str = None,
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        # If metal matrices are provided, we just copy them into the bookkeeper
        # as if they were computed.
        copy_metal_matrix = self.paths.copied_metal_matrix(
            absorber, region, absorber2, region2
        )
        if copy_metal_matrix is not None:
            filename = self.paths.metal_fname(absorber, region, absorber2, region2)
            filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(
                copy_metal_matrix,
                filename,
            )
            return

        command = "picca_metal_dmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
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
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"metal_{absorber}{region}_{absorber2}{region2}"

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
            "out": str(self.paths.metal_fname(absorber, region, absorber2, region2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args,
                dict(qos="debug", time="00:30:00"),
            )
            args = DictUtils.merge_dicts(args, dict(nspec=1000))

        self.paths.metal_fname(absorber, region, absorber2, region2).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_xcf_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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

        command = "picca_xcf.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xcf_{absorber}{region}"

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
            "out": str(self.paths.xcf_fname(absorber, region)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )

        self.paths.xcf_fname(absorber, region).parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_xdmat_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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

        Returns:
            Tasker: Tasker object to run forest-quasar distortion matrix.
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

        command = "picca_xdmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xdmat_{absorber}{region}"

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
            "out": str(self.paths.xdmat_fname(absorber, region)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if debug:  # pragma: no cover
            slurm_header_args = DictUtils.merge_dicts(
                slurm_header_args, dict(qos="debug", time="00:30:00")
            )
            args = DictUtils.merge_dicts(
                args,
                dict(nspec=1000),
            )
        self.paths.xdmat_fname(absorber, region).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_xcf_exp_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
        no_dmat: bool = False,
    ) -> Tasker:
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
            no_dmat: Do not use disortion matrix

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

        command = "picca_export.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xcf_exp_{absorber}{region}"

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
            "out": str(self.paths.exp_xcf_fname(absorber, region)),
            "blind-corr-type": "qsoxlya",
        }
        if not no_dmat:
            args["dmat"] = str(self.paths.xdmat_fname(absorber, region))

        args = DictUtils.merge_dicts(args, updated_extra_args)

        if (
            self.paths.config["data"]["survey"] == "main"
            or self.paths.config["data"]["survey"] == "all"
        ):
            args["blind-corr-type"] = "qsoxlya"

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options={"cpus-per-task": 1},
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_xmetal_tasker(
        self,
        region: str = "lya",
        absorber: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        extra_args: Dict = dict(),
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

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        copy_metal_matrix = self.paths.copied_xmetal_matrix(absorber, region)
        if copy_metal_matrix is not None:
            filename = self.paths.xmetal_fname(absorber, region)
            filename.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(
                copy_metal_matrix,
                filename,
            )
            return

        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )

        command = "picca_metal_xdmat.py"

        updated_extra_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            extra_args=extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="correlations",
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xmetal_{absorber}{region}"

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
            "out": str(self.paths.xmetal_fname(absorber, region)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

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

        self.paths.xmetal_fname(absorber, region).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            environmental_variables=environmental_variables,
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.correlations_path / f"logs/jobids.log",
            wait_for=wait_for,
        )

    def get_fit_tasker(
        self,
        auto_correlations: List[str] = [],
        cross_correlations: List[str] = [],
        system: str = None,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        # vega_extra_args: Dict = dict(),
        slurm_header_extra_args: Dict = dict(),
    ) -> Tasker:
        """Method to get a Tasker object to run vega with correlation data.

        Args:
            auto_correlations: List of auto-correlations to include in the vega
                fits. The format of the strings should be 'lya.lya-lya.lya'.
                This is to allow splitting.
            cross_correlations: List of cross-correlations to include in the vega
                fits. The format of the strings should be 'lya.lya'.
            system: Shell to use for job. 'slurm_cori' to use slurm scripts on
                cori, 'slurm_perlmutter' to use slurm scripts on perlmutter,
                'bash' to run it in login nodes or computer shell.
                Default: None, read from config file.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            # vega_extra_args : Set extra options for each vega config file
            #     The format should be a dict of dicts: wanting to change
            #     "num masks" in "masks" section one should pass
            #     {'num masks': {'masks': value}}.
            slurm_header_extra_args: Change slurm header default options if
                needed (time, qos, etc...). Use a dictionary with the format
                {'option_name': 'option_value'}.

        Returns:
            Tasker: Tasker object to run vega.
        """
        if self.defaults_diff != {}:
            raise ValueError(
                "Default values changed since last run of the "
                f"bookkeeper. Remove the file:\n\n {self.paths.defaults_file} "
                "\n\n to be able to write jobs (with the new default "
                f"values\). Defaults diff:\n\n"
                f"{DictUtils.print_dict(self.defaults_diff)}"
            )
        updated_system = self.generate_system_arg(system)

        ini_files = []

        for auto_correlation in auto_correlations:
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "-", "."
            ).split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            vega_args = self.generate_extra_args(
                config=self.config,
                default_config=self.defaults,
                section="fits",
                extra_args=dict(),
                command="vega_auto.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
                region2=region2,
                absorber2=absorber2,
            )

            args = {
                "data": {
                    "filename": self.paths.exp_cf_fname(
                        absorber, region, absorber2, region2
                    ),
                },
                "metals": {
                    "filename": self.paths.metal_fname(
                        absorber, region, absorber2, region2
                    ),
                },
            }

            args = DictUtils.merge_dicts(args, vega_args)

            filename = self.paths.fit_auto_fname(absorber, region, absorber2, region2)
            self.write_ini(args, filename)

            ini_files.append(str(filename))

        for cross_correlation in cross_correlations:
            absorber, region = cross_correlation.split(".")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            vega_args = self.generate_extra_args(
                config=self.config,
                default_config=self.defaults,
                section="fits",
                extra_args=dict(),
                command="vega_cross.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
            )

            args = {
                "data": {
                    "filename": self.paths.exp_xcf_fname(absorber, region),
                },
                "metals": {
                    "filename": self.paths.xmetal_fname(absorber, region),
                },
            }

            args = DictUtils.merge_dicts(args, vega_args)

            filename = self.paths.fit_cross_fname(absorber, region)
            self.write_ini(args, filename)
            ini_files.append(str(filename))

        # Now the main file
        vega_args = self.generate_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="fits",
            extra_args=dict(),
            command="vega_main.py",  # The .py needed to make use of same function
        )

        args = {"output": {"filename": self.paths.fit_out_fname()}}

        if "fiducial" not in args:
            shutil.copy(
                files(resources).joinpath("fit_models/PlanckDR16.fits"),
                self.paths.fit_main_fname().parent / "PlanckDR16.fits",
            )
            args = DictUtils.merge_dicts(
                args,
                {
                    "fiducial": {
                        "filename": (
                            self.paths.fit_main_fname().parent / "PlanckDR16.fits"
                        )
                    }
                },
            )

        args = DictUtils.merge_dicts(args, vega_args)

        filename = self.paths.fit_main_fname()
        self.write_ini(args, filename)

        ini_files.append(str(filename))

        # Now slurm args
        command = "run_vega.py"
        updated_slurm_header_args = self.generate_slurm_header_extra_args(
            config=self.config,
            default_config=self.defaults,
            section="fits",
            slurm_args=slurm_header_extra_args,
            command=command,
        )
        job_name = "vega_fit"

        return get_Tasker(
            updated_system,
            command=command,
            command_args={"": str(self.paths.fit_main_fname().resolve())},
            slurm_header_args=updated_slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.fits_path / f"scripts/run_{job_name}.sh",
            jobid_log_file=self.paths.fits_path / f"logs/jobids.log",
            wait_for=wait_for,
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
            Path(self.config["data"]["early dir"])
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
        if self.config.get("correlations", dict()).get("delta extraction", "") != "":
            delta_name = self.config["correlations"]["delta extraction"]
        elif self.config.get("fits", dict()).get("delta extraction", "") != "":
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
        if self.config.get("fits", dict()).get("correlation run name", "") != "":
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
        return self.correlations_path / "configs" / "bookkeeper_correlation_config.yaml"

    @property
    def fit_config_file(self) -> Path:
        """Default path to the fit config file inside bookkeeper

        Returns
            Path
        """
        return self.fits_path / "configs" / "bookkeeper_fit_config.yaml"

    def get_catalog_from_field(self, field) -> Path:
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
                not (self.config["delta extraction"].get("dla catalog", "") is None)
                and Path(
                    self.config["delta extraction"].get("dla catalog", "")
                ).is_file()
            ):
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
                not (self.config["delta extraction"].get("bal catalog", "") is None)
                and Path(
                    self.config["delta extraction"].get("bal catalog", "")
                ).is_file()
            ):
                catalog = Path(self.config["delta extraction"]["bal catalog"])
            else:
                catalog = self.get_catalog_from_field("catalog")
        elif field == "catalog_tracer":
            if (
                not (self.config["correlations"].get("catalog tracer", "") is None)
                and Path(
                    self.config["correlations"].get("catalog tracer", "")
                ).is_file()
            ):
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
    def get_fits_file_name(file) -> str:
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
        suffix = self.config["delta extraction"]["suffix"]
        bal = self.config["delta extraction"]["bal"]
        dla = self.config["delta extraction"]["dla"]

        return "{}_{}_{}.{}.{}_{}".format(prefix, calib, calib_region, dla, bal, suffix)

    @staticmethod
    def compare_config_files(
        file1: Union[str, Path],
        file2: Union[str, Path],
        section: str = None,
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
        for folder in ("scripts", "correlations", "logs", "deltas", "configs"):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)

    def check_correlation_directories(self) -> None:
        """Method to create basic directories in correlations directory."""
        for folder in ("scripts", "correlations", "fits", "logs", "configs"):
            (self.correlations_path / folder).mkdir(exist_ok=True, parents=True)

    def check_fit_directories(self) -> None:
        """Method to create basic directories in fits directory."""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.fits_path / folder).mkdir(exist_ok=True, parents=True)

    def deltas_path(self, region: str = None, calib_step: int = None) -> Path:
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
            region = Bookkeeper.validate_region(region)
            return self.run_path / "results" / region / "Delta"

    def deltas_log_path(self, region: str, calib_step: int = None) -> Path:
        """Method to get the path to deltas log.

        Args:
            region: Region used (in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas direct
        """
        deltas_path = self.deltas_path(region, calib_step)
        return deltas_path.parent / "Log"

    def delta_attributes_file(self, region: str, calib_step: int = None) -> Path:
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

    def cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: str = None,
        region2: str = None,
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
            / f"{absorber}{region}_{absorber2}{region2}"
            / f"cf.fits.gz"
        )

    def dmat_fname(
        self,
        absorber: str,
        region: str,
        absorber2: str = None,
        region2: str = None,
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
        absorber2: str = None,
        region2: str = None,
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
        return (
            self.cf_fname(absorber, region, absorber2, region2).parent
            / f"metal.fits.gz"
        )

    def exp_cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: str = None,
        region2: str = None,
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

    def copied_metal_matrix(
        self,
        absorber: str,
        region: str,
        absorber2: str = None,
        region2: str = None,
    ) -> Union[Path, None]:
        """Method to get path to metal matrix if it appears in the bookkeeper config.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to metal matrix
        """
        metal_matrices = self.config["correlations"].get("xmetal matrices", None)

        if metal_matrices is None:
            matrix = None
        else:
            matrix = metal_matrices.get(
                f"{absorber}{region}_{absorber2}{region2}", None
            )

            if matrix is None:
                matrix = metal_matrices.get("all", None)

        name = f"{absorber}{region}_{absorber2}{region2}"
        if matrix is not None:
            if not Path(matrix).is_file:
                raise ValueError(f"{name}: Invalid metal matrix provided", matrix)
            logger.info(f"{name}: Using metal matrix from file:\n\t{str(matrix)}")
        else:
            logger.info(f"{name}: No metal matrix provided, it will be computed.")

        return matrix

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
            / f"qso_{absorber}{region}"
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
        return self.xcf_fname(absorber, region).parent / f"xmetal.fits.gz"

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

    def copied_xmetal_matrix(self, absorber: str, region: str) -> Union[Path, None]:
        """Method to get path to xmetal matrix if it appears in the bookkeeper config.

        Args:
            absorber: First absorber
            region: region of the forest used

        Returns:
            Path: Path to metal matrix
        """
        metal_matrices = self.config["correlations"].get("metal matrices", None)

        matrix = (
            self.config["correlations"]
            .get("xmetal matrices", dict())
            .get(f"{absorber}{region}", None)
        )
        if matrix is None:
            # If no specific metal matrix, use all (or None if not)
            matrix = (
                self.config["correlations"]
                .get("xmetal matrices", dict())
                .get("all", None)
            )

        name = f"{absorber}{region}"
        if matrix is not None:
            if not Path(matrix).is_file:
                raise ValueError(f"{name}: Invalid metal matrix provided", matrix)
            logger.info(f"{name}: Using xmetal matrix from file:\n\t{str(matrix)}")
        else:
            logger.info(f"{name}: No xmetal matrix provided, it will be computed.")

        return matrix

    def fit_auto_fname(
        self,
        absorber: str,
        region: str,
        absorber2: str = None,
        region2: str = None,
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
        return (
            self.fits_path / "configs" / f"{absorber}{region}x{absorber2}{region2}.ini"
        )

    def fit_cross_fname(self, absorber: str, region: str) -> Path:
        """Method to get te path to a given fit cross config file.

        Args:
            region: Region where the correlation is computed.
            absorber: First absorber

        Returns:
            Path: Path to fit config file.
        """
        return self.fits_path / "configs" / f"qsox{absorber}{region}.ini"

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
