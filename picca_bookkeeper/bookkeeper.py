from pathlib import Path
from typing import *
import configparser
import copy
import collections
import filecmp
import shutil
import os
import yaml
from importlib import resources

from picca.constants import ABSORBER_IGM

from picca_bookkeeper.tasker import get_Tasker, ChainedTasker, Tasker
from picca_bookkeeper import resources as bkp_resources

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


def get_quasar_catalog(release, survey, catalog, bal=False):  # pragma: no cover
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


def get_dla_catalog(release, survey, version=1):
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


def merge_dicts(dict1: Dict, dict2: Dict):
    """Merges two dictionaries recursively preserving values in dict2"""
    result = copy.deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])

    return result


def remove_matching(dict1: Dict, dict2: Dict):
    """Removes occurrences happening in two dictionaries."""
    result = copy.deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = remove_matching(result.get(key, {}), value)
        elif key in result:
            result.pop(key)

    return result


class Bookkeeper:
    """Class to generate Tasker objects which can be used to run different picca jobs.

    Attributes:
        config (configparser.ConfigParser): Configuration file for the bookkeeper.
    """

    def __init__(self, config_path: Union[str, Path], overwrite_config: bool = False):
        """
        Args:
            config_path (Path or str): Path to configuration file or to an already
                created run path.
            overwrite_config (bool, optional): overwrite bookkeeper config without
                asking if it already exists inside bookkeeper.
        """
        # Try to read the file of the folder
        config_path = Path(config_path)
        if not config_path.is_file():
            if (config_path / "configs/bookkeeper_config.yaml").is_file():
                config_path = config_path / "configs/bookkeeper_config.yaml"
            else:
                raise FileNotFoundError("Config file couldn't be found", config_path)

        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.BaseLoader)

        self.paths = PathBuilder(self.config)

        # Read defaults
        self.defaults = yaml.load(
            resources.read_text(bkp_resources, "defaults.yaml"),
            Loader=yaml.BaseLoader,
        )

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
            correlation_config_file = (
                self.paths.correlations_path / "configs" / "bookkeeper_config.yaml"
            )
            with open(correlation_config_file, "r") as f:
                correlation_config = yaml.load(f, Loader=yaml.BaseLoader)
            self.correlations = correlation_config["correlations"]
            self.config["correlations"] = self.correlations

        if config_type in ("fits", "correlations"):
            # In this case, delta extraction is not defined in the config file
            # and therefore, we should search for it.
            delta_config_file = (
                self.paths.run_path / "configs" / "bookkeeper_config.yaml"
            )
            with open(delta_config_file, "r") as f:
                delta_config = yaml.load(f, Loader=yaml.BaseLoader)
            self.delta_extraction = delta_config["delta extraction"]
            self.config["delta extraction"] = self.delta_extraction

        self.paths = PathBuilder(self.config)

        self.paths.check_delta_directories()

        if self.correlations is not None:
            self.paths.check_correlation_directories()

        if self.fits is not None:
            self.paths.check_fit_directories()

        # Potentially could add fits things here.
        # Copy bookkeeper configuration into destination
        # If bookkeeper included delta
        if config_type == "deltas":
            if not self.paths.delta_config_file.is_file():
                shutil.copyfile(config_path, self.paths.delta_config_file)
            elif filecmp.cmp(self.paths.delta_config_file, config_path):
                # If files are the same we can continue
                pass
            elif overwrite_config:
                # If we want to directly overwrite the config file in destination
                shutil.copyfile(config_path, self.paths.delta_config_file)
            else:
                if PathBuilder.compare_config_files(
                    config_path, self.paths.delta_config_file, "delta extraction"
                ):
                    shutil.copyfile(config_path, self.paths.delta_config_file)
                else:
                    raise ValueError(
                        "delta extraction section of config file should match delta "
                        "extraction section from file already in the bookkeeper."
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
                if PathBuilder.compare_config_files(
                    config_path,
                    self.paths.correlation_config_file,
                    "correlations",
                    ["delta extraction"],
                ):
                    self.write_bookkeeper(
                        config_corr, self.paths.correlation_config_file
                    )
                else:
                    raise ValueError(
                        "correlations section of config file should match correlation section "
                        "from file already in the bookkeeper."
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
                if PathBuilder.compare_config_files(
                    config_path,
                    self.paths.fit_config_file,
                    "fits",
                    ["delta extraction", "correlation run name"],
                ):
                    self.write_bookkeeper(config_fit, self.paths.fit_config_file)
                else:
                    raise ValueError(
                        "fits section of config file should match fits section "
                        "from file already in the bookkeeper."
                    )

        # Finally, if the calibration is taken from another place, we should
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

    def write_bookkeeper(self, config: Dict, file: Union[Path, str]):
        """Method to write bookkeeper yaml file to file

        Args:
            config: Dict to store as yaml file.
            file: path where to store the bookkeeper.
        """
        correct_order = {
            "general": ["conda environment", "system"],
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
                "picca args",
                "slurm args",
            ],
            "correlations": [
                "delta extraction",
                "run name",
                "catalog tracer",
                "picca args",
                "slurm args",
            ],
            "fits": [
                "delta extraction",
                "correlation run name",
                "run name",
                "picca args",
                "slurm args",
            ],
        }
        config = dict(
            sorted(config.items(), key=lambda s: list(correct_order).index(s[0]))
        )

        for key, value in config.items():
            config[key] = dict(
                sorted(value.items(), key=lambda s: correct_order[key].index(s[0]))
            )

        with open(file, "w") as f:
            yaml.dump(config, f, sort_keys=False)

    @property
    def is_mock(self):
        if "v9." in self.config["data"]["release"]:
            return True
        else:
            return False

    @staticmethod
    def validate_region(region: str):
        """Method to check if a region string is valid. Also converts it into lowercase.

        Will raise value error if the region is not in forest_regions.

        Args:
            region: Region (should be in forest_regions to pass the validation).
        """
        if region.lower() not in forest_regions:
            raise ValueError("Invalid region", region)

        return region.lower()

    @staticmethod
    def validate_absorber(absorber: str):
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
        slurm_args: Dict,
        command: str,
        region: str = None,
        absorber: str = None,
        region2: str = None,
        absorber2: str = None,
    ):
        """Add extra slurm header args to the run.

        Args:
            config: Section of the bookkeeper config to look into.
            default_config: Section of the deafults config to look into.
            slurm_args: Slurm args passed through the get_tasker method. They
                should be prioritized.
            command: Picca command to be run.
            region: Specify region where the command will be run.
            absorber: First absorber to use for correlations.
            region2: For scripts where two regions are needed.
            absorber2: Second absorber to use for correlations.
        """
        copied_args = copy.deepcopy(slurm_args)
        config = copy.deepcopy(config)
        defaults = copy.deepcopy(default_config)

        sections = ["general", command.split(".py")[0]]

        if absorber is not None:
            sections.append(command.split(".py")[0] + f"_{absorber}")
            if region is not None:
                sections[-1] = sections[-1] + region

        if absorber2 is not None:
            sections[-1] = sections[-1] + f"_{absorber2}"
            if region2 is not None:
                sections[-1] = sections[-1] + region2

        args = dict()
        # We iterate over the sections from low to high priority
        # overriding the previous set values if there is a coincidence
        if "slurm args" in defaults.keys() and isinstance(defaults["slurm args"], dict):
            for section in sections:
                if section in defaults["slurm args"] and isinstance(
                    defaults["slurm args"][section], dict
                ):
                    args = merge_dicts(args, defaults["slurm args"][section])

        # We iterate over the sections from low to high priority
        # overriding the previous set values if there is a coincidence
        # Now with the values set by user
        if "slurm args" in config.keys() and isinstance(config["slurm args"], dict):
            for section in sections:
                if section in config["slurm args"] and isinstance(
                    config["slurm args"][section], dict
                ):
                    args = merge_dicts(args, config["slurm args"][section])

        # Copied args is the highest priority
        return merge_dicts(args, copied_args)

    def generate_picca_extra_args(
        self,
        config: Dict,
        default_config: Dict,
        picca_args: Dict,
        command: str,
        region: str = None,
        absorber: str = None,
        region2: str = None,
        absorber2: str = None,
    ):
        """Add extra picca args to the run.

        Args:
            config: Section of the bookkeeper config to look into.
            default_config: Section of the deafults config to look into.
            picca_args: picca args passed through the get_tasker method. They
                should be prioritized.
            command: picca command to be run.
            region: specify region where the command will be run.
            absorber: First absorber to use for correlations.
            region2: For scripts where two regions are needed.
            absorber2: Second absorber to use for correlations.
        """
        copied_args = copy.deepcopy(picca_args)
        config = copy.deepcopy(config)
        defaults = copy.deepcopy(default_config)

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
        if "picca args" in defaults.keys() and isinstance(defaults["picca args"], dict):
            for section in sections:
                if section in defaults["picca args"] and isinstance(
                    defaults["picca args"][section], dict
                ):
                    args = merge_dicts(args, defaults["picca args"][section])

        if "picca args" in config.keys() and isinstance(config["picca args"], dict):
            for section in sections:
                if section in config["picca args"] and isinstance(
                    config["picca args"][section], dict
                ):
                    args = merge_dicts(args, config["picca args"][section])

            # remove args marked as remove_
            for section in sections:
                if "remove_" + section in config["picca args"] and isinstance(
                    config["picca args"]["remove_" + section], dict
                ):
                    args = remove_matching(
                        args, config["picca args"]["remove_" + section]
                    )

        # Copied args is the highest priority
        return merge_dicts(args, copied_args)

    def generate_system_arg(self, system):
        if system is None:
            return copy.copy(
                self.config.get("general", dict()).get("system", "slurm_perlmutter")
            )
        else:
            return system

    def get_raw_deltas_tasker(
        self,
        region: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run delta extraction.
        """
        region = self.validate_region(region)

        command = "picca_convert_transmission.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["delta extraction"],
            default_config=self.defaults["delta extraction"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["delta extraction"],
            default_config=self.defaults["delta extraction"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"raw_deltas_{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = merge_dicts(
                dict(nspec=1000), updated_picca_extra_args
            )
        else:
            qos = "regular"
            time = "03:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "object-cat": str(self.paths.catalog),
            "in-dir": str(self.paths.transmission_data),
            "out-dir": str(self.paths.deltas_path(region)),
        }

        self.paths.deltas_path(region).mkdir(exist_ok=True, parents=True)

        args = {
            **args,
            **{
                "lambda-rest-min": forest_regions[region]["lambda-rest-min"],
                "lambda-rest-max": forest_regions[region]["lambda-rest-max"],
            },
        }

        args = merge_dicts(args, updated_picca_extra_args)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_delta_extraction_tasker(
        self,
        region: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        picca_extra_args: Dict = dict(),
        calib_step: int = None,
    ):
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
            picca_extra_args: Set extra options for picca delta extraction.
                The format should be a dict of dicts: wanting to change
                "num masks" in "masks" section one should pass
                {'num masks': {'masks': value}}.
            calib_step: Calibration step. Default: None, no calibration

        Returns:
            Tasker: Tasker object to run delta extraction.
        """
        region = self.validate_region(region)

        command = "picca_delta_extraction.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["delta extraction"],
            default_config=self.defaults["delta extraction"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["delta extraction"],
            default_config=self.defaults["delta extraction"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"delta_extraction_{region}"
        if calib_step is not None:
            job_name += "_calib_step_" + str(calib_step)
        config_file = self.paths.run_path / f"configs/{job_name}.ini"

        # MASKS
        # add masks section if necessary
        updated_picca_extra_args = merge_dicts(
            dict(
                masks={
                    "num masks": 0,
                },
            ),
            updated_picca_extra_args,
        )
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

            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            updated_picca_extra_args["masks"]["num masks"] = prev_mask_number + 1
            updated_picca_extra_args["masks"][f"type {prev_mask_number}"] = "LinesMask"
            updated_picca_extra_args[f"mask arguments {prev_mask_number}"] = dict(
                filename=self.paths.continuum_fitting_mask,
            )

        # CORRECTIONS
        # add corrections section if necessary
        updated_picca_extra_args = merge_dicts(
            dict(
                corrections={
                    "num corrections": 0,
                }
            ),
            updated_picca_extra_args,
        )

        # update corrections section
        # here we are dealing with calibration runs
        # If there is no calibration, we should not have calib_steps
        if self.config["delta extraction"]["calib"] == "0" and calib_step is not None:
            raise ValueError("Trying to run calibration with calib = 0 in config file.")
        if self.config["delta extraction"]["calib"] not in ("0", "10"):
            if (
                "CalibrationCorrection"
                in updated_picca_extra_args["corrections"].values()
                or "IvarCorrection" in updated_picca_extra_args["corrections"].values()
            ):
                raise ValueError(
                    "Calibration corrections added by user with calib option != 10"
                )

        # Now we deal with dMdB20 option (calib = 1)
        if self.config["delta extraction"]["calib"] == "1":
            # only for calibrating runs
            if calib_step is not None:
                if calib_step == 2:
                    prev_n_corrections = int(
                        updated_picca_extra_args.get("corrections").get(
                            "num corrections", 0
                        )
                    )
                    updated_picca_extra_args["corrections"]["num corrections"] = (
                        prev_n_corrections + 1
                    )
                    updated_picca_extra_args["corrections"][
                        f"type {prev_n_corrections}"
                    ] = "CalibrationCorrection"
                    updated_picca_extra_args[
                        f"correction arguments {prev_n_corrections}"
                    ] = dict()
                    updated_picca_extra_args[
                        f"correction arguments {prev_n_corrections}"
                    ]["filename"] = str(
                        self.calibration.paths.delta_attributes_file(
                            None, calib_step=1
                        )  # @TODO fix this
                    )
            # actual run using with both corrections
            else:
                if not self.calibration.paths.deltas_path(calib_step=2).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration_tasker "
                        "before running deltas."
                    )
                prev_n_corrections = int(
                    updated_picca_extra_args.get("corrections").get(
                        "num corrections", 0
                    )
                )
                updated_picca_extra_args["corrections"]["num corrections"] = (
                    prev_n_corrections + 2
                )
                updated_picca_extra_args["corrections"][
                    f"type {prev_n_corrections}"
                ] = "CalibrationCorrection"
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections}"
                ] = dict()
                updated_picca_extra_args[f"correction arguments {prev_n_corrections}"][
                    "filename"
                ] = str(
                    self.calibration.paths.delta_attributes_file(None, calib_step=1)
                )

                updated_picca_extra_args["corrections"][
                    f"type {prev_n_corrections+1}"
                ] = "IvarCorrection"
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections+1}"
                ] = dict()
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections+1}"
                ]["filename"] = str(
                    self.calibration.paths.delta_attributes_file(None, calib_step=2)
                )
        elif self.config["delta extraction"]["calib"] == "2":
            # Set expected flux
            updated_picca_extra_args = merge_dicts(
                updated_picca_extra_args,
                {
                    "expected flux": {
                        "type": "Dr16FixedFudgeExpectedFlux",
                        "fudge value": 0,
                    },
                },
            )

            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.paths.deltas_path(calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration tasker "
                        "before running deltas."
                    )
                prev_n_corrections = int(
                    updated_picca_extra_args.get("corrections").get(
                        "num corrections", 0
                    )
                )
                updated_picca_extra_args["corrections"]["num corrections"] = (
                    prev_n_corrections + 1
                )
                updated_picca_extra_args["corrections"][
                    f"type {prev_n_corrections}"
                ] = "CalibrationCorrection"
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections}"
                ] = dict()
                updated_picca_extra_args[f"correction arguments {prev_n_corrections}"][
                    "filename"
                ] = str(
                    self.calibration.paths.delta_attributes_file(None, calib_step=1)
                )
        elif self.config["delta extraction"]["calib"] == "3":
            # Set expected flux
            updated_picca_extra_args = merge_dicts(
                updated_picca_extra_args,
                {
                    "expected flux": {
                        "type": "Dr16ExpectedFlux",
                    },
                },
            )

            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.paths.deltas_path(calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration tasker "
                        "before running deltas."
                    )
                prev_n_corrections = int(
                    updated_picca_extra_args.get("corrections").get(
                        "num corrections", 0
                    )
                )
                updated_picca_extra_args["corrections"]["num corrections"] = (
                    prev_n_corrections + 1
                )
                updated_picca_extra_args["corrections"][
                    f"type {prev_n_corrections}"
                ] = "CalibrationCorrection"
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections}"
                ] = dict()
                updated_picca_extra_args[f"correction arguments {prev_n_corrections}"][
                    "filename"
                ] = str(
                    self.calibration.paths.delta_attributes_file(None, calib_step=1)
                )

        # update masks sections if necessary
        if (
            self.config["delta extraction"]["dla"] != "0"
            or self.config["delta extraction"]["bal"] != "0"
        ) and calib_step is None:
            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            if self.config["delta extraction"]["dla"] != "0":
                if "DlaMask" in updated_picca_extra_args["masks"].values():
                    raise ValueError("DlaMask set by user with dla option != 0")

                updated_picca_extra_args["masks"][
                    f"type {prev_mask_number}"
                ] = "DlaMask"
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"] = dict()
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                    "filename"
                ] = self.paths.catalog_dla
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                    "los_id name"
                ] = "TARGETID"
                updated_picca_extra_args["masks"]["num masks"] = prev_mask_number + 1

            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            if self.config["delta extraction"]["bal"] != "0":
                if self.config["delta extraction"]["bal"] == "2":
                    if "BalMask" in updated_picca_extra_args["masks"].values():
                        raise ValueError("BalMask set by user with bal option !=0")

                    updated_picca_extra_args["masks"][
                        f"type {prev_mask_number}"
                    ] = "BalMask"
                    updated_picca_extra_args[
                        f"mask arguments {prev_mask_number}"
                    ] = dict()
                    updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                        "filename"
                    ] = self.paths.catalog_bal
                    updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                        "los_id name"
                    ] = "TARGETID"
                    updated_picca_extra_args["masks"]["num masks"] = (
                        prev_mask_number + 1
                    )
                else:
                    raise ValueError(
                        "Invalid value for bal: ",
                        self.config["delta extraction"]["bal"],
                    )

        # update expected flux section if necessary
        if self.config["delta extraction"]["prefix"] not in [
            "dMdB20",
            "raw",
            "true",
            "custom",
        ]:
            raise ValueError(
                f"Unrecognized continuum fitting prefix: "
                f"{self.config['delta extraction']['prefix']}"
            )
        elif self.config["delta extraction"]["prefix"] == "raw":
            raise ValueError(
                f"raw continuum fitting provided in config file, use "
                "get_raw_deltas_tasker instead"
            )
        elif self.config["delta extraction"]["prefix"] == "true":
            if (
                "expected flux" not in updated_picca_extra_args
                or "raw statistics file"
                not in updated_picca_extra_args["expected flux"]
                or updated_picca_extra_args["expected flux"]["raw statistics file"]
                in ("", None)
            ):
                raise ValueError(
                    f"Should define expected flux and raw statistics file in picca "
                    "args section in order to run TrueContinuum"
                )
            updated_picca_extra_args["expected flux"]["type"] = "TrueContinuum"
            if (
                updated_picca_extra_args.get("expected flux").get("input directory", 0)
                == 0
            ):
                updated_picca_extra_args["expected flux"][
                    "input directory"
                ] = self.paths.healpix_data

        # create config for delta_extraction options
        self.paths.deltas_path(region, calib_step).mkdir(parents=True, exist_ok=True)
        deltas_config_dict = {
            "general": {
                "overwrite": True,
                "out dir": str(
                    self.paths.deltas_path(region, calib_step).parent.resolve()
                )
                + "/",
            },
            "data": updated_picca_extra_args.get("data", {}),
            "corrections": updated_picca_extra_args.get(
                "corrections",
                {
                    "num corrections": 0,
                },
            ),
            "masks": updated_picca_extra_args.get(
                "masks",
                {
                    "num masks": 0,
                },
            ),
            "expected flux": updated_picca_extra_args.get("expected flux", {}),
        }

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            deltas_config_dict.get("data").update({"max num spec": 1000})
        else:
            qos = "regular"
            time = "03:00:00"

        # update data section with extra options
        # but leave them if provided by user
        deltas_config_dict["data"]["type"] = deltas_config_dict["data"].get(
            "type",
            "DesisimMocks"
            if "v9." in self.config["data"]["release"]
            else "DesiHealpix",
        )
        deltas_config_dict["data"]["catalogue"] = deltas_config_dict["data"].get(
            "catalogue", str(self.paths.catalog)
        )
        deltas_config_dict["data"]["input directory"] = deltas_config_dict["data"].get(
            "input directory", str(self.paths.healpix_data)
        )
        deltas_config_dict["data"]["lambda min rest frame"] = deltas_config_dict[
            "data"
        ].get("lambda min rest frame", forest_regions[region]["lambda-rest-min"])
        deltas_config_dict["data"]["lambda max rest frame"] = deltas_config_dict[
            "data"
        ].get("lambda max rest frame", forest_regions[region]["lambda-rest-max"])

        # update corrections section
        num_corrections = int(
            deltas_config_dict.get("corrections").get("num corrections")
        )
        assert isinstance(num_corrections, int) and num_corrections >= 0
        for i in range(num_corrections):
            correction_section = updated_picca_extra_args.get(
                f"correction arguments {i}", None
            )
            if correction_section is not None:
                deltas_config_dict.update(
                    {f"correction arguments {i}": correction_section}
                )

        # # update mask section
        # num_masks = deltas_config_dict.get("masks").get("num masks")
        # assert (isinstance(num_masks, int) and num_masks >= 0)
        # for i in range(num_masks):
        #     mask_section = updated_picca_extra_args.get(f"mask arguments {i}", None)
        #     if mask_section is not None:
        #         deltas_config_dict.update({
        #             f"mask arguments {i}": mask_section
        #         })

        deltas_config_dict = merge_dicts(deltas_config_dict, updated_picca_extra_args)

        # parse config
        deltas_config = configparser.ConfigParser()
        deltas_config.read_dict(deltas_config_dict)

        with open(config_file, "w") as file:
            deltas_config.write(file)

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args={"": str(config_file.resolve())},
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_calibration_extraction_tasker(
        self,
        system: str = None,
        debug: str = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
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

        if self.config["delta extraction"]["calib"] not in [
            "0",
            "1",
            "2",
            "3",
            "10",
        ]:
            raise ValueError(
                "Invalid calib value in config file. (Valid values are 0 1 2 3 4)"
            )
        elif self.config["delta extraction"]["calib"] in ("1",):
            steps.append(
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    slurm_header_extra_args=slurm_header_extra_args,
                    picca_extra_args=picca_extra_args,
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
                    picca_extra_args=picca_extra_args,
                    calib_step=2,
                )
            )
        elif self.config["delta extraction"]["calib"] in ("2", "3", "10"):
            steps = (
                self.get_delta_extraction_tasker(
                    region=region,
                    system=system,
                    debug=debug,
                    wait_for=wait_for,
                    slurm_header_extra_args=slurm_header_extra_args,
                    picca_extra_args=picca_extra_args,
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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_cf.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"cf_{absorber}{region}_{absorber2}{region2}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "02:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "out": str(self.paths.cf_fname(region, region2, absorber, absorber2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.cf_fname(region, region2, absorber, absorber2).parent.mkdir(
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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_dmat.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"dmat_{absorber}{region}_{absorber2}{region2}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "02:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "out": str(self.paths.dmat_fname(region, region2, absorber, absorber2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.dmat_fname(region, region2, absorber, absorber2).parent.mkdir(
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
            wait_for=wait_for,
        )

    def get_cf_exp_tasker(
        self,
        region: str = "lya",
        region2: str = None,
        absorber: str = "LYA",
        absorber2: str = None,
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        picca_extra_args: Dict = dict(),
        no_dmat: bool = False,
    ):
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
            debug: Whether to use debug options.
            wait_for: In NERSC, wait for a given job to finish before running
                the current one. Could be a  Tasker object or a slurm jobid
                (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header
                default options if needed (time, qos, etc...). Use a
                dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            no_dmat: Do not use distortion matrix.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_export.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
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
            "qos": "regular",
            "time": "00:10:00",
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "data": str(self.paths.cf_fname(region, region2, absorber, absorber2)),
            "out": str(self.paths.exp_cf_fname(region, region2, absorber, absorber2)),
        }
        if not no_dmat:
            args["dmat"] = str(
                self.paths.dmat_fname(region, region2, absorber, absorber2)
            )

        args = merge_dicts(args, updated_picca_extra_args)

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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        absorber2 = absorber if absorber2 is None else absorber2
        absorber2 = self.validate_absorber(absorber2)
        absorber = self.validate_absorber(absorber)

        command = "picca_metal_dmat.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
            region2=region2,
            absorber2=absorber2,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"metal_{region}_{region2}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "10:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "out": str(self.paths.metal_fname(region, region2, absorber, absorber2)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if absorber2 != absorber:
            args["lambda-abs2"]: absorber_igm[absorber2.lower()]

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        if region2 != region:
            args["in-dir2"] = str(self.paths.deltas_path(region2))

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.metal_fname(region, region2, absorber, absorber2).parent.mkdir(
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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_xcf.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xcf_{absorber}{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "02:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "drq": str(drq),
            "out": str(self.paths.xcf_fname(region, absorber)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.xcf_fname(region, absorber).parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["general"]["conda environment"],
            run_file=self.paths.correlations_path / f"scripts/run_{job_name}.sh",
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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to picca_deltas.py script.
                Use a dictionary with the format {'argument_name', 'argument_value'}.
                Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-quasar distortion matrix.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_xdmat.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xdmat_{absorber}{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "02:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "drq": str(drq),
            "out": str(self.paths.xdmat_fname(region, absorber)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.xdmat_fname(region, absorber).parent.mkdir(
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
            wait_for=wait_for,
        )

    def get_xcf_exp_tasker(
        self,
        region: str = "lya",
        system: str = None,
        debug: bool = False,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        slurm_header_extra_args: Dict = dict(),
        picca_extra_args: Dict = dict(),
        no_dmat: bool = False,
    ):
        """Method to get a Tasker object to run forest-quasar correlation export
        with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            no_dmat: Do not use disortion matrix

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_export.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xcf_exp_{absorber}{region}"

        slurm_header_args = {
            "qos": "regular",
            "time": "00:10:00",
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "data": str(self.paths.xcf_fname(region, absorber)),
            "out": str(self.paths.exp_xcf_fname(region, absorber)),
            "blind-corr-type": "qsoxlya",
        }
        if not no_dmat:
            args["dmat"] = str(self.paths.xdmat_fname(region, absorber))

        args = merge_dicts(args, updated_picca_extra_args)

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
        picca_extra_args: Dict = dict(),
    ):
        """Method to get a Tasker object to run forest-quasar metal distortion matrix
        measurements with picca.

        Args:
            region: Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_xdmat.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xdmat_{absorber}{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "02:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "drq": str(drq),
            "out": str(self.paths.xdmat_fname(region, absorber)),
            "mode": "desi_healpix",
            "nproc": 128,
            "rej": 0.99,
            "nside": 16,
            "rp-min": -300,
            "rp-max": 300,
            "rt-max": 200,
            "np": 150,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = merge_dicts(args, updated_picca_extra_args)

        self.paths.xdmat_fname(region, absorber).parent.mkdir(
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
        picca_extra_args: Dict = dict(),
        no_dmat: bool = False,
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.
            no_dmat: Do not use disortion matrix

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_export.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xcf_exp_{absorber}{region}"

        slurm_header_args = {
            "qos": "regular",
            "time": "00:10:00",
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        args = {
            "data": str(self.paths.xcf_fname(region, absorber)),
            "out": str(self.paths.exp_xcf_fname(region, absorber)),
            "blind-corr-type": "qsoxlya",
            "lambda-abs": absorber_igm[absorber.lower()],
        }
        if not no_dmat:
            args["dmat"] = str(self.paths.xdmat_fname(region, absorber))

        args = merge_dicts(args, updated_picca_extra_args)

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
        picca_extra_args: Dict = dict(),
    ):
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
            picca_extra_args : Send extra arguments to
                picca_deltas.py script. Use a dictionary with the format
                {'argument_name', 'argument_value'}. Use {'argument_name': ''}
                if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        region = self.validate_region(region)
        absorber = self.validate_absorber(absorber)

        command = "picca_metal_xdmat.py"

        updated_picca_extra_args = self.generate_picca_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            picca_args=picca_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            config=self.config["correlations"],
            default_config=self.defaults["correlations"],
            slurm_args=slurm_header_extra_args,
            command=command,
            region=region,
            absorber=absorber,
        )
        updated_system = self.generate_system_arg(system)

        job_name = f"xmetal_{absorber}{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {
                **dict(nspec=200000),
                **updated_picca_extra_args,
            }
        else:
            qos = "regular"
            time = "10:00:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.paths.correlations_path / f"logs/{job_name}-%j.out"),
            "error": str(self.paths.correlations_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = merge_dicts(
            slurm_header_args, updated_slurm_header_extra_args
        )

        drq = self.paths.catalog_tracer

        args = {
            "in-dir": str(self.paths.deltas_path(region)),
            "drq": str(drq),
            "out": str(self.paths.xmetal_fname(region, absorber)),
            "lambda-abs": absorber_igm[absorber.lower()],
        }

        if "v9." in self.config["data"]["release"]:
            args["mode"] = "desi_mocks"

        args = merge_dicts(args, updated_picca_extra_args)

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        self.paths.xmetal_fname(region, absorber).parent.mkdir(
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
            wait_for=wait_for,
        )

    def get_fit_tasker(
        self,
        auto_correlations: List[str] = [],
        cross_correlations: List[str] = [],
        system: str = None,
        wait_for: Union[Tasker, ChainedTasker, int, List[int]] = None,
        vega_extra_args: Dict = dict(),
        slurm_header_extra_args: Dict = dict(),
    ):
        """Method to get a Tasker object to run vega with correlation data.

        Args:
            auto_correlations: List of auto-correlations to include in the vega
                fits. The format of the strings should be 'lya-lya_lya-lya'.
                This is to allow splitting.
            cross_correlations: List of cross-correlations to include in the vega
                fits. The format of the strings should be 'lya-lya'.
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
        updated_system = self.generate_system_arg(system)

        ini_files = []

        for auto_correlation in auto_correlations:
            absorber, region, absorber2, region2 = auto_correlation.replace(
                "_", "-"
            ).split("-")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)
            region2 = self.validate_region(region2)
            absorber2 = self.validate_absorber(absorber2)

            vega_args = self.generate_picca_extra_args(
                config=self.config["fits"],
                default_config=self.defaults["fits"],
                picca_args=dict(),
                command="vega_auto.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
                region2=region2,
                absorber2=absorber2,
            )

            args = {
                "data": {
                    "filename": self.paths.exp_cf_fname(
                        region, region2, absorber, absorber2
                    ),
                },
                "metals": {
                    "filename": self.paths.metal_fname(
                        region, region2, absorber, absorber2
                    ),
                },
            }

            args = merge_dicts(args, vega_args)

            # parse config
            fit_config = configparser.ConfigParser()
            fit_config.read_dict(args)

            filename = self.paths.fit_auto_fname(region, region2, absorber, absorber2)
            with open(filename, "w") as file:
                fit_config.write(file)

            ini_files.append(str(filename))

        for cross_correlation in cross_correlations:
            absorber, region = cross_correlation.split("-")
            region = self.validate_region(region)
            absorber = self.validate_absorber(absorber)

            vega_args = self.generate_picca_extra_args(
                config=self.config["fits"],
                default_config=self.defaults["fits"],
                picca_args=dict(),
                command="vega_cross.py",  # The use of .py only for using same function
                region=region,
                absorber=absorber,
            )

            args = {
                "data": {
                    "filename": self.paths.exp_xcf_fname(region, absorber),
                },
                "metals": {
                    "filename": self.paths.xmetal_fname(region, absorber),
                },
            }

            args = merge_dicts(args, vega_args)

            # parse config
            fit_config = configparser.ConfigParser()
            fit_config.read_dict(args)

            filename = self.paths.fit_cross_fname(region, absorber)
            with open(filename, "w") as file:
                fit_config.write(file)

            ini_files.append(str(filename))

        # Now the main file
        vega_args = self.generate_picca_extra_args(
            config=self.config["fits"],
            picca_args=dict(),
            default_config=self.defaults["fits"],
            command="vega_main.py",  # The .py needed to make use of same function
        )

        args = {  # TODO: Figure out this.
            "fiducial": {
                "filename": "/global/homes/c/cgordon9/desi/vega/"
                "vega/models/PlanckDR16/PlanckDR16.fits"
            },
            "output": {
                "filename": self.paths.fit_out_fname(),
            },
        }

        args = merge_dicts(args, vega_args)

        # parse config
        fit_config = configparser.ConfigParser()
        fit_config.read_dict(args)

        filename = self.paths.fit_main_fname()
        with open(filename, "w") as file:
            fit_config.write(file)

        ini_files.append(str(filename))

        # Now slurm args
        command = "run_vega.py"
        updated_slurm_header_args = self.generate_slurm_header_extra_args(
            config=self.config["fits"],
            default_config=self.defaults["fits"],
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
    def healpix_data(self):
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
    def transmission_data(self):
        """Path: Location of transmission LyaCoLoRe mock data."""
        if (
            self.config["data"]["healpix data"] != ""
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
    def survey_path(self):
        """Path: Survey path following bookkeeper convention."""
        return (
            Path(self.config["data"]["early dir"])
            / self.config["data"]["release"]
            / self.config["data"]["survey"]
        )

    @property
    def run_path(self):
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
    def continuum_fitting_mask(self):
        """Path: file with masking used in continuum fitting."""
        return self.run_path / "configs" / "continuum_fitting_mask.txt"

    @property
    def correlations_path(self):
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
    def fits_path(self):
        """Give full path to the fits runs.

        Returns:
            Path
        """
        fit_name = self.config["fits"]["run name"]

        return self.correlations_path / "fits" / fit_name

    @property
    def delta_config_file(self):
        """Default path to the deltas config file inside bookkeeper.

        Returns
            Path
        """
        return self.run_path / "configs" / "bookkeeper_config.yaml"

    @property
    def correlation_config_file(self):
        """Default path to the correlation config file inside bookkeeper.

        Returns
            Path
        """
        return self.correlations_path / "configs" / "bookkeeper_config.yaml"

    @property
    def fit_config_file(self):
        """Default path to the fit config file inside bookkeeper

        Returns
            Path
        """
        return self.fits_path / "configs" / "bookkeeper_config.yaml"

    def get_catalog_from_field(self, field):
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
            if Path(self.config["delta extraction"].get("dla catalog", "")).is_file():
                catalog = Path(self.config["delta extraction"]["dla catalog"])
            else:
                field_value = self.config["delta extraction"]["dla"]
                catalog = get_dla_catalog(
                    self.config["data"]["release"],
                    self.config["data"]["survey"],
                    version=field_value,
                )
        elif field == "bal":
            if Path(self.config["delta extraction"].get("bal catalog", "")).is_file():
                catalog = Path(self.config["delta extraction"]["bal catalog"])
            else:
                catalog = self.get_catalog_from_field("catalog")
        elif field == "catalog_tracer":
            if Path(self.config["correlations"].get("catalog tracer", "")).is_file():
                catalog = Path(self.config["correlations"]["catalog tracer"])
            else:
                catalog = self.get_catalog_from_field("catalog")
        else:
            # Here is the normal catalog
            if Path(self.config["data"][field]).is_file():
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
                    bal=self.config.get("delta extraction", dict()).get("bal", "0")
                    != "0",
                )

        if catalog.is_file():
            return catalog
        else:
            raise FileNotFoundError("catalog not found in path: ", str(catalog))

    @property
    def catalog(self):
        """catalog to be used for deltas."""
        return self.get_catalog_from_field("catalog")

    @property
    def catalog_dla(self):
        """DLA catalog to be used."""
        return self.get_catalog_from_field("dla")

    @property
    def catalog_bal(self):
        """catalog to be used for BAL masking."""
        return self.get_catalog_from_field("bal")

    @property
    def catalog_tracer(self):
        """catalog to be used for cross-correlations with quasars"""
        return self.get_catalog_from_field("catalog_tracer")

    @property
    def catalog_name(self):
        """Returns catalog standardize name."""
        name = Path(self.config["data"]["catalog"]).name
        if Path(self.config["data"]["catalog"]).is_file():
            return self.get_fits_file_name(Path(self.config["data"]["catalog"]))
        else:
            return name

    @staticmethod
    def get_fits_file_name(file):
        name = Path(file).name
        if name[-8:] == ".fits.gz":
            return name[:-8]
        elif name[-5:] == ".fits":
            return name[:-5]
        else:
            raise ValueError("Unrecognized fits catalog filename", name)

    @property
    def continuum_tag(self):
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

        dla_field = self.config["delta extraction"]["dla"]
        if Path(dla_field).is_file():
            dla = self.get_fits_file_name(Path(dla_field))
        else:
            dla = dla_field
        return "{}_{}_{}.{}.{}_{}".format(prefix, calib, calib_region, dla, bal, suffix)

    @staticmethod
    def compare_config_files(
        file1: Union[str, Path],
        file2: Union[str, Path],
        section: str,
        ignore_fields: List[str] = [],
    ):
        """Compare two config files to determine if they are the same.

        Args:
            file1
            file2
            section: Section of the yaml file to compare.
            ignore_fields: Fields to ignore in the comparison
        """
        with open(file1, "r") as f:
            config1 = yaml.load(f, Loader=yaml.BaseLoader)

        with open(file2, "r") as f:
            config2 = yaml.load(f, Loader=yaml.BaseLoader)

        for field in ignore_fields:
            for config in config1[section], config2[section]:
                if config.get(field, "") != "":
                    config.pop(field)

        if config1.get(section, dict()) == config2.get(section, dict()):
            return True
        else:
            return False

    def check_delta_directories(self):
        """Method to create basic directories in run directory."""
        for folder in ("scripts", "correlations", "logs", "deltas", "configs"):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)

    def check_correlation_directories(self):
        """Method to create basic directories in correlations directory."""
        for folder in ("scripts", "correlations", "fits", "logs", "configs"):
            (self.correlations_path / folder).mkdir(exist_ok=True, parents=True)

    def check_fit_directories(self):
        """Method to create basic directories in fits directory."""
        for folder in ("scripts", "results", "logs", "configs"):
            (self.fits_path / folder).mkdir(exist_ok=True, parents=True)

    def deltas_path(self, region: str = None, calib_step: int = None):
        """Method to obtain the path to deltas output.

        Args:
            region: Region used (in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas directory.
        """
        if calib_step is not None:
            return self.run_path / "deltas" / f"calibration_{calib_step}" / "Delta"
        else:
            region = Bookkeeper.validate_region(region)
            return self.run_path / "deltas" / region / "Delta"

    def deltas_log_path(self, region: str, calib_step: int = None):
        """Method to get the path to deltas log.

        Args:
            region: Region used (in valid_regions).
            calib_step: Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas direct
        """
        deltas_path = self.deltas_path(region, calib_step)
        return deltas_path.parent / "Log"

    def delta_attributes_file(self, region: str, calib_step: int = None):
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
        region: str,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
    ):
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
        return (
            self.correlations_path
            / "correlations"
            / f"{absorber}{region}_{absorber2}{region2}"
            / f"cf.fits.gz"
        )

    def dmat_fname(
        self,
        region: str,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
    ):
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
        region2 = region if region2 is None else region2
        return (
            self.cf_fname(region, region2, absorber, absorber2).parent / f"dmat.fits.gz"
        )

    def metal_fname(
        self,
        region: str,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
    ):
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
        region2 = region if region2 is None else region2
        return (
            self.cf_fname(region, region2, absorber, absorber2).parent
            / f"metal.fits.gz"
        )

    def exp_cf_fname(
        self,
        region: str,
        region2: str = None,
        absorber: str = "lya",
        absorber2: str = None,
    ):
        """Method to get the path to a forest-forest correlation export file.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.cf_fname(region, region2, absorber, absorber2)
        return cor_file.parent / f"cf_exp.fits.gz"

    def xcf_fname(self, region: str, absorber: str):
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to correlation file.
        """
        return (
            self.correlations_path
            / "correlations"
            / f"lya{absorber}{region}_qso"
            / f"xcf.fits.gz"
        )

    def xdmat_fname(self, region: str, absorber: str):
        """Method to get the path to a distortion matrix file for forest-quasar
        correlations.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        return self.xcf_fname(region, absorber).parent / f"xdmat.fits.gz"

    def xmetal_fname(self, region: str, absorber: str):
        """Method to get the path to a metal distortion matrix file for forest-quasar
        correlations.

        Args:
            region (str): Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        return self.xcf_fname(region, absorber).parent / f"xmetal.fits.gz"

    def exp_xcf_fname(self, region: str, absorber: str):
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region: Region of the forest used.
            absorber: Absorber to use (lya)

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.xcf_fname(region, absorber)
        return cor_file.parent / f"xcf_exp.fits.gz"

    def fit_auto_fname(self, region: str, absorber: str, region2: str, absorber2: str):
        """Method to get te path to a given fit auto config file.

        Args:
            region: Region where the correlation is computed.
            region2: Second region used (if cross-correlation).
            absorber: First absorber
            absorber2: Second absorber

        Returns:
            Path: Path to fit config file.
        """
        return self.fits_path / "configs" / f"{absorber}{region}x{absorber}{region}.ini"

    def fit_cross_fname(self, region: str, absorber: str):
        """Method to get te path to a given fit cross config file.

        Args:
            region: Region where the correlation is computed.
            absorber: First absorber

        Returns:
            Path: Path to fit config file.
        """
        return self.fits_path / "configs" / f"qsox{absorber}{region}.ini"

    def fit_main_fname(self):
        """Method to get the path to the main fit config file.

        Returns:
            Path: Path to main fit config file.
        """
        return self.fits_path / "configs" / "main.ini"

    def fit_out_fname(self):
        """Method to get the path to the fit output file.

        Returns:
            Path: Path to fit output file.
        """
        return self.fits_path / "results" / "fit_output.fits"
