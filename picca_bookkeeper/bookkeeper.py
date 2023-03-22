from pathlib import Path
import configparser
import copy
import filecmp
import shutil
import os
import yaml

from picca_bookkeeper.tasker import get_Tasker, ChainedTasker

forest_regions = {
    "lya": {
        "lambda-rest-min": 1040.0,
        "lambda-rest-max": 1200.0,
    },
    "lyb": {
        "lambda-rest-min": 920.0,
        "lambda-rest-max": 1020.0,
    },
    "mgii": {
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
}


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
        basedir = Path("/global/cfs/cdirs/desi/users/cramirez/Continuum_fitting_Y1/catalogs/qso")

    if bal:
        catalog += "-bal"

    catalog = basedir / release / survey / catalog
    for suffix in (".fits", ".fits.gz", "-bal.fits", "-bal.fits.gz"):
        if catalog.with_name(catalog.name + suffix).is_file():
            return catalog.with_name(catalog.name + suffix)
    else:
        raise FileNotFoundError(
            f"Could not find a compatible catalog inside the bookkeeper. (Path: {catalog})"
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
        basedir = Path("/global/cfs/cdirs/desi/users/cramirez/Continuum_fitting_Y1/catalogs/dla")

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
        default_input (Path): Path to the default input directories which contain input picca files. Different inputs could be used for different picca steps, this is the default one.
        output (Path): Path to the output directories where picca outputs will be written.
    """

    def __init__(self, config_path, overwrite_config=False):
        """
        Args:
            config_path (Path or str): Path to configuration file or to an already created run path.
            overwrite_config (bool, optional): overwrite bookkeeper config without asking if it already exists inside bookkeeper.
        """
        config_path = Path(config_path)
        if not config_path.is_file():
            if (config_path / "configs/bookkeeper_config.yaml").is_file():
                config_path = config_path / "configs/bookkeeper_config.yaml"
            else:
                raise FileNotFoundError("Config file couldn't be found", config_path)
        if config_path.name.endswith(".ini"):  # pragma: no cover
            from warnings import warn

            warn(
                "Using .ini files is deprecated, .yaml file will be created and used instead"
            )
            _config = configparser.ConfigParser()
            _config.read(config_path)
            _conf_dict = {s: dict(_config.items(s)) for s in _config.sections()}

            new_conf = config_path.parent / (config_path.name[:-4] + ".yaml")
            with open(new_conf, "w") as file:
                yaml.dump(_conf_dict, file)
            config_path = new_conf

        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.BaseLoader)

        # Defaulting default input dir to the same place as default output dir
        # this is to make sure that further steps beyond deltas are also reading
        # data from the output folder by default.
        if (
            self.config["data"]["input_dir"] is None
            or self.config["data"]["input_dir"] == ""
        ):
            self.config["data"]["input_dir"] = self.config["data"]["output_dir"]

        self.default_input = self._get_pathbuilder("data", "input")
        self.output = self._get_pathbuilder("data", "output")

        self.output.check_directories()

        # Copy bookkeeper configuration into destination
        if not self.output.config_file.is_file():
            shutil.copyfile(config_path, self.output.config_file)
        elif filecmp.cmp(self.output.config_file, config_path):
            pass
        elif overwrite_config:
            shutil.copyfile(config_path, self.output.config_file)
        else:
            while True:
                answer = input(
                    "Config file inside bookkeeper differs, do you want to overwrite it? (yes/no)"
                )
                if answer == "yes":
                    shutil.copyfile(config_path, self.output.config_file)
                    break
                elif answer == "no":
                    print("Exiting")
                    exit()
                else:  # pragma: no cover
                    print("Please enter yes or no")

        # Copying quasar catalog into destination
        self.set_output_catalog("catalog")
        if self.config["data"]["catalog_tracer"] in (None, ""):
            self.output._catalog_tracer = self.output._catalog
        else:
            self.set_output_catalog("catalog_tracer")

        # Copying DLA/BAL catalogs if needed
        if self.config["continuum fitting"]["dla"] != "0":
            self.set_output_catalog("dla")

    def set_output_catalog(self, field):
        """Method to incorporate catalog into the output bookkeeper"""
        input_catalog = self.default_input.get_catalog_from_field(field)
        catname = self.output.catalog_name
        output_catalog = self.output.catalogs_dir / input_catalog.name

        if not output_catalog.is_file():
            output_catalog.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(input_catalog, output_catalog)
        elif not filecmp.cmp(output_catalog, input_catalog):
            raise FileExistsError(
                "Different catalog placed in the output bookkeeper directory",
                str(input_catalog),
                str(output_catalog),
            )

        if field == "catalog":
            self.output._catalog = output_catalog
        elif field == "catalog_tracer":
            self.output._catalog_tracer = output_catalog
        elif field == "dla":
            self.output._catalog_dla = output_catalog

    def _get_pathbuilder(self, section, input_output):
        """Method to get a PathBuilder object which can be used to define paths following the bookkeeper convention.

        Will read the path (early-dir) from the config file, given the input arguments, and generate a PathBuilder object from it.

        Args:
            section (str): Section in config file to read the path.
            input_output (str): Whether to read input or output path from configuration file.

        Returns:
            PathBuilder: PathBuilder object for the corresponding early-dir path read from the config file.
        """
        if input_output not in ("input", "output"):
            raise ValueError(f"input/output expected, got {input_output}")
        value = self.config[section][input_output + "_dir"]
        if value is None or value == "":
            # Defaulting to default input if not provided in the section.
            return PathBuilder(self.config, self.config["data"][input_output + "_dir"])
        else:
            return PathBuilder(self.config, value)

    @staticmethod
    def validate_region(region, calibration_fields=False):
        """Method to check if a region string is valid. Also converts it into lowercase.

        Will raise value error if the region is not in ['lya', 'lyb']. If calibration_fields is enabled, it will also accept ['civ', 'mgii'].

        Args:
            region (str): Region (should be in ['civ', 'mgii'] to pass the validation).
            calibration_fields (bool, optional): Whether to include calibration regions. (Default: False).
        """
        if region.lower() not in ["lya", "lyb"]:
            if not calibration_fields or region.lower() not in forest_regions:
                raise ValueError("Invalid region", region)

        return region.lower()

    def generate_slurm_header_extra_args(self, slurm_args, command, calib_step=None):
        copied_args = copy.deepcopy(slurm_args)
        config = copy.deepcopy(self.config)

        if calib_step is None:
            command_ = command.split(".py")[0]
        else:
            command_ = command.split(".py")[0] + f"_calib_{calib_step}"

        if "slurm args" in config.keys() and config["slurm args"] not in (
            "",
            None,
        ):
            if "general" in config["slurm args"] and config["slurm args"][
                "general"
            ] not in ("", None):
                general = config["slurm args"]["general"]
            else:
                general = dict()

            if command_ in config["slurm args"] and config["slurm args"][
                command_
            ] not in ("", None):
                return {**general, **config["slurm args"][command_], **copied_args}
            else:
                return {**general, **copied_args}
        else:
            return copied_args

    def generate_picca_extra_args(self, picca_args, command, calib_step=None):
        copied_args = copy.deepcopy(picca_args)
        config = copy.deepcopy(self.config)

        if calib_step is None:
            command_ = command.split(".py")[0]
        else:
            command_ = command.split(".py")[0] + f"_calib_{calib_step}"

        if (
            "picca args" in config.keys()
            and config["picca args"] not in ("", None)
            and command_ in config["picca args"]
            and config["picca args"][command_] not in ("", None)
        ):
            return {**config["picca args"][command_], **copied_args}
        else:
            return copied_args

    def generate_system_arg(self, system):
        if system is None:
            return copy.copy(self.config["system"].get("system", "slurm_cori"))
        else:
            return system

    def get_raw_deltas_tasker(
        self,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        region="lya",
        picca_extra_args=dict(),
    ):
        command = "picca_convert_transmission.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region, calibration_fields=True)

        job_name = f"raw_deltas_{region}"

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
            updated_picca_extra_args = {**dict(nspec=1000), **updated_picca_extra_args}
        else:
            qos = "regular"
            time = "03:00:00"

        input = self._get_pathbuilder("data", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "object-cat": str(self.output.catalog),
            "in-dir": str(input.transmission_data),
            "out-dir": str(self.output.get_deltas_path(region)),
        }

        self.output.get_deltas_path(region).mkdir(exist_ok=True, parents=True)

        args = {
            **args,
            **{
                "lambda-rest-min": forest_regions[region]["lambda-rest-min"],
                "lambda-rest-max": forest_regions[region]["lambda-rest-max"],
            },
        }

        args = {**args, **updated_picca_extra_args}

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_delta_extraction_tasker(
        self,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        region="lya",
        picca_extra_args=dict(),
        calib_step=None,
    ):
        """Method to get a Tasker object to run delta extraction with picca.

        Args:
            region (str, optional): Region where to compute deltas. Options: ('lya', 'lyb', 'mgii', 'civ'). Default: 'lya'
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Set extra options for picca delta extraction. The format should be a dict of dicts: wanting to change "num masks" in "masks" section one should pass {'num masks': {'masks': value}}.
            calib_step (int, optional): Calibration step. Default: None, no calibration

        Returns:
            Tasker: Tasker object to run delta extraction.
        """
        command = "picca_delta_extraction.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command, calib_step
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command, calib_step
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region, calibration_fields=True)

        job_name = f"delta_extraction_{region}"
        if calib_step is not None:
            job_name += "_calib_step_" + str(calib_step)
        config_file = self.output.run_path / f"configs/{job_name}.ini"

        input = self._get_pathbuilder("data", "input")

        # add masks section if necessary
        # only apply it in not calibration runs
        if "masks" not in updated_picca_extra_args or isinstance(
            updated_picca_extra_args["masks"], str
        ):
            updated_picca_extra_args["masks"] = dict()
        if updated_picca_extra_args.get("masks").get("num masks", 0) == 0:
            updated_picca_extra_args["masks"]["num masks"] = 0
        if (
            "mask file" in self.config["continuum fitting"]
            and self.config["continuum fitting"]["mask file"] != ""
        ):
            if not Path(self.config["continuum fitting"]["mask file"]).is_file():
                raise FileNotFoundError("Provided mask file does not exist.")
            else:
                if not self.output.continuum_fitting_mask.is_file():
                    shutil.copy(
                        self.config["continuum fitting"]["mask file"],
                        self.output.continuum_fitting_mask,
                    )
                elif not filecmp.cmp(
                    self.output.continuum_fitting_mask,
                    self.config["continuum fitting"]["mask file"],
                ):
                    raise FileExistsError(
                        "different mask file already stored in the bookkeeper",
                        self.output.continuum_fitting_mask,
                    )

            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            updated_picca_extra_args["masks"]["num masks"] = prev_mask_number + 1
            updated_picca_extra_args["masks"][f"type {prev_mask_number}"] = "LinesMask"
            updated_picca_extra_args[f"mask arguments {prev_mask_number}"] = dict(
                filename=self.output.continuum_fitting_mask,
            )

        # add corrections section if necessary
        if "corrections" not in updated_picca_extra_args or isinstance(
            updated_picca_extra_args["corrections"], str
        ):
            updated_picca_extra_args["corrections"] = dict()
        if updated_picca_extra_args.get("corrections").get("num corrections", 0) == 0:
            updated_picca_extra_args["corrections"]["num corrections"] = 0

        # update corrections section
        # here we are dealing with calibration runs
        # If there is no calibration, we should not have calib_steps
        if self.config["continuum fitting"]["calib"] == "0" and calib_step is not None:
            raise ValueError("Trying to run calibration with calib = 0 in config file.")
        if self.config["continuum fitting"]["calib"] != "4":
            if (
                "CalibrationCorrection"
                in updated_picca_extra_args["corrections"].values()
                or "IvarCorrection" in updated_picca_extra_args["corrections"].values()
            ):
                raise ValueError(
                    "Calibration corrections added by user with calib option != 3"
                )

        # Now we deal with dMdB20 option
        if self.config["continuum fitting"]["calib"] == "1":
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
                        self.output.get_delta_attributes_file(None, calib_step=1)
                    )
            # actual run using with both corrections
            else:
                if not self.output.get_deltas_path(calib_step=2).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration_tasker before running deltas."
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
                ] = str(self.output.get_delta_attributes_file(None, calib_step=1))

                updated_picca_extra_args["corrections"][
                    f"type {prev_n_corrections+1}"
                ] = "IvarCorrection"
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections+1}"
                ] = dict()
                updated_picca_extra_args[
                    f"correction arguments {prev_n_corrections+1}"
                ]["filename"] = str(
                    self.output.get_delta_attributes_file(None, calib_step=2)
                )
        elif self.config["continuum fitting"]["calib"] == "2":
            if "expected flux" not in updated_picca_extra_args or isinstance(
                updated_picca_extra_args["expected flux"], str
            ):
                updated_picca_extra_args["expected flux"] = dict()

            updated_picca_extra_args["expected flux"][
                "type"
            ] = "Dr16FixedFudgeExpectedFlux"
            updated_picca_extra_args["expected flux"]["fudge value"] = 0

            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.output.get_deltas_path(calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration tasker before running deltas."
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
                ] = str(self.output.get_delta_attributes_file(None, calib_step=1))
        elif self.config["continuum fitting"]["calib"] == "3":
            if "expected flux" not in updated_picca_extra_args or isinstance(
                updated_picca_extra_args["expected flux"], str
            ):
                updated_picca_extra_args["expected flux"] = dict()

            updated_picca_extra_args["expected flux"]["type"] = "Dr16ExpectedFlux"

            # No special action for calibration steps,
            # only add extra actions for main run
            if calib_step is None:
                if not self.output.get_deltas_path(calib_step=1).is_dir():
                    raise FileNotFoundError(
                        "Calibration folder does not exist. run get_calibration tasker before running deltas."
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
                ] = str(self.output.get_delta_attributes_file(None, calib_step=1))

        # update masks sections if necessary
        if (
            self.config["continuum fitting"]["dla"] != "0"
            or self.config["continuum fitting"]["bal"] != "0"
        ) and calib_step is None:
            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            if self.config["continuum fitting"]["dla"] != "0":
                if "DlaMask" in updated_picca_extra_args["masks"].values():
                    raise ValueError("DlaMask set by user with dla option != 0")

                updated_picca_extra_args["masks"][
                    f"type {prev_mask_number}"
                ] = "DlaMask"
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"] = dict()
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                    "filename"
                ] = self.output.catalog_dla
                updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                    "los_id name"
                ] = "TARGETID"
                updated_picca_extra_args["masks"]["num masks"] = prev_mask_number + 1

            prev_mask_number = int(updated_picca_extra_args["masks"]["num masks"])
            if self.config["continuum fitting"]["bal"] != "0":
                if self.config["continuum fitting"]["bal"] == "2":
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
                    ] = self.output.catalog
                    updated_picca_extra_args[f"mask arguments {prev_mask_number}"][
                        "los_id name"
                    ] = "TARGETID"
                    updated_picca_extra_args["masks"]["num masks"] = (
                        prev_mask_number + 1
                    )
                else:
                    raise ValueError(
                        "Invalid value for bal: ",
                        self.config["continuum fitting"]["bal"],
                    )

        # update expected flux section if necessary
        if self.config["continuum fitting"]["prefix"] not in [
            "dMdB20",
            "raw",
            "true",
        ]:
            raise ValueError(
                f"Unrecognized continuum fitting prefix: {self.config['continuum fitting']['prefix']}"
            )
        elif self.config["continuum fitting"]["prefix"] == "raw":
            raise ValueError(
                f"raw continuum fitting provided in config file, use get_raw_deltas_tasker instead"
            )
        elif self.config["continuum fitting"]["prefix"] == "true":
            if (
                "expected flux" not in updated_picca_extra_args
                or "raw statistics file"
                not in updated_picca_extra_args["expected flux"]
                or updated_picca_extra_args["expected flux"]["raw statistics file"]
                in ("", None)
            ):
                raise ValueError(
                    f"Should define expected flux and raw statistics file in picca args section in order to run TrueContinuum"
                )
            updated_picca_extra_args["expected flux"]["type"] = "TrueContinuum"
            if (
                updated_picca_extra_args.get("expected flux").get("input directory", 0)
                == 0
            ):
                updated_picca_extra_args["expected flux"][
                    "input directory"
                ] = input.healpix_data

        # add linear scheme by default
        if "data" not in updated_picca_extra_args or isinstance(
            updated_picca_extra_args["data"], str
        ):
            updated_picca_extra_args["data"] = dict()
        if updated_picca_extra_args.get("data").get("wave solution", 0) == 0:
            updated_picca_extra_args["data"]["wave solution"] = "lin"

        # create config for delta_extraction options
        self.output.get_deltas_path(region, calib_step).mkdir(
            parents=True, exist_ok=True
        )
        deltas_config_dict = {
            "general": {
                "overwrite": True,
                "out dir": str(
                    self.output.get_deltas_path(region, calib_step).parent.resolve()
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
            "catalogue", str(self.output.catalog)
        )
        deltas_config_dict["data"]["input directory"] = deltas_config_dict["data"].get(
            "input directory", str(input.healpix_data)
        )
        deltas_config_dict["data"]["lambda min rest frame"] = deltas_config_dict[
            "data"
        ].get("lambda min rest frame", forest_regions[region]["lambda-rest-min"])
        deltas_config_dict["data"]["lambda max rest frame"] = deltas_config_dict[
            "data"
        ].get("lambda max rest frame", forest_regions[region]["lambda-rest-max"])

        deltas_config_dict["data"]["lambda min"] = deltas_config_dict["data"].get(
            "lambda min", 3600
        )
        deltas_config_dict["data"]["lambda max"] = deltas_config_dict["data"].get(
            "lambda max", 5500
        )

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

        # update expected flux section with extra options
        if "type" not in deltas_config_dict["expected flux"].keys():
            deltas_config_dict.get("expected flux").update(
                {
                    "type": "Dr16ExpectedFlux",
                }
            )

        deltas_config_dict = {**deltas_config_dict, **updated_picca_extra_args}

        # parse config
        deltas_config = configparser.ConfigParser()
        deltas_config.read_dict(deltas_config_dict)

        with open(config_file, "w") as file:
            deltas_config.write(file)

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        return get_Tasker(
            updated_system,
            command=command,
            command_args={"": str(config_file.resolve())},
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_calibration_extraction_tasker(
        self,
        region=None,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run calibration with picca delta extraction method.

        Args:
            region (str, optional): Region where to compute deltas. Options: ('lya', 'lyb', 'mgii', 'ciii', 'civ'). Default: 'mgii'
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run delta extraction for calibration.
        """
        steps = []
        if region is None:
            cfg_val = self.config["continuum fitting"].get("calib region", "mgii")
            if cfg_val is None or cfg_val == "":
                region = "mgii"
            else:
                region = cfg_val

        if self.config["continuum fitting"]["calib"] not in [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]:
            raise ValueError(
                "Invalid calib value in config file. (Valid values are 0 1 2 3 4)"
            )
        elif self.config["continuum fitting"]["calib"] in ("1",):
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
        elif self.config["continuum fitting"]["calib"] in ("2", "3"):
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
        region="lya",
        region2=None,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-forest correlations with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations. Default: None, auto-correlation.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        command = "picca_cf.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        job_name = f"cf_{region}_{region2}"

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

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "out": str(self.output.get_cf_fname(region, region2)),
            "nproc": 32,
            "rp-min": 0,
            "rp-max": 300,
            "rt-max": 200,
            "np": 75,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        if region2 != region:
            args["in-dir2"] = str(input.get_deltas_path(region2))
        args = {**args, **updated_picca_extra_args}

        self.output.get_cf_fname(region, region2).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_xcf_tasker(
        self,
        region="lya",
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-quasar correlations with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-quasar correlation.
        """
        command = "picca_xcf.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region)

        job_name = f"xcf_{region}"

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

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        drq = self.output.catalog_tracer

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "drq": str(drq),
            "out": str(self.output.get_xcf_fname(region)),
            "mode": "desi_healpix",
            "nproc": 32,
            "nside": 16,
            "rp-min": -300,
            "rp-max": 300,
            "rt-max": 200,
            "np": 150,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        args = {**args, **updated_picca_extra_args}

        self.output.get_xcf_fname(region).parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_dmat_tasker(
        self,
        region="lya",
        region2=None,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-forest distortion matrix measurements with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations. Default: None, auto-correlation.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest distortion matrix.
        """
        command = "picca_dmat.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        job_name = f"dmat_{region}_{region2}"

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

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "out": str(self.output.get_dmat_fname(region, region2)),
            "nproc": 32,
            "rej": 0.99,
            "rp-min": 0,
            "rp-max": 300,
            "rt-max": 200,
            "np": 75,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        if region2 != region:
            args["in-dir2"] = str(input.get_deltas_path(region2))

        args = {**args, **updated_picca_extra_args}

        self.output.get_dmat_fname(region, region2).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_xdmat_tasker(
        self,
        region="lya",
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-quasar distortion matrix measurements with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-quasar distortion matrix.
        """
        command = "picca_xdmat.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region)

        job_name = f"xdmat_{region}"

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

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        drq = self.output.catalog_tracer

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "drq": str(drq),
            "out": str(self.output.get_xdmat_fname(region)),
            "mode": "desi_healpix",
            "nproc": 32,
            "rej": 0.99,
            "nside": 16,
            "rp-min": -300,
            "rp-max": 300,
            "rt-max": 200,
            "np": 150,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        args = {**args, **updated_picca_extra_args}

        self.output.get_xdmat_fname(region).parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_metal_tasker(
        self,
        region="lya",
        region2=None,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-forest metal distortion matrix measurements with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations. Default: None, auto-correlation.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-forest metal distortion matrix.
        """
        command = "picca_metal_dmat.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)
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
            time = "02:00:00"

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "out": str(self.output.get_metal_fname(region, region2)),
            "nproc": 32,
            "rej": 0.995,
            "abs-igm": "SiII(1260) SiIII(1207) SiII(1193) SiII(1190)",
            "rp-min": 0,
            "rp-max": 300,
            "rt-max": 200,
            "np": 75,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        if region2 != region:
            args["in-dir2"] = str(input.get_deltas_path(region2))

        args = {**args, **updated_picca_extra_args}

        self.output.get_metal_fname(region, region2).parent.mkdir(
            exist_ok=True, parents=True
        )

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_xmetal_tasker(
        self,
        region="lya",
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run forest-quasar metal distortion matrix measurements with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run forest-quasar metal distortion matrix.
        """
        command = "picca_metal_xdmat.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)
        region = self.validate_region(region)

        job_name = f"xmetal_{region}"

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

        input = self._get_pathbuilder("correlations", "input")

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        drq = self.output.catalog_tracer

        args = {
            "in-dir": str(input.get_deltas_path(region)),
            "drq": str(drq),
            "out": str(self.output.get_xmetal_fname(region)),
            "mode": "desi_healpix",
            "nproc": 32,
            "rej": 0.995,
            "abs-igm": "SiII(1260) SiIII(1207) SiII(1193) SiII(1190)",
            "rp-min": -300,
            "rp-max": 300,
            "rt-max": 200,
            "np": 150,
            "nt": 50,
            "fid-Or": 7.975e-5,
        }

        args = {**args, **updated_picca_extra_args}

        environmental_variables = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        self.output.get_xmetal_fname(region).parent.mkdir(exist_ok=True, parents=True)

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_cf_exp_tasker(
        self,
        region="lya",
        region2=None,
        system=None,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
        no_dmat=False,
    ):
        """Method to get a Tasker object to run forest-forest correlation export with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations. Default: None, auto-correlation.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.
            no_dmat (bool, optional): Do not use distortion matrix. Default: False (use it).

        Returns:
            Tasker: Tasker object to run forest-forest correlation export with picca.
        """
        command = "picca_export.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)
        region = self.validate_region(region)

        job_name = f"cf_exp_{region}_{region2}"

        input = self._get_pathbuilder("export", "input")

        slurm_header_args = {
            "qos": "regular",
            "time": "00:10:00",
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "data": str(input.get_cf_fname(region, region2)),
            "out": str(self.output.get_exp_cf_fname(region, region2)),
        }
        if not no_dmat:
            args["dmat"] = str(input.get_dmat_fname(region, region2))

        args = {**args, **updated_picca_extra_args}

        if (
            input.config["data"]["survey"] == "main"
            or input.config["data"]["survey"] == "all"
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
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_xcf_exp_tasker(
        self,
        region="lya",
        system=None,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
        no_dmat=False,
    ):
        """Method to get a Tasker object to run forest-quasar correlation export with picca.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.
            no_dmat (bool, optional): Do not use distortion matrix. Default: False (use it).

        Returns:
            Tasker: Tasker object to run forest-quasar correlation export with picca.
        """
        command = "picca_export.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)
        region = self.validate_region(region)

        job_name = f"xcf_exp_{region}"

        input = self._get_pathbuilder("export", "input")

        slurm_header_args = {
            "qos": "regular",
            "time": "00:05:00",
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = {
            "data": str(input.get_xcf_fname(region)),
            "out": str(self.output.get_exp_xcf_fname(region)),
            "blind-corr-type": "qsoxlya",
        }
        if not no_dmat:
            args["dmat"] = str(input.get_xdmat_fname(region))

        args = {**args, **updated_picca_extra_args}

        if (
            input.config["data"]["survey"] == "main"
            or input.config["data"]["survey"] == "all"
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
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_cf_fit_tasker(
        self,
        bao_mode,
        region="lya",
        region2=None,
        system=None,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):  # pragma: no cover
        """Method to get a Tasker object to run fitter2 on auto-correlation measurements.

        Args:
            bao_mode (str): Bao mode to use. Options ('free', 'fixed').
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations on forest-forest correlations. Default: None, auto-correlation.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run fitter2 on auto-correlation measurements.
        """
        command = "picca_fitter2.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region)
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)

        job_name = f"fit_{bao_mode}_lya{region}_lya{region2}"

        # copy chi2 config
        chi2_fname = self.output.get_fit_chi2_cf_fname(region, region2)
        chi2_fname.parent.mkdir(parents=True, exist_ok=True)

        input_config_dir = Path(self.config["fitter"]["fit_config_dir"])
        os.environ["FIT_CONFIG_DIR"] = str(input_config_dir.resolve())
        shutil.copy(input_config_dir / "auto" / f"chi2_{bao_mode}_BAO.ini", chi2_fname)

        # Change paths inside chi2 file
        chi2_config = configparser.ConfigParser()
        chi2_config.read(chi2_fname)

        ini_files_source = chi2_config["data sets"]["ini files"].split(" ")
        if len(ini_files_source) != 1:
            raise ValueError(
                "Length of configuration files should be one for auto-correlations"
            )
        ini_file_source = Path(os.path.expandvars(ini_files_source[0]))

        if not ini_file_source.is_file():
            raise FileNotFoundError(
                "Configuration file does not exist:", str(ini_file_source)
            )

        ini_config = configparser.ConfigParser()
        ini_config.optionxform = str
        ini_config.read(ini_file_source)

        if ini_config["data"]["name"] != "cf":
            raise ValueError(
                "Invalid name value in config file", ini_config["data"]["name"]
            )
        ini_config["data"]["filename"] = str(
            self.output.get_exp_cf_fname(region, region2)
        )
        ini_config["metals"]["filename"] = str(
            self.output.get_metal_fname(region, region2)
        )

        ini_files = [self.output.get_fit_config_cf_fname(chi2_fname, region, region2)]
        with open(ini_files[0], "w") as f:
            ini_config.write(f)

        chi2_config["data sets"]["ini files"] = " ".join(map(str, ini_files))
        chi2_config["output"]["filename"] = str(
            self.output.get_fit_results_cf_fname(region, region2=region2)
        )

        with open(chi2_fname, "w") as f:
            chi2_config.write(f)

        command = f"picca_fitter2.py {str(chi2_fname)}"

        input = self._get_pathbuilder("fitter", "input")

        slurm_header_args = {
            "qos": "regular",
            "time": "00:30:00",
            "job-name": f"fit_{job_name}",
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        environmental_variables = {
            "ALL": "",
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        return get_Tasker(
            updated_system,
            command=command,
            command_args=updated_picca_extra_args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_xcf_fit_tasker(
        self,
        bao_mode,
        region="lya",
        system=None,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):  # pragma: no cover
        """Method to get a Tasker object to run fitter2 on cross-correlation measurements.

        Args:
            bao_mode (str): Bao mode to use. Options ('free', 'fixed').
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run fitter2 on cross-correlation measurements.
        """
        command = "picca_fitter2.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region)

        job_name = f"fit_{bao_mode}_qsoxlya{region}"

        # copy chi2 config
        chi2_fname = self.output.get_fit_chi2_xcf_fname(region)
        chi2_fname.parent.mkdir(parents=True, exist_ok=True)

        input_config_dir = Path(self.config["fitter"]["fit_config_dir"])
        os.environ["FIT_CONFIG_DIR"] = str(input_config_dir.resolve())
        shutil.copy(input_config_dir / "cross" / f"chi2_{bao_mode}_BAO.ini", chi2_fname)

        # Change paths inside chi2 file
        chi2_config = configparser.ConfigParser()
        chi2_config.read(chi2_fname)

        ini_files_source = chi2_config["data sets"]["ini files"].split(" ")
        if len(ini_files_source) != 1:
            raise ValueError(
                "Length of configuration files should be one for cross-correlations"
            )
        ini_file_source = Path(os.path.expandvars(ini_files_source[0]))

        if not ini_file_source.is_file():
            raise FileNotFoundError(
                "Configuration file does not exist:", str(ini_file_source)
            )

        ini_config = configparser.ConfigParser()
        ini_config.optionxform = str
        ini_config.read(ini_file_source)

        if ini_config["data"]["name"] != "xcf":
            raise ValueError(
                "Invalid name value in config file", ini_config["data"]["name"]
            )
        ini_config["data"]["filename"] = str(self.output.get_exp_xcf_fname(region))
        ini_config["metals"]["filename"] = str(self.output.get_xmetal_fname(region))

        ini_files = [self.output.get_fit_config_xcf_fname(chi2_fname, region)]

        with open(ini_files[0], "w") as f:
            ini_config.write(f)

        chi2_config["data sets"]["ini files"] = " ".join(map(str, ini_files))
        chi2_config["output"]["filename"] = str(
            self.output.get_fit_results_xcf_fname(region)
        )

        with open(chi2_fname, "w") as f:
            chi2_config.write(f)

        command = f"picca_fitter2.py {str(chi2_fname)}"

        input = self._get_pathbuilder("fitter", "input")

        slurm_header_args = {
            "qos": "regular",
            "time": "00:30:00",
            "job-name": f"fit_{job_name}",
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        environmental_variables = {
            "ALL": "",
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        return get_Tasker(
            updated_system,
            command=command,
            command_args=updated_picca_extra_args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_combined_fit_tasker(
        self,
        bao_mode,
        region="lya",
        region2=None,
        cross_region=None,
        system=None,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):  # pragma: no cover
        """Method to get a Tasker object to run fitter2 combined fit.

        Args:
            corr (str): Correlation type to use. Options (forest-forest->'auto', forest-quasar->'cross', both->'combined').
            bao_mode (str): Bao mode to use. Options ('free', 'fixed').
            region (str, optional): Region to use. Options: ('lya', 'lyb'). Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations on forest-forest correlations. Default: None, auto-correlation.
            region_for_combined-cross (str, optional): Region to use for forest-quasar correlation when using combined fit. Default: None, use same as region.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to picca_deltas.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.

        Returns:
            Tasker: Tasker object to run fitter2 combined.
        """
        command = "picca_fitter2.py"
        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        region = self.validate_region(region)
        region2 = region if region2 is None else region2
        region2 = self.validate_region(region2)

        cross_region = region if cross_region is None else cross_region
        cross_region = self.validate_region(cross_region)

        job_name = f"fit_{bao_mode}_lya{region}_lya{region2}_qsoxlya{cross_region}"

        # copy chi2 config
        chi2_fname = self.output.get_fit_chi2_combined_fname(
            region, region2, cross_region
        )
        chi2_fname.parent.mkdir(parents=True, exist_ok=True)

        input_config_dir = Path(self.config["fitter"]["fit_config_dir"])
        os.environ["FIT_CONFIG_DIR"] = str(input_config_dir.resolve())
        shutil.copy(
            input_config_dir / "combined" / f"chi2_{bao_mode}_BAO.ini", chi2_fname
        )

        # Change paths inside chi2 file
        chi2_config = configparser.ConfigParser()
        chi2_config.read(chi2_fname)

        ini_files_source = chi2_config["data sets"]["ini files"].split(" ")
        if len(ini_files_source) != 2:
            raise ValueError(
                "Length of configuration files should be two for combined fits"
            )

        ini_files = []
        for ini_file_source in ini_files_source:
            ini_file_source = Path(os.path.expandvars(ini_file_source))
            if not ini_file_source.is_file():
                raise FileNotFoundError(
                    "Configuration file does not exist:", str(ini_file_source)
                )

            ini_config = configparser.ConfigParser()
            ini_config.optionxform = str
            ini_config.read(ini_file_source)

            if ini_config["data"]["name"] == "xcf":
                ini_config["data"]["filename"] = str(
                    self.output.get_exp_xcf_fname(cross_region)
                )
                ini_config["metals"]["filename"] = str(
                    self.output.get_xmetal_fname(cross_region)
                )
                fname = self.output.get_fit_config_xcf_fname(chi2_fname, cross_region)
            elif ini_config["data"]["name"] == "cf":
                ini_config["data"]["filename"] = str(
                    self.output.get_exp_cf_fname(region, region2)
                )
                ini_config["metals"]["filename"] = str(
                    self.output.get_metal_fname(region, region2)
                )
                fname = self.output.get_fit_config_cf_fname(chi2_fname, region, region2)
            else:
                raise ValueError(
                    "Invalid name value in config file", ini_config["data"]["name"]
                )

            with open(fname, "w") as f:
                ini_config.write(f)

            ini_files.append(fname)

        chi2_config["data sets"]["ini files"] = " ".join(map(str, ini_files))
        chi2_config["output"]["filename"] = str(
            self.output.get_fit_results_combined_fname(region, region2=region2)
        )

        with open(chi2_fname, "w") as f:
            chi2_config.write(f)

        command = f"picca_fitter2.py {str(chi2_fname)}"

        input = self._get_pathbuilder("fitter", "input")

        slurm_header_args = {
            "qos": "regular",
            "time": "00:30:00",
            "job-name": f"fit_{job_name}",
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        environmental_variables = {
            "ALL": "",
            "HDF5_USE_FILE_LOCKING": "FALSE",
        }

        return get_Tasker(
            updated_system,
            command=command,
            command_args=updated_picca_extra_args,
            slurm_header_args=slurm_header_args,
            srun_options=dict(),
            environment=self.config["system"]["python_environment"],
            environmental_variables=environmental_variables,
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )

    def get_convert_deltas_tasker(
        self,
        region="lya",
        calib_step=None,
        add_flux_properties=False,
        system=None,
        debug=False,
        wait_for=None,
        slurm_header_extra_args=dict(),
        picca_extra_args=dict(),
    ):
        """Method to get a Tasker object to run build_new_deltas_format in order to generate new picca deltas format.

        Args:
            region (str, optional): Region to use. Options: ('lya', 'lyb'). If calibration, this field is not used. Default: 'lya'.
            region2 (str, optional): Region to use for cross-correlations. Default: None, auto-correlation.
            add_flux_properties (bool, optional): Add flux properties to final deltas. Default: False.
            system (str, optional): Shell to use for job. 'slurm_cori' to use slurm scripts on cori, 'slurm_perlmutter' to use slurm scripts on perlmutter, 'bash' to run it in login nodes or computer shell. Default: None, read from config file.
            debug (bool, optional): Whether to use debug options.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            slurm_header_extra_args (dict, optional): Change slurm header default options if needed (time, qos, etc...). Use a dictionary with the format {'option_name': 'option_value'}.
            picca_extra_args (dict, optional): Send extra arguments to build_new_deltas_format.py script. Use a dictionary with the format {'argument_name', 'argument_value'}. Use {'argument_name': ''} if a action-like option is used.
            calib_step (int, optional): Calibration step. Default: None, no calibration

        Returns:
            Tasker: Tasker object to run forest-forest correlation.
        """
        command = "desi_bookkeeper_convert_deltas"

        updated_picca_extra_args = self.generate_picca_extra_args(
            picca_extra_args, command
        )
        updated_slurm_header_extra_args = self.generate_slurm_header_extra_args(
            slurm_header_extra_args, command
        )
        updated_system = self.generate_system_arg(system)

        if calib_step is None:
            job_name = f"convert_deltas_{region}"
        else:
            job_name = "convert_deltas_calib_step_" + str(calib_step)

        if debug:  # pragma: no cover
            qos = "debug"
            time = "00:30:00"
        else:
            qos = "regular"
            time = "00:30:00"

        slurm_header_args = {
            "time": time,
            "qos": qos,
            "job-name": job_name,
            "output": str(self.output.run_path / f"logs/{job_name}-%j.out"),
            "error": str(self.output.run_path / f"logs/{job_name}-%j.err"),
        }

        slurm_header_args = {**slurm_header_args, **updated_slurm_header_extra_args}

        args = dict()
        args["in-dir"] = self.output.get_deltas_path(
            region=region,
            calib_step=calib_step,
        )

        args["out-dir"] = args["in-dir"].parent / "Delta_new_format"
        args["config-file"] = args["in-dir"].parent / ".config.ini"

        if add_flux_properties:
            if calib_step is not None:
                args["config-for-flux"] = next(
                    (self.output.run_path / "configs").glob(
                        f"delta_extraction*calib_step_{calib_step}.ini"
                    )
                )
            else:
                args["config-for-flux"] = (
                    self.output.run_path / "configs" / f"delta_extraction_{region}.ini"
                )

        args = {**args, **updated_picca_extra_args}

        if "nproc" in args.keys():
            srun_options = {"cpus-per-task": args["nproc"]}
        else:
            srun_options = dict()

        return get_Tasker(
            updated_system,
            command=command,
            command_args=args,
            slurm_header_args=slurm_header_args,
            srun_options=srun_options,
            environment=self.config["system"]["python_environment"],
            run_file=self.output.run_path / f"scripts/run_{job_name}.sh",
            wait_for=wait_for,
        )


class PathBuilder:
    """Class to define paths following the bookkeeper convention.

    Attributes:
        config (configparser.ConfigParser): Configuration used to build paths.
        early_dir (Path)
        healpix_data (Path): Location of healpix data.
    """

    def __init__(self, config, early_dir):
        """
        Args:
            config (configparser.ConfigParser): Configuration to be used.
            early_dir (Path or str)
        """
        self.config = config
        self.early_dir = Path(early_dir)

    @property
    def healpix_data(self):
        """Path: Location of healpix data."""
        if self.config["data"]["healpix data"] in (None, ""):
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
                            f"/global/cfs/cdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.{version}/"
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
            self.early_dir
            / self.config["data"]["release"]
            / self.config["data"]["survey"]
        )

    @property
    def catalog(self):
        """Path: catalog to be used based on config file."""
        return self._catalog

    @property
    def catalog_tracer(self):
        """Path: catalog to be used based on config file (for tracer computations)."""
        return self._catalog_tracer

    @property
    def catalog_dla(self):
        """Path: catalog to be used for DLA masking."""
        return self._catalog_dla

    @property
    def continuum_fitting_mask(self):
        """Path: file with masking used in continuum fitting."""
        return self.run_path / "configs" / "continuum_fitting_mask.txt"

    def get_catalog_from_field(self, field):
        """Method to obtain catalogs given a catalog name in config file.

        It will check if the file exists, expand the full path if only the filename
        is given and raise a ValueError if the file does not exist.

        Args:
            field (str): whether to use catalog, catalog tracer fields or dla fields. (Options: ["catalog", "catalog_tracer", "dla"])

        Returns:
            Path: catalog to be used.
        """
        if field == "dla":
            field_value = self.config["continuum fitting"]["dla"]
            if Path(field_value).is_file():
                catalog = Path(field_value)
            else:
                catalog = get_dla_catalog(
                    self.config["data"]["release"],
                    self.config["data"]["survey"],
                    version=field_value,
                )
        else:
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
                    bal=self.config["continuum fitting"]["bal"] != "0",
                )

        if catalog.is_file():
            return catalog
        else:
            raise FileNotFoundError("catalog not found in path: ", str(catalog))

    @property
    def catalog_name(self):
        """str: get catalogue standardized name."""
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
        prefix = self.config["continuum fitting"]["prefix"]
        calib = self.config["continuum fitting"]["calib"]
        suffix = self.config["continuum fitting"]["suffix"]
        bal = self.config["continuum fitting"]["bal"]

        dla_field = self.config["continuum fitting"]["dla"]
        if Path(dla_field).is_file():
            dla = self.get_fits_file_name(Path(dla_field))
        else:
            dla = dla_field
        return "{}_{}.{}.{}_{}".format(prefix, calib, dla, bal, suffix)

    @property
    def run_path(self):
        """Path: full path to the picca run."""
        return self.survey_path / self.catalog_name / self.continuum_tag

    @property
    def config_file(self):
        """Path: Default path to config file inside bookkeeper."""
        return self.run_path / "configs" / "bookkeeper_config.yaml"

    @property
    def catalogs_dir(self):
        """Method to get directory where catalogs are stored."""
        return self.survey_path / self.catalog_name / "catalogs"

    def get_deltas_path(self, region=None, calib_step=None):
        """Method to obtain the path to deltas output.

        Args:
            region (str): Region used (lya, lyb).
            calib_step (int, optional): Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas directory.
        """
        if calib_step is not None:
            return self.run_path / "deltas" / f"calibration_{calib_step}" / "Delta"
        else:
            region = Bookkeeper.validate_region(region)
            return self.run_path / "deltas" / region / "Delta"

    def get_deltas_log_path(self, region, calib_step=None):
        """Method to get the path to deltas log.

        Args:
            region (str): Region used (lya, lyb).
            calib_step (int, optional): Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to deltas direct
        """
        deltas_path = self.get_deltas_path(region, calib_step)
        return deltas_path.parent / "Log"

    def get_delta_attributes_file(self, region, calib_step=None):
        """Method to get the path to deltas attributes file.

        Args:
            region (str): Region used (lya, lyb).
            calib_step (int, optional): Calibration step of the run (1 or 2 for usual runs).

        Returns:
            Path: Path to delta attributes file
        """
        return (
            self.get_deltas_log_path(region=region, calib_step=calib_step)
            / "delta_attributes.fits.gz"
        )

    def get_cf_fname(self, region, region2=None):
        """Method to get the path to a forest-forest correlation file.

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to correlation file.
        """
        region2 = region if region2 is None else region2
        return (
            self.run_path / "correlations" / f"lya{region}_lya{region2}" / f"cf.fits.gz"
        )

    def get_exp_cf_fname(self, region, region2=None):
        """Method to get the path to a forest-forest correlation export file.

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.get_cf_fname(region, region2)
        return cor_file.parent / f"cf_exp.fits.gz"

    def get_fit_chi2_cf_fname(self, region, region2=None):  # pragma: no cover
        """Method to get the path to a forest-forest correlation chi2 config file

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to chi2 fit config correlation file.
        """
        region2 = region if region2 is None else region2
        return self.run_path / "fits" / f"lya{region}_lya{region2}" / f"chi2_cf.ini"

    def get_fit_config_cf_fname(
        self, chi2_file, region, region2=None
    ):  # pragma: no cover
        """Method to get the path to a forest-forest correlation config file

        Args:
            chi2_file (Path): Path to the chi2 file that calls this file.
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to fit config correlation file.
        """
        region2 = region if region2 is None else region2
        return chi2_file.parent / f"config_cf.ini"

    def get_fit_results_cf_fname(self, region, region2=None):  # pragma: no cover
        chi2_file = self.get_fit_chi2_cf_fname(region, region2)
        return chi2_file.parent / f"results_cf.ini"

    def get_xcf_fname(self, region):
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region (str): Region of the forest used.

        Returns:
            Path: Path to correlation file.
        """
        return self.run_path / "correlations" / f"lya{region}_qso" / f"xcf.fits.gz"

    def get_exp_xcf_fname(self, region):
        """Method to get the path to a forest-quasar correlation export file.

        Args:
            region (str): Region of the forest used.

        Returns:
            Path: Path to export correlation file.
        """
        cor_file = self.get_xcf_fname(region)
        return cor_file.parent / f"xcf_exp.fits.gz"

    def get_fit_chi2_xcf_fname(self, region):  # pragma: no cover
        """Method to get the path to a quasar-forest correlation chi2 config file

        Args:
            region (str): Region where the correlation is computed.

        Returns:
            Path: Path to chi2 fit config correlation file.
        """
        return self.run_path / "fits" / f"lya{region}_qso" / f"chi2_xcf.ini"

    def get_fit_config_xcf_fname(
        self, chi2_file, region, region2=None
    ):  # pragma: no cover
        """Method to get the path to a quasar-forest correlation config file

        Args:
            chi2_file (Path): Path to the chi2 file that calls this file.
            region (str): Region where the correlation is computed.

        Returns:
            Path: Path to fit config correlation file.
        """
        return chi2_file.parent / f"config_xcf_{region}.ini"

    def get_fit_results_xcf_fname(self, region, region2=None):  # pragma: no cover
        chi2_file = self.get_fit_chi2_xcf_fname(region)
        return chi2_file.parent / f"results_xcf_{region}.ini"

    def get_dmat_fname(self, region, region2=None):
        """Method to get the path to a distortion matrix file for forest-forest correlations.

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to distortion matrix output.
        """
        region2 = region if region2 is None else region2
        return (
            self.run_path
            / "correlations"
            / f"lya{region}_lya{region2}"
            / f"dmat.fits.gz"
        )

    def get_xdmat_fname(self, region):
        """Method to get the path to a distortion matrix file for forest-quasar correlations.

        Args:
            region (str): Region of the forest used.

        Returns:
            Path: Path to export correlation file.
        """
        return self.run_path / "correlations" / f"lya{region}_qso" / f"xdmat.fits.gz"

    def get_metal_fname(self, region, region2=None):
        """Method to get the path to a metal distortion matrix file for forest-forest correlations.

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to distortion matrix output.
        """
        region2 = region if region2 is None else region2
        return (
            self.run_path
            / "correlations"
            / f"lya{region}_lya{region2}"
            / f"metal.fits.gz"
        )

    def get_xmetal_fname(self, region):
        """Method to get the path to a metal distortion matrix file for forest-quasar correlations.

        Args:
            region (str): Region of the forest used.

        Returns:
            Path: Path to export correlation file.
        """
        return self.run_path / "correlations" / f"lya{region}_qso" / f"xmetal.fits.gz"

    def get_fit_chi2_combined_fname(
        self, region, region2=None, cross_region=None
    ):  # pragma: no cover
        """Method to get the path to a forest-forest correlation chi2 config file

        Args:
            region (str): Region where the correlation is computed.
            region2 (str, optional): Second region used (if cross-correlation).

        Returns:
            Path: Path to chi2 fit config correlation file.
        """
        region2 = region if region2 is None else region2
        cross_region = region if cross_region is None else cross_region
        return (
            self.run_path
            / "fits"
            / f"lya{region}_lya{region2}_lya{cross_region}_qso"
            / f"chi2_cf_xcf.ini"
        )

    def get_fit_results_combined_fname(
        self, region, region2=None, cross_region=None
    ):  # pragma: no cover
        chi2_file = self.get_fit_chi2_combined_fname(region, region2, cross_region)
        return chi2_file.parent / f"results_cross_{region}_{region2}.ini"

    def check_directories(self):
        """Method to create basic directories in run directory."""
        for folder in ("scripts", "correlations", "logs", "fits", "deltas", "configs"):
            (self.run_path / folder).mkdir(exist_ok=True, parents=True)
