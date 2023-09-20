from __future__ import annotations

import logging
import os
import sys
import textwrap

from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Self, Type

logger = logging.getLogger(__name__)


def get_Tasker(system: str) -> Type[Tasker]:
    """Function to get a Tasker object for a given system.

    Args:
        system : Shell the Tasker will use. 'slurm_cori' to use
            slurm scripts on cori, slurm_perlmutter' to use slurm scripts
            on perlmutter, 'bash' to run it in login nodes or computer shell.
    """
    if system == "slurm":
        from warnings import warn

        warn(
            "slurm system option is deprecated,"
            "use slurm_cori or slurm_perlmutter instead",
            DeprecationWarning,
        )
        return SlurmCoriTasker
    elif system == "slurm_cori":
        return SlurmCoriTasker
    elif system == "slurm_perlmutter":
        return SlurmPerlmutterTasker
    elif system == "bash":
        return BashTasker
    else:
        raise ValueError(
            f"System not valid: {system} "
            "allowed options are slurm_cori, slurm_perlmutter, "
            "bash."
        )


class Tasker:
    """Object to write and run jobs.

    Attributes:
        slurm_header_args (dict): Header options to write if slurm tasker is selected.
            Use a dictionary with the format {'option_name': 'option_value'}
        command (str): Command to be run in the job.
        command_args (str): Arguments to the command.
        environment (str): Conda/python environment to load before running the command.
        environmental_variables (dict): Environmental variables to set before running the job.
            Format: {'environmental_variable': 'value'}.
        srun_options (str): If slurm tasker selected. Options for the srun command.
        run_file (Path): Location of the job file.
        wait_for (Tasker or str): If slurm tasker selected. Tasker to wait for before
            running (if Tasker) or jobid of the task to wait (if str).
        in_files: Input files that must exists or contain a jobid in order for the
            job to be launched.
        out_file: Out file that will be write by the job (to add jobid if available).
    """

    default_srun_options: Dict = dict()
    default_header: Dict = dict()

    def __init__(
        self,
        command: str,
        command_args: Dict,
        slurm_header_args: Dict,
        environment: str,
        run_file: Path | str,
        jobid_log_file: Path | str,
        wait_for: Optional[
            ChainedTasker | Tasker | List[Type[Tasker]] | int | List[int]
        ] = None,
        environmental_variables: Dict = dict(),
        srun_options: Dict = dict(),
        in_files: List[Path] | List[str] = list[Path](),
        out_file: Optional[Path | str] = None,
    ):
        """
        Args:
            command (str): Command to be run in the job.
            command_args (str): Arguments to the command.
            slurm_header_args (dict): Header options to write if slurm tasker is selected. Use a dictionary with the format {'option_name': 'option_value'}.
            srun_options (str): If slurm tasker selected. Options for the srun command.
            environment (str): Conda/python environment to load before running the command.
            run_file (Path): Location of the job file.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            environmental_variables (dict, optional): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}. Default: No environmental variables defined.
            jobid_log_file (Path): Location of log file where to include jobids of runs.
            in_files: Input files that must exists or contain a jobid in order for the
            job to be launched.
            out_file: Out file that will be write by the job (to add jobid if available).
        """
        self.slurm_header_args = {**self.default_header, **slurm_header_args}

        if isinstance(self.slurm_header_args.get("time", ""), int):
            print(self.slurm_header_args["time"])
            logger.warning(
                "Detected int value in field time inside slurm args. "
                "Be sure to quote time values in config file if the "
                "format is hours:minutes:seconds. "
                "If you sent an int as minutes, ignore this warning."
            )
        self.command = command
        self.command_args = command_args
        self.environment = environment
        self.environmental_variables = environmental_variables
        self.srun_options = {**self.default_srun_options, **srun_options}
        self.run_file = Path(run_file)
        self.wait_for = wait_for
        self.jobid_log_file = jobid_log_file
        self.in_files = in_files
        self.out_file = out_file

        self.jobid: Optional[int] = None

    def get_wait_for_ids(self) -> None:
        """Method to standardise wait_for Taskers or ids, in such a way that can be easily used afterwards."""
        self.wait_for = list(np.array(self.wait_for).reshape(-1))
        self.wait_for_ids = []

        for x in self.wait_for:
            if isinstance(x, (int, np.integer)):
                self.wait_for_ids.append(x)
            elif isinstance(x, Tasker):
                try:
                    self.wait_for_ids.append(x.jobid)
                except AttributeError as e:
                    raise Exception(
                        f"jobid not defined for some of the wait_for. Have you send "
                        "any script with each of the wait_for objects?"
                    ).with_traceback(e.__traceback__)
            elif isinstance(x, type(None)):
                pass
            else:
                raise ValueError("Unrecognized wait_for object: ", x)

        for file in self.in_files:
            file = Path(file)
            if not file.is_file():
                raise FileNotFoundError("Input file for run not found", str(file))

            size = file.stat().st_size
            if size > 1 and size < 40:
                self.wait_for_ids.append(int(Path(file).read_text().splitlines()[0]))

    def _make_command(self) -> str:
        """Method to generate a command with args

        Returns:
            str: Command to write in job run.
        """
        args = " ".join(
            [
                f"--{key} {value}" if key != "" else str(value)
                for key, value in self.command_args.items()
            ]
        )
        return f'command="{self.command} {args}"'

    def _make_body(self) -> str:
        """Method to generate the body of the job script.

        Return:
            str: Body of the job script."""
        header = self._make_header()
        env_opts = self._make_env_opts()
        command = self._make_command()
        run_command = self._make_run_command()

        return "\n".join([header, env_opts, command, run_command])

    def write_job(self) -> None:
        """Method to write job script into file."""
        with open(self.run_file, "w") as f:
            f.write(self._make_body())

    def write_jobid(self) -> None:
        """Method to write jobid into log file."""
        with open(self.jobid_log_file, "a") as file:
            file.write(str(self.run_file.name) + " " + str(self.jobid) + "\n")

        if self.out_file is not None:
            with open(self.out_file, "w") as file:
                file.write(str(self.jobid))

    def send_job(self) -> None:
        raise ValueError(
            "Tasker class has no send_job defined, use child classes instead."
        )

    def _make_header(self) -> str:
        raise ValueError(
            "Tasker class has no _make_header defined, use child classes instead."
        )

    def _make_env_opts(self) -> str:
        raise ValueError(
            "Tasker class has no _make_env_opts defined, use child classes instead."
        )

    def _make_run_command(self) -> str:
        raise ValueError(
            "Tasker class has no _make_run_command defined, use child classes instead."
        )

    @staticmethod
    def get_jobid_status(jobid: int) -> str:
        logger.warning(
            "Requesting jobid status for a non-slurm object. Returning COMPLETED"
        )
        return "COMPLETED"


class SlurmTasker(Tasker):
    """Object to write and run jobs.

    Attributes:
        slurm_header_args (dict): Header options to write. Use a dictionary with the format {'option_name': 'option_value'}
        command (str): Command to be run in the job.
        command_args (str): Arguments to the command.
        environment (str): Conda/python environment to load before running the command.
        environmental_variables (dict): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}.
        srun_options (str): Options for the srun command.
        run_file (Path): Location of the job file.
        wait_for (Tasker or str): Tasker to wait for before running (if Tasker) or jobid of the task to wait (if str).
    """

    default_srun_options = {
        "nodes": 1,  # N
        "ntasks": 1,  # n
    }

    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
        "cpus-per-task": "256",
    }

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            command (str): Command to be run in the job.
            command_args (str): Arguments to the command.
            slurm_header_args (dict): Header options to write if slurm tasker is selected. Use a dictionary with the format {'option_name': 'option_value'}.
            srun_options (str): If slurm tasker selected. Options for the srun command.
            environment (str): Conda/python environment to load before running the command.
            run_file (Path): Location of the job file.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            environmental_variables (dict, optional): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}. Default: No environmental variables defined.
        """
        super().__init__(*args, **kwargs)

    def _make_header(self) -> str:
        """Method to generate a slurm header given the slurm_header_args attribute.

        Returns:
            str: Slurm header.
        """
        header = "#!/bin/bash -l\n\n"
        header += "\n".join(
            [
                f"#SBATCH --{key} {value}"
                for key, value in self.slurm_header_args.items()
            ]
        )
        return header

    def _make_env_opts(self) -> str:
        """Method to generate environmental options.

        Returns:
            str: environmental options."""
        if self.slurm_header_args.get("cpus-per-task", None) is not None:
            self.srun_options["cpus-per-task"] = self.slurm_header_args["cpus-per-task"]

        text = textwrap.dedent(
            f"""
module load python
source activate {self.environment}
umask 0002
export OMP_NUM_THREADS={self.slurm_header_args['cpus-per-task']}

"""
        )
        for key, value in self.environmental_variables.items():
            text += f"export {key}={value}\n"

        return text

    def _make_run_command(self) -> str:
        """Method to generate the srun command.

        Returns:
            str: srun command."""
        args = " ".join(
            [f"--{key} {value}" for key, value in self.srun_options.items()]
        )
        return f"srun {args} $command\n"

    def send_job(self) -> None:
        """Method to send job to slurm queue. Setting the wait_for variables beforehand."""
        # Changing dir into the file position to
        # avoid slurm files being written in unwanted places
        _ = Path(".").resolve()
        os.chdir(self.run_file.parent)

        if self.wait_for is None and len(self.in_files) == 0:
            wait_for_str = ""
        else:
            self.get_wait_for_ids()

            if len(self.wait_for_ids) == 0:
                wait_for_str = ""
            else:
                wait_for_str = f"--dependency=afterok:"
                wait_for_str += ",afterok:".join(map(str, self.wait_for_ids))

        self.sbatch_process = run(
            f"sbatch --parsable {wait_for_str} {self.run_file}",
            shell=True,
            capture_output=True,
        )
        os.chdir(_)

        if self.sbatch_process.returncode == 0:
            self.jobid = int(self.sbatch_process.stdout.decode("utf-8"))
            self.write_jobid()
        else:
            raise ValueError(
                f'Submitting job failed. {self.sbatch_process.stderr.decode("utf-8")}'
            )

    @staticmethod
    def get_jobid_status(jobid: int) -> str:
        """
        Method to return the status of a given jobid in SLURM systems.

        Args:
            jobid: Identificator of the job to obtain the status for.
        """
        sbatch_process = run(
            f"sacct -j {jobid} -o State --parsable2 -n",
            shell=True,
            capture_output=True,
        )

        try:
            return sbatch_process.stdout.decode("utf-8").splitlines()[0]
        except:
            logger.info(
                f"Retreiving status for jobid {jobid} failed. Assuming failed status."
            )
            return "FAILED"


class SlurmCoriTasker(SlurmTasker):
    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
        "constraint": "haswell",
        "account": "desi",
    }


class SlurmPerlmutterTasker(SlurmTasker):
    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
        "constraint": "cpu",
        "account": "desi",
    }

    default_srun_options = {
        "nodes": 1,  # N
        "ntasks": 1,  # n
    }


class BashTasker(Tasker):
    """Object to write and run jobs.

    Attributes:
        command (str): Command to be run in the job.
        command_args (str): Arguments to the command.
        environment (str): Conda/python environment to load before running the command.
        environmental_variables (dict): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            command (str): Command to be run in the job.
            command_args (str): Arguments to the command.
            environment (str): Conda/python environment to load before running the command.
            run_file (Path): Location of the job file.
            environmental_variables (dict, optional): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}. Default: No environmental variables defined.
        """
        super().__init__(*args, **kwargs)
        if self.wait_for != None:
            raise ValueError("BachTasker won't work with wait_for feature.")

    def _make_header(self) -> str:
        """Dummy method to generate an empty header and keep compatibility with Tasker default class."""
        return ""

    def _make_env_opts(self) -> str:
        """Method to generate environmental options.

        Returns:
            str: environmental options."""
        return textwrap.dedent(
            f"""
module load python
source activate {self.environment}
"""
        )

    def _make_run_command(self) -> str:
        """Method to genearte the run command line.

        Returns:
            str: run command line."""
        return f"$command\n"

    def send_job(self) -> None:
        """Method to run bash job script."""
        out = open(self.slurm_header_args["output"], "w")
        err = open(self.slurm_header_args["error"], "w")

        _ = Path(".").resolve()
        os.chdir(self.run_file.parent)
        self.retcode = run(["sh", f"{self.run_file}"], stdout=out, stderr=err)


class ChainedTasker:
    """Object to run chained Taskers.

    Attributes:
        taskers (tasker.Tasker or list of tasker.Tasker): tasker objects associated with the class
    """

    def __init__(self, taskers: List[Tasker]):
        """
        Args:
            taskers (tasker.Tasker or list of tasker.Tasker): tasker objects associated with the class
        """
        self.taskers = taskers

    def write_job(self) -> None:
        """Method to write jobs associated with taskers."""
        for tasker in self.taskers:
            tasker.write_job()

    def send_job(self) -> None:
        """Method to send jobs associated with taskers."""
        for tasker in self.taskers:
            tasker.send_job()

    @property
    def jobid(self) -> Optional[int]:
        return self.taskers[-1].jobid


class DummyTasker(Tasker):
    """Tasker object that performs no action. Useful for when files
    are copied and runs are not needed.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def write_job(self) -> None:
        pass

    def send_job(self) -> None:
        self.jobid = None
        return None
