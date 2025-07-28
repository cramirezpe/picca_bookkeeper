"""
picca_bookkeeper/tasker.py

This module defines the core job scheduling infrastructure for picca_bookkeeper.
It provides classes and utilities for creating, managing, and executing
computational tasks, both locally and on SLURM-based HPC systems (such as Cori
and Perlmutter).

The principal components are:
    - Tasker: The base class for individual job definitions, encapsulating
              command-line calls, environment setup, and file dependencies.
              Child classes implement system-specific logic for writing and
              submitting job scripts.
    - SlurmTasker, SlurmCoriTasker, SlurmPerlmutterTasker: Specializations
              for submitting jobs to specific SLURM-managed clusters.
    - BashTasker: For running jobs directly in a shell environment without SLURM.
    - ChainedTasker: For managing and executing sequences of dependent tasks.
    - DummyTasker: A no-op placeholder used when actual computation is skipped
              (e.g., when copying output files instead of recomputing).

Interaction with Other Modules:
------------------------------
    - Used heavily by picca_bookkeeper/bookkeeper.py, which orchestrates pipelines
      by generating Tasker objects for each processing step
      (e.g., correlation computation, fitting, exporting).
    - Bookkeeper generates Taskers tailored to the user's system and workflow,
      then coordinates their execution via this module.
    - Taskers interact with other modules such as dict_utils for configuration and
      paths for file management, and are responsible for writing job scripts and
      logging job IDs.
    - Supports dependencies between jobs using SLURM's afterok mechanism or
      file-based signaling, enabling complex workflows.

Usage:
------
    - Typical usage involves a higher-level workflow module (Bookkeeper) instantiating
      Taskers for each step in a pipeline, configuring command-line arguments,
      input/output files, and system-specific settings. Taskers then write and submit
      job scripts, handling environment setup and logging.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import textwrap
import time
from datetime import datetime
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


def get_Tasker(system: str) -> Type[Tasker]:
    """
    Returns the appropriate Tasker subclass based on the specified system.

    This function selects the appropriate Tasker implementation for the target
    computing environment. It supports SLURM-based systems (e.g., Cori and
    Perlmutter) as well as local shell execution via Bash.

    Arguments:
    ----------
        system (str): Identifier for the target system. Valid options are:
            - 'slurm_cori': Use SLURM scripts for the Cori HPC system.
            - 'slurm_perlmutter': Use SLURM scripts for the Perlmutter HPC system.
            - 'bash': Use a local shell script for login nodes or standalone machines.
            - 'slurm': Deprecated alias for 'slurm_cori'.

    Returns:
    ----------
        Type[Tasker]: A subclass of Tasker configured for the specified system.

    Raises:
    ----------
        ValueError: If an invalid system name is provided.

    Warnings:
    ----------
        DeprecationWarning: If the 'slurm' alias is used instead of a specific system.
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
    """
    Base class for defining and managing computational jobs.

    This class provides functionality for generating job scripts, managing
    dependencies, setting environments and variables, and writing job metadata.
    It is intended to be subclassed by system-specific implementations that
    define how jobs are submitted and executed (e.g., SLURM-based clusters).

    Attributes:
    ----------
        slurm_header_args (dict): SLURM header options for job submission.
            Use a dictionary with the format {'option_name': 'option_value'}
        command (str): Base command to be executed.
        command_args (dict): Dictionary of command-line arguments.
        packages (List[str]): Python packages to log version info for.
        precommand (str): Optional shell command to run before the main command.
        environment (str): Name of the conda or Python environment to activate.
        environmental_variables (dict): Key-value pairs of environment variables to set.
            Format: {'environmental_variable': 'value'}.
        srun_options (dict): SLURM `srun` options for job execution.
        run_file (Path): File path for the job script to be written.
        jobid_log_file (Path): Path to the file logging job submission IDs.
        wait_for (various): Taskers or job IDs that this job should wait for.
            running (if Tasker) or jobid of the task to wait (if str).
        in_files (List[Path | str]): Input files whose presence/status gates job execution.
            Input files that must exists or contain a jobid in order for the
            job to be launched.
        out_files (List[Path | str]): Output files where job ID may be written.
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
        packages: List[str] = [],
        wait_for: Optional[
            ChainedTasker | Tasker | List[Type[Tasker]] | int | List[int]
        ] = None,
        environmental_variables: Dict = dict(),
        srun_options: Dict = dict(),
        in_files: List[Path] | List[str] = list[Path](),
        out_files: List[Path | str] = [],
        precommand: str = "",
    ):
        """
        Arguments:
        ----------
            command (str): Command to be run in the job.
            command_args (str): Arguments to the command.
            packages (List[str]): Packages to list versions from.
            slurm_header_args (dict): Header options to write if slurm tasker is selected. Use a dictionary with the format {'option_name': 'option_value'}.
            srun_options (str): If slurm tasker selected. Options for the srun command.
            environment (str): Conda/python environment to load before running the command.
            run_file (Path): Location of the job file.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before running the current one. Could be a  Tasker object or a slurm jobid (int). (Default: None, won't wait for anything).
            environmental_variables (dict, optional): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}. Default: No environmental variables defined.
            jobid_log_file (Path): Location of log file where to include jobids of runs.
            in_files: Input files that must exists or contain a jobid in order for the
            job to be launched.
            out_files: Out files that will be write by the job (to add jobid if available).
        """
        self.jobid: Optional[int] = None

        if "OMP_NUM_THREADS" in slurm_header_args:
            if slurm_header_args["OMP_NUM_THREADS"] not in ("", None):
                self.OMP_threads = slurm_header_args["OMP_NUM_THREADS"]
            else:
                self.OMP_threads = None

            del slurm_header_args["OMP_NUM_THREADS"]
        else:
            self.OMP_threads = None

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
        self.packages = packages
        self.environment = environment
        self.environmental_variables = environmental_variables
        self.srun_options = {**self.default_srun_options, **srun_options}
        self.run_file = Path(run_file)
        self.wait_for = wait_for
        self.jobid_log_file = jobid_log_file
        self.in_files = in_files
        self.out_files = out_files
        self.precommand = precommand

    def get_wait_for_ids(self) -> None:
        """
        Resolves and stores the list of job IDs this task should wait for.
    
        This method processes the `wait_for` attribute, which may contain
        Tasker instances, integers (SLURM job IDs), or input files with
        embedded job IDs. It also checks for input file existence and extracts
        job IDs from those that appear to be dependency markers.
    
        Raises:
        ----------
            Exception: If a Tasker object in `wait_for` does not have a job ID set.
            FileNotFoundError: If any required input file is missing.
            ValueError: If `wait_for` contains an unrecognized type.
        """
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
            if size > 1 and size < 20:
                jobid = int(Path(file).read_text().splitlines()[0])
                status = self.get_jobid_status(jobid)

                if status != "COMPLETED":
                    self.wait_for_ids.append(jobid)

    def _make_command(self) -> str:
        """
        Constructs the shell command with its arguments.
    
        Returns:
        ----------
            str: The full command line string, wrapped for assignment to a shell variable.
        """
        args_list = []
        for key, value in self.command_args.items():
            if key != "":
                if key.startswith("-"):
                    args_list.append(f"{key} {value}")
                else:
                    args_list.append(f"--{key} {value}")
            else:
                args_list.append(str(value))

        args = " ".join(args_list)
        return f'command="{self.command} {args}"'

    def _make_body(self) -> str:
        """
        Constructs the full content of the job script.
    
        This includes the header, environment setup, version logging, and
        final execution call wrapped with date stamps.
    
        Returns:
        ----------
            str: The complete shell script body as a string.
        """
        header = self._make_header()
        env_opts = self._make_env_opts()
        version_control = self._make_version_control()
        command = self._make_command()
        run_command = self._make_run_command()

        return "\n".join(
            [
                header,
                env_opts,
                version_control,
                command,
                "date",
                run_command,
                "date",
                "",
            ]
        )

    def write_job(self) -> None:
        """
        Writes the constructed job script into the specified run file.
        """
        with open(self.run_file, "w") as f:
            f.write(self._make_body())

    def write_jobid(self) -> None:
        """
        Records the submitted job ID in the log file and output files.
    
        Appends the job ID to a global job log file, and optionally writes
        it to specified output files.
        """
        with open(self.jobid_log_file, "a") as file:
            file.write(str(self.run_file.name) + " " + str(self.jobid) + "\n")

        for out_file in self.out_files:
            with open(out_file, "w") as file:
                file.write(str(self.jobid))

    def send_job(self) -> None:
        """
        Placeholder for job submission logic.
    
        Raises:
        ----------
            ValueError: Always raised in the base class. Subclasses must override this method.
        """
        raise ValueError(
            "Tasker class has no send_job defined, use child classes instead."
        )

    def _make_header(self) -> str:
        """
        Placeholder for generating SLURM or script headers.
    
        Raises:
        ----------
            ValueError: Always raised in the base class. 
                        Subclasses must override this method.
        """
        raise ValueError(
            "Tasker class has no _make_header defined, use child classes instead."
        )

    def _make_env_opts(self) -> str:
        """
        Placeholder for generating environment activation commands.
    
        Raises:
            ValueError: Always raised in the base class. 
                        Subclasses must override this method.
        """
        raise ValueError(
            "Tasker class has no _make_env_opts defined, use child classes instead."
        )

    def _make_version_control(self) -> str:
        """
        Creates shell commands for printing version information of used packages.
    
        Returns:
            str: Bash code that echoes the version of picca_bookkeeper and other packages.
        """
        text = "\necho used picca_bookkeeper version: "
        text += importlib.metadata.version("picca_bookkeeper")

        for package in self.packages:
            text += f'\necho using {package} version: $(python -c "import importlib.metadata; '
            text += f"print(importlib.metadata.version('{package}'))\")"

        return text + "\necho -e '\\n'\n"

    def _make_run_command(self) -> str:
        """
        Placeholder for defining how the job's command is executed.
    
        Raises:
            ValueError: Always raised in the base class. 
                        Subclasses must override this method.
        """
        raise ValueError(
            "Tasker class has no _make_run_command defined, use child classes instead."
        )

    @staticmethod
    def get_jobid_status(jobid: int) -> str:
        """
        Query the SLURM scheduler for the status of a specific job ID.
    
        Attempts to retrieve the job's state using the `sacct` command, retrying up
        to 10 times with short delays if the query fails.
    
        Arguments:
        ----------
            jobid (int): SLURM job ID to check.
    
        Returns:
        ----------
            str: The final reported job state (e.g., 'COMPLETED', 'FAILED').
    
        Logs:
        ----------
            - Debug message on each retry if status retrieval fails.
            - Info message if all attempts fail, defaulting to "FAILED".
        """
        tries = 0
        while tries < 10:
            sbatch_process = run(
                f"sacct -j {jobid} -o State --parsable2 -n",
                shell=True,
                capture_output=True,
            )

            try:
                return sbatch_process.stdout.decode("utf-8").splitlines()[-1]
            except:
                logger.debug(
                    f"Retrieving status for jobid {jobid} failed. Retrying in 2 seconds..."
                )
                time.sleep(2)

        logger.info(f"Retrieving status failed. Assuming job failed.")
        return "FAILED"


class SlurmTasker(Tasker):
    """
    SLURM-specific subclass of Tasker for writing and running jobs to SLURM-based clusters.

    This class defines how to build SLURM-compliant job scripts, load environments,
    and submit jobs using `sbatch`.

    Attributes:
    ----------
        default_srun_options (dict): Default `srun` options such as node/task count.
        default_header (dict): Default SLURM script headers (e.g., time, cpus).
        slurm_header_args (dict): SLURM-specific options for job header.
            Use a dictionary with the format {'option_name': 'option_value'}
        command (str): Command to execute.
        command_args (dict): Arguments passed to the command.
        environment (str): Environment to activate (conda or sourced script).
            Conda/python environment to load before running the command.
        environmental_variables (dict): Shell variables to export before execution.
            Format: {'environmental_variable': 'value'}.
        srun_options (dict): Arguments passed to the `srun` command.
        run_file (Path): Path to the generated job script.
        wait_for (Tasker | int | list): Job dependencies.
            Tasker to wait for before running (if Tasker) or jobid of the task to wait (if str).
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
        Initialize a SlurmTasker with inherited Tasker options.
    
        Arguments:
        ----------
            *args, **kwargs: All arguments are passed to the base Tasker class.

            command (str): Command to be run in the job.
            command_args (str): Arguments to the command.
            slurm_header_args (dict): Header options to write if slurm tasker is selected. 
                Use a dictionary with the format {'option_name': 'option_value'}.
            srun_options (str): If slurm tasker selected. Options for the srun command.
            environment (str): Conda/python environment to load before running the command.
            run_file (Path): Location of the job file.
            wait_for (Tasker or int, optional): In NERSC, wait for a given job to finish before 
                running the current one. Could be a  Tasker object or a slurm jobid (int). 
                (Default: None, won't wait for anything).
            environmental_variables (dict, optional): Environmental variables to set before 
                running the job. Format: {'environmental_variable': 'value'}. 
                Default: No environmental variables defined.
    
        Inherits:
        ----------
            See Tasker.__init__ for full list of accepted arguments.
        """
        super().__init__(*args, **kwargs)

    def _make_header(self) -> str:
        """
        Generate the SLURM header section for the job script given the slurm_header_args attribute.
    
        Returns:
        ----------
            str: SLURM job header containing `#SBATCH` directives.
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
        """
        Generate shell commands to set environment variables and activate the environment.
    
        Returns:
        ----------
            str: Shell script section for environment setup, including module loading,
            conda activation or shell sourcing, and exported variables.
        """
        if self.OMP_threads is not None:
            text = f"export OMP_NUM_THREADS={self.OMP_threads}\n"
        else:
            text = ""

        if self.environment.endswith(".sh"):
            activate = "source "
        else:
            activate = "conda activate "

        text += textwrap.dedent(
            f"""
module load python
{activate}{self.environment}
umask 0002

"""
        )
        for key, value in self.environmental_variables.items():
            text += f"export {key}={value}\n"

        text += self.precommand + "\n"

        return text

    def _make_run_command(self) -> str:
        """
        Construct the `srun` command line used to execute the job.
    
        Returns:
            str: Final line to run the job with `srun` and user-defined options.
        """
        args = " ".join(
            [f"--{key} {value}" for key, value in self.srun_options.items()]
        )
        return f"srun {args} $command\n"

    def send_job(self) -> None:
        """
        Submit the job script to the SLURM scheduler / queue.
    
        Handles job dependencies via SLURM's `--dependency=afterok:` flag, writes
        the assigned job ID to the log, and updates the Tasker's `jobid`.
        Sets the wait_for variables beforehand.
    
        Raises:
        ----------
            ValueError: If SLURM job submission fails (e.g., sbatch returns non-zero).
        """
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


class SlurmCoriTasker(SlurmTasker):
    """
    SLURM tasker with default options for NERSC Cori Haswell nodes.
    """    
    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
        "constraint": "haswell",
        "account": "desi",
        "cpus-per-task": 128,
    }

    default_srun_options = {
        "nodes": 1,  # N
        "ntasks": 1,  # n
        "cpus-per-task": 128,
    }


class SlurmPerlmutterTasker(SlurmTasker):
    """
    SLURM tasker with default options for NERSC Perlmutter CPU nodes.
    """
    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
        "constraint": "cpu",
        "account": "desi",
        "ntasks-per-node": 1,
    }

    default_srun_options = {}


class BashTasker(Tasker):
    """
    Tasker that creates and runs a simple bash script outside of SLURM.

    Attributes:
    ----------
        command (str): Base command to run.
        command_args (str): Optional arguments to the command.
        environment (str): Conda or shell environment to activate.
        environmental_variables (dict): Environment variables to export prior to execution.
            Format: {'environmental_variable': 'value'}.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize BashTasker.
                
        Arguments:
        ----------
            command (str): Base command to run.
            command_args (str): Optional arguments to the command.
            environment (str): Conda or shell environment to activate.
            environmental_variables (dict): Environment variables to export prior to execution.
                Format: {'environmental_variable': 'value'}.
            
         Raises:
         ----------
            ValueError: If `wait_for` is set, which is unsupported for BashTasker.
        """
        super().__init__(*args, **kwargs)
        if self.wait_for != None:
            raise ValueError("BachTasker won't work with wait_for feature.")

    def _make_header(self) -> str:
        """
        Return an empty string to override SLURM header behavior.
        
        Dummy method to generate an empty header and keep 
        compatibility with Tasker default class.
        """
        return ""

    def _make_env_opts(self) -> str:
        """
        Generate commands to set environment variables and activate environment.

        Returns:
        ----------
            str: Environment setup commands to prepend to the job script.
        """
        if self.OMP_threads is not None:
            text = f"export OMP_NUM_THREADS={self.OMP_threads}\n"
        else:
            text = ""

        if "sh" in self.environment:
            activate = ""
        else:
            activate = "activate "

        text += textwrap.dedent(
            f"""
module load python
source {activate}{self.environment}
umask 0002

"""
        )

        text += self.precommand + "\n"

        return text

    def _make_run_command(self) -> str:
        """
        Method to genearte the run command line.

        Returns:
        ----------
            str: Bash command line to run.
            """
        return f"$command\n"

    def send_job(self) -> None:
        """
        Execute the generated bash job script, 
        redirecting stdout and stderr.
        """
        out = open(
            self.slurm_header_args["output"].replace(
                "%j", datetime.now().strftime("%d%m%Y%H%M%S")
            ),
            "w",
        )
        err = open(
            self.slurm_header_args["error"].replace(
                "%j", datetime.now().strftime("%d%m%Y%H%M%S")
            ),
            "w",
        )

        _ = Path(".").resolve()
        os.chdir(self.run_file.parent)
        self.retcode = run(["sh", f"{self.run_file}"], stdout=out, stderr=err)
        os.chdir(_)


class ChainedTasker:
    """
    Run a list of Tasker objects sequentially (chained taskers).

    Attributes:
    ----------
        taskers (List[Tasker]): List of Tasker instances to run.
            (tasker.Tasker or list of tasker.Tasker)
    """

    def __init__(self, taskers: List[Tasker]):
        """
        Args:
            taskers (tasker.Tasker or list of tasker.Tasker): tasker objects associated with the class
        """
        self.taskers = taskers

    def write_job(self) -> None:
    """
    Write job files for each tasker in the chain.
    """
        for tasker in self.taskers:
            tasker.write_job()

    def send_job(self) -> None:
        """
        Method to send jobs associated with taskers.
        """
        for tasker in self.taskers:
            tasker.send_job()

    @property
    def jobid(self) -> Optional[int]:
        """
        Return the job ID of the last tasker in the chain.
        """
        return self.taskers[-1].jobid


class DummyTasker(Tasker):
    """Tasker object that performs no action. Useful for when files
    are copied and runs are not needed.
    """

    """
    Tasker object that performs no action when no execution is needed.
    
    Used in workflows where job writing or running is intentionally skipped 
    (e.g. files are coppied but runs not needed).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize DummyTasker with ignored arguments.
        """
        pass

    def write_job(self) -> None:
        """
        No-op for write_job.
        """
        pass

    def send_job(self) -> None:
        """
        No-op for send_job. Sets `jobid` to None.
        """
        self.jobid = None
        return None
