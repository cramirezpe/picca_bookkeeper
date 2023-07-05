import logging
import os
import sys
import textwrap
from pathlib import Path
from subprocess import run

import numpy as np

logger = logging.getLogger(__name__)


def get_Tasker(system, *args, **kwargs):
    """Function to get a Tasker object for a given system.

    Args:
        system (str): Shell the Tasker will use. 'slurm_cori' to use
            slurm scripts on cori, slurm_perlmutter' to use slurm scripts
            on perlmutter, 'bash' to run it in login nodes or computer shell.
        *args: Will be sent to the Tasker constructor.
        **kwargs: Will be sent to the Tasker constructor.
    """
    if system == "slurm":
        from warnings import warn

        warn(
            "slurm system option is deprecated,"
            "use slurm_cori or slurm_perlmutter instead",
            DeprecationWarning,
        )
        return SlurmCoriTasker(*args, **kwargs)
    elif system == "slurm_cori":
        return SlurmCoriTasker(*args, **kwargs)
    elif system == "slurm_perlmutter":
        return SlurmPerlmutterTasker(*args, **kwargs)
    elif system == "bash":
        return BashTasker(*args, **kwargs)
    else:
        raise ValueError(
            f"System not valid: {system} "
            "allowed options are slurm_cori, slurm_perlmutter, "
            "bash."
        )


class Tasker:
    """Object to write and run jobs.

    Attributes:
        slurm_header_args (dict): Header options to write if slurm tasker is selected. Use a dictionary with the format {'option_name': 'option_value'}
        command (str): Command to be run in the job.
        command_args (str): Arguments to the command.
        environment (str): Conda/python environment to load before running the command.
        environmental_variables (dict): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}.
        srun_options (str): If slurm tasker selected. Options for the srun command.
        run_file (Path): Location of the job file.
        wait_for (Tasker or str): If slurm tasker selected. Tasker to wait for before running (if Tasker) or jobid of the task to wait (if str).
    """

    default_srun_options = dict()
    default_header = dict()

    def __init__(
        self,
        command,
        command_args,
        slurm_header_args,
        srun_options,
        environment,
        run_file,
        wait_for,
        environmental_variables=dict(),
        jobid_log_file=None,
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

    def get_wait_for_ids(self):
        """Method to standardise wait_for Taskers or ids, in such a way that can be easily used afterwards."""
        if self.wait_for is None:
            self.wait_for_ids = None
        if np.ndim(self.wait_for) == 0:
            if isinstance(self.wait_for, (int, np.integer)):
                self.wait_for_ids = [self.wait_for]
            elif isinstance(self.wait_for, Tasker) or isinstance(
                self.wait_for, ChainedTasker
            ):
                try:
                    self.wait_for_ids = [self.wait_for.jobid]
                except AttributeError as e:
                    raise Exception(
                        f"jobid not defined for wait_for input. Have you send any script with this object?"
                    ).with_traceback(e.__traceback__)
        else:
            self.wait_for = list(np.array(self.wait_for).reshape(-1))
            if isinstance(self.wait_for[0], (int, np.integer)):
                self.wait_for = [x for x in self.wait_for if x is not None]
                self.wait_for_ids = list(self.wait_for)
            elif self.wait_for[0] == None:
                self.wait_for = [x for x in self.wait_for if x is not None]
                if len(self.wait_for) == 0:
                    self.wait_for = None
                    self.wait_for_ids = None
                else:
                    self.wait_for_ids = list(self.wait_for)
            elif isinstance(self.wait_for[0], Tasker) or isinstance(
                self.wait_for[0], ChainedTasker
            ):
                try:
                    self.wait_for_ids = [x.jobid for x in self.wait_for]
                except AttributeError as e:
                    raise Exception(
                        f"jobid not defiend for some of the wait_for. Have you send any script with each of the wait_for objects?"
                    ).with_traceback(e.__traceback__)
            else:
                raise TypeError(
                    f"Only types supported by wait_for are int, Tasker, list of int or list of Tasker: {type(self.wait_for)}"
                )

    def _make_command(self):
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

    def _make_body(self):
        """Method to generate the body of the job script.

        Return:
            str: Body of the job script."""
        header = self._make_header()
        env_opts = self._make_env_opts()
        command = self._make_command()
        run_command = self._make_run_command()

        return "\n".join([header, env_opts, command, run_command])

    def write_job(self):
        """Method to write job script into file."""
        if self.run_file.is_file():
            with open(self.run_file, "r") as f:
                if self._make_body() != f.read():
                    raise ValueError(
                        "Script already generated and different in target.",
                        str(self.run_file),
                    )

        with open(self.run_file, "w") as f:
            f.write(self._make_body())

    def write_jobid(self):
        """Method to write jobid into log file."""
        with open(self.jobid_log_file, "a") as file:
            file.write(str(self.jobid) + "\n")


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
        "cpus-per-task": 32,  # c
    }

    default_header = {
        "qos": "regular",
        "nodes": 1,
        "time": "00:30:00",
    }

    def __init__(self, *args, **kwargs):
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

    def _make_header(self):
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

    def _make_env_opts(self):
        """Method to generate environmental options.

        Returns:
            str: environmental options."""
        text = textwrap.dedent(
            f"""
module load python
source activate {self.environment}
umask 0002
export OMP_NUM_THREADS={self.srun_options['cpus-per-task']}

"""
        )
        for key, value in self.environmental_variables.items():
            text += f"export {key}={value}\n"

        return text

    def _make_run_command(self):
        """Method to generate the srun command.

        Returns:
            str: srun command."""
        args = " ".join(
            [f"--{key} {value}" for key, value in self.srun_options.items()]
        )
        return f"srun {args} $command\n"

    def send_job(self):
        """Method to send job to slurm queue. Setting the wait_for variables beforehand."""
        # Changing dir into the file position to
        # avoid slurm files being written in unwanted places
        _ = Path(".").resolve()
        os.chdir(self.run_file.parent)

        if self.wait_for is None:
            wait_for_str = ""
        else:
            self.get_wait_for_ids()

            if self.wait_for_ids is None:
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
        "cpus-per-task": 128,  # c
    }


class BashTasker(Tasker):
    """Object to write and run jobs.

    Attributes:
        command (str): Command to be run in the job.
        command_args (str): Arguments to the command.
        environment (str): Conda/python environment to load before running the command.
        environmental_variables (dict): Environmental variables to set before running the job. Format: {'environmental_variable': 'value'}.
    """

    def __init__(self, *args, **kwargs):
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

    def _make_header(self):
        """Dummy method to generate an empty header and keep compatibility with Tasker default class."""
        return ""

    def _make_env_opts(self):
        """Method to generate environmental options.

        Returns:
            str: environmental options."""
        return textwrap.dedent(
            f"""
module load python
source activate {self.environment}
"""
        )

    def _make_run_command(self):
        """Method to genearte the run command line.

        Returns:
            str: run command line."""
        return f"$command\n"

    def send_job(self):
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

    def __init__(self, taskers):
        """
        Args:
            taskers (tasker.Tasker or list of tasker.Tasker): tasker objects associated with the class
        """
        self.taskers = taskers

    def write_job(self):
        """Method to write jobs associated with taskers."""
        for tasker in self.taskers:
            tasker.write_job()

    def send_job(self):
        """Method to send jobs associated with taskers."""
        for tasker in self.taskers:
            tasker.send_job()

    @property
    def jobid(self):
        return self.taskers[-1].jobid
