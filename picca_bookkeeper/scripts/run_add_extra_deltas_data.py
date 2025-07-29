"""
run_add_extra_deltas_data.py
----------------------------

Script to generate and optionally submit SLURM jobs for adding extra info to
picca delta files.

Functionality:
--------------
    - Prepares and manages SLURM batch jobs to run the
      `picca_bookkeeper_add_extra_deltas_data` command, which processes delta
      files to add mask and/or flux information.
    - Handles job submission or script writing for later execution.
    - Determines the correct input / output directories, and region-specific
      config files, from the Bookkeeper object and user arguments.
    - Allows selective addition of mask and/or flux metadata to delta files,
      and parallelizes the computation.
    - Supports job dependencies (e.g. `--wait-for`) and writing jobs without
      immediately submitting them (e.g. `--only-write`).

Usage:
------
    - Run from the command line, providing a Bookkeeper config file and desired options.
      This script is not intended to be run directly on data but to manage and
      orchestrate SLURM job scripts.

Example:
--------
python run_add_extra_deltas_data.py /path/to/bookkeeper_config.yaml --region lya
--add-mask --add-flux --num-processors 32

This command will:
    - Generate and submit a SLURM job to add mask and flux information to delta
      files in the 'lya' region, using 32 processors.
    - Input/output locations and environment are configured via the provided
      Bookkeeper config.

Key Arguments:
--------------
    - bookkeeper_config: Path to the Bookkeeper configuration file.
    - --overwrite-config: Overwrite Bookkeeper config with the current run's settings.
    - --region: Region to process (e.g., 'lya', 'calibration_1').
    - --num-processors: Number of processors to use for parallelization
                        (default: 16).
    - --add-mask: Add extra mask information to the output.
    - --add-flux: Add flux information to the output.
    - --only-write: Only write SLURM scripts, do not submit them.
    - --wait-for: One or more SLURM job IDs to wait for before submitting this job.

Interaction:
-------------
    - Uses `Bookkeeper` to manage data paths, configurations, and SLURM header args.
    - Uses `get_Tasker` to create and manage SLURM job scripts and submission.
    - Ultimately launches the `add_extra_deltas_data.py` script, which performs
      the actual data processing (reading deltas, applying masks, saving new files).
    - Input / output files are organized in directories determined by Bookkeeper conventions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import get_Tasker

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Entry point for generating and optionally submitting SLURM jobs to add mask
    and/or flux information to PICCA delta files.

    Functionality:
    --------------
        - Loads a Bookkeeper configuration.
        - Determines paths and settings for delta file processing based on the
          specified region.
        - Constructs arguments for the `picca_bookkeeper_add_extra_deltas_data`
          command.
        - Builds a SLURM job script using `get_Tasker`.
        - Optionally submits the job unless `--only-write` is specified.

    Arguments:
    ----------
        args (argparse.Namespace, optional): Command-line arguments. If None,
            arguments are parsed from sys.argv using `get_args()`.

    Raises:
    -------
        - FileNotFoundError: If required configuration or delta files are missing.
        - StopIteration: If no matching configuration file is found for the region.
    """
    if args is None:
        args = get_args()
    bookkeeper = Bookkeeper(
        args.bookkeeper_config, overwrite_config=args.overwrite_config
    )

    command = "picca_bookkeeper_add_extra_deltas_data"

    if args.region == "calibration_1":
        region = None
        calib_step = 1
        pattern = "delta_extraction_*_calib_step_1.ini"
    elif args.region == "calibration_2":
        region = None
        calib_step = 2
        pattern = "delta_extraction_*_calib_step_1.ini"
    else:
        region = args.region
        calib_step = None
        pattern = f"delta_extraction_{region}.ini"

    deltas_path = bookkeeper.paths.deltas_path(
        region=args.region,
        calib_step=calib_step,
    )

    picca_config = next((bookkeeper.paths.run_path / "configs").glob(pattern))

    output_dir_mask = deltas_path.parent / "Mask"
    output_dir_flux = deltas_path.parent / "Flux"

    command_args = {
        "deltas-input": deltas_path,
        "picca-config": picca_config,
        "num-processors": args.num_processors,
        "log-level": "INFO",
    }

    if args.add_mask:
        command_args["output-dir-mask"] = output_dir_mask
    if args.add_flux:
        command_args["output-dir-flux"] = output_dir_flux

    if region is None:
        region = f"calibration_{calib_step}"

    job_name = f"add_extra_deltas_data_{region}"

    updated_slurm_header_args = bookkeeper.generate_slurm_header_extra_args(
        config=bookkeeper.config,
        section="delta extraction",
        command=command,
        region=region,
    )
    srun_options = {
        "nodes": 1,
        "ntasks": 1,
        "cpus-per-task": args.num_processors,
    }

    system = bookkeeper.generate_system_arg(None)

    tasker = get_Tasker(system)(
        command=command,
        command_args=command_args,
        slurm_header_args=updated_slurm_header_args,
        srun_options=srun_options,
        environment=bookkeeper.config["general"]["conda environment"],
        in_files=[bookkeeper.paths.delta_attributes_file(region)],
        run_file=bookkeeper.paths.run_path / f"scripts/run_{job_name}.sh",
        jobid_log_file=bookkeeper.paths.run_path / f"logs/jobids.log",
        wait_for=args.wait_for,
    )

    tasker.write_job()
    if not args.only_write:
        tasker.send_job()


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the SLURM job orchestration script.

    Returns:
    --------
        argparse.Namespace: Parsed arguments with the following attributes:
            - bookkeeper_config (Path): Path to the Bookkeeper config file.
            - overwrite_config (bool): Whether to overwrite the config during
                    job generation.
            - region (str): The region to process (default: 'lya').
            - num_processors (int): Number of processors to use (default: 16).
            - add_mask (bool): Whether to add extra mask metadata.
            - add_flux (bool): Whether to add flux metadata.
            - only_write (bool): Whether to only write the SLURM script (not submit).
            - wait_for (list[int] or None): SLURM job IDs to wait for before s
                    ubmitting.

    Example:
    --------
        Namespace(
            bookkeeper_config=Path('config.yaml'),
            overwrite_config=True,
            region='lya',
            num_processors=32,
            add_mask=True,
            add_flux=False,
            only_write=False,
            wait_for=[123456, 123457]
        )
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use"
    )

    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Force overwrite bookkeeper config.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="lya",
        help="Region to compute deltas in",
    )

    parser.add_argument(
        "--num-processors",
        type=int,
        help="Num processors to use when parallelizing.",
        default=16,
    )

    parser.add_argument(
        "--add-mask", action="store_true", help="Add extra mask information."
    )

    parser.add_argument(
        "--add-flux",
        action="store_true",
        help="Add flux information",
    )

    parser.add_argument(
        "--only-write", action="store_true", help="Only write scripts, not send them."
    )

    parser.add_argument("--wait-for", nargs="+", type=int,
                        default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
