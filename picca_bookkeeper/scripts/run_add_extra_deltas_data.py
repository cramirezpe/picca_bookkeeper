""" Script to run add_extra_deltas_data
given a bookkeeper config file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.tasker import get_Tasker

if TYPE_CHECKING:
    from typing import Optional


def main(args: Optional[argparse.Namespace] = None) -> None:
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

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
