#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:16:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name delta_extraction_mgii_r_calib_step_2
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_mgii_r_calib_step_2-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_mgii_r_calib_step_2-%j.err

module load python
conda activate picca
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/results/deltas/configs/delta_extraction_mgii_r_calib_step_2.ini"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
