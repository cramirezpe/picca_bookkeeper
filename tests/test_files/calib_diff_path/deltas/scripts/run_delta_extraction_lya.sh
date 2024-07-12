#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/calib_diff/deltas/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/calib_diff/deltas/logs/delta_extraction_lya-%j.err

module load python
conda activate picca
umask 0002



command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/results/calib_diff/deltas/configs/delta_extraction_lya.ini"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date