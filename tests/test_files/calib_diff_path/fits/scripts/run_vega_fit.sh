#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name vega_fit
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/calib_diff/fits/logs/vega_fit-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/calib_diff/fits/logs/vega_fit-%j.err

module load python
source activate picca
umask 0002



command="run_vega.py /picca_bookkeeper/tests/test_files/output/results/calib_diff/fits/configs/main.ini"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
