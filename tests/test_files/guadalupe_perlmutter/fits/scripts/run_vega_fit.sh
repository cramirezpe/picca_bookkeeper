#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name vega_fit
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_fit-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_fit-%j.err
export OMP_NUM_THREADS=1

module load python
conda activate picca
umask 0002



command="run_vega.py /picca_bookkeeper/tests/test_files/output/results/fits/configs/main.ini"
date
srun  $command

date
