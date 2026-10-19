#!/bin/bash -l

#SBATCH --qos shared
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name vega_fit
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_fit-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_fit-%j.err
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8

module load python
conda activate picca_add_tests
umask 0002




echo used picca_bookkeeper version: x.xx
echo using vega version: $(python3 -m picca_bookkeeper.get_version vega)
echo -e '\n'

command="run_vega.py /picca_bookkeeper/tests/test_files/output/results/fits/configs/main.ini"
date
srun --nodes 1 --ntasks 1 $command

date
