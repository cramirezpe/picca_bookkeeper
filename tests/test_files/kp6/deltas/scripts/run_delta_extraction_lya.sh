#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_lya-%j.err
#SBATCH --cpus-per-task 256

module load python
conda activate picca
umask 0002



command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/results/deltas/configs/delta_extraction_lya.ini"
date
srun  $command

date
