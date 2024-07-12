#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name delta_extraction_lyb
#SBATCH --output /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/deltas/logs/delta_extraction_lyb-%j.out
#SBATCH --error /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/deltas/logs/delta_extraction_lyb-%j.err

module load python
conda activate picca
umask 0002



command="picca_delta_extraction.py /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/deltas/configs/delta_extraction_lyb.ini"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
