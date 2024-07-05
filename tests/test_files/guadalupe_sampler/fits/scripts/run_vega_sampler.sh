#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 2
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 192
#SBATCH --job-name vega_sampler
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_sampler-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/vega_sampler-%j.err

module load python
source /global/common/software/desi/users/acuceu/stable_vega/activate_vega.sh
umask 0002



command="run_vega_mpi.py /picca_bookkeeper/tests/test_files/output/results/fits/configs/main.ini"
date
srun  $command

date