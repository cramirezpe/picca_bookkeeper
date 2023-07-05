#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128


command="run_vega.py /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/fits/default/configs/main.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
