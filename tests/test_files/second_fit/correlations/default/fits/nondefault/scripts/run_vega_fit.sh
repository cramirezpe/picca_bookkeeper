#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="run_vega.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/fits/nondefault/configs/main.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
