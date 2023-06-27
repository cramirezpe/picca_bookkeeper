#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 30
#SBATCH --constraint haswell
#SBATCH --account desi

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="run_vega.py /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/fits/default/configs/main.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
