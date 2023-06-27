#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 30
#SBATCH --constraint cpu
#SBATCH --account desi

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="run_vega.py /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/fits/default/configs/main.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
