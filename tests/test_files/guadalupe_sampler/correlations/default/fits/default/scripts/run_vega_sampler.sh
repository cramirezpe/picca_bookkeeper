#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 2
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name vega_sampler
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/fits/default/logs/vega_sampler-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/fits/default/logs/vega_sampler-%j.err
#SBATCH --ntasks-per-node 192
export OMP_NUM_THREADS=1

module load python
source activate /global/common/software/desi/users/acuceu/stable_vega/activate_vega.sh
umask 0002


command="run_vega_mpi.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/fits/default/configs/main.ini"
srun  $command
