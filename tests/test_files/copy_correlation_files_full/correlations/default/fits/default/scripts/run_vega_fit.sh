#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name vega_fit
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs_full/correlations/default/fits/default/logs/vega_fit-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs_full/correlations/default/fits/default/logs/vega_fit-%j.err

module load python
source activate picca
umask 0002



command="run_vega.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs_full/correlations/default/fits/default/configs/main.ini"
date
srun  $command

date