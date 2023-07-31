#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name cf_exp_lyalya_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs/correlations/default/logs/cf_exp_lyalya_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs/correlations/default/logs/cf_exp_lyalya_lyalyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=1

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs/correlations/default/results/lyalya_lyalyb/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_copy_corrs/correlations/default/results/lyalya_lyalyb/cf_exp.fits.gz --blind-corr-type lyaxlya"
srun --nodes 1 --ntasks 1 --cpus-per-task 1 $command
