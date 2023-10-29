#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cf_exp_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/logs/cf_exp_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/logs/cf_exp_lyalyb_lyalya-%j.err
#SBATCH --mail-type fail
#SBATCH --mail-user user@host.com

module load python
source activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/results/lyalyb_lyalya/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/results/lyalyb_lyalya/cf_exp.fits.gz --blind-corr-type lyaxlyb --smooth-per-r-par "
srun  $command
