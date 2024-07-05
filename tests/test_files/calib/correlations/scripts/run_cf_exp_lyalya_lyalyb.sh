#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name cf_exp_lyalya_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_exp_lyalya_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_exp_lyalya_lyalyb-%j.err

module load python
source activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalyb/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalyb/cf_exp.fits.gz --blind-corr-type lyaxlyb --smooth-per-r-par "
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date