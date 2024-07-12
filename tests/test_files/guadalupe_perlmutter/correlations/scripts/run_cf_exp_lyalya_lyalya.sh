#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cf_exp_lyalya_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_exp_lyalya_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_exp_lyalya_lyalya-%j.err
export OMP_NUM_THREADS=1

module load python
conda activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalya/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalya/cf_exp.fits.gz --blind-corr-type lyaxlya"
date
srun  $command

date
