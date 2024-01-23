#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:15:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xcf_exp_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/xcf_exp_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/xcf_exp_lyalya-%j.err
#SBATCH --cpus-per-task 1

module load python
source activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/results/qso_lyalya/xcf.fits.gz --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/results/qso_lyalya/xcf_exp.fits.gz --blind-corr-type qsoxlya"
date
srun  $command

date