#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_exp_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/logs/xcf_exp_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/logs/xcf_exp_lyalya-%j.err

module load python
source activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/results/qso_lyalya/xcf.fits.gz --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/results/qso_lyalya/xcf_exp.fits.gz --blind-corr-type qsoxlya"
srun --nodes 1 --ntasks 1 $command
