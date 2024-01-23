#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xmetal_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/xmetal_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/xmetal_lyalyb-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_fast_metal_xdmat.py --in-attributes /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lyb/Log/delta_attributes.fits.gz --delta-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lyb/Delta --drq /picca_bookkeeper/tests/test_files/dummy_catalog-bal.fits --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/results/qso_lyalyb/xmetal.fits --lambda-abs LYA --mode desi_healpix --abs-igm SiII(1190) SiII(1193) SiIII(1207) SiII(1260) --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.963e-05 --fid-Om 0.315 --rebin-factor 3 --coef-binning-model 2"
date
srun  $command

date