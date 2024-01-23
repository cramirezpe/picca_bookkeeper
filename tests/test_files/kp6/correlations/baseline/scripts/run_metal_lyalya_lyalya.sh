#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:20:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name metal_lyalya_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/metal_lyalya_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/metal_lyalya_lyalya-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002



command="picca_fast_metal_dmat.py --in-attributes /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lya/Log/delta_attributes.fits.gz --delta-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lya/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/results/lyalya_lyalya/metal.fits --lambda-abs LYA --abs-igm SiII(1190) SiII(1193) SiIII(1207) SiII(1260) CIV(eff) --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.963e-05 --fid-Om 0.315 --rebin-factor 3 --coef-binning-model 2"
date
srun  $command

date