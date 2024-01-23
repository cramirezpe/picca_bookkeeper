#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:20:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cf_lyalya_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/cf_lyalya_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/logs/cf_lyalya_lyalyb-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002



command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lya/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/correlations/baseline/results/lyalya_lyalyb/cf.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/results/lyb/Delta --rp-min 0 --rp-max 200 --rt-max 200 --np 50 --nt 50 --fid-Or 7.963e-05 --fid-Om 0.315 --rebin-factor 3"
date
srun  $command

date