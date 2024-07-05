#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:40:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xdmat_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xdmat_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xdmat_lyalyb-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002



command="picca_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --drq /picca_bookkeeper/tests/test_files/dummy_catalog-bal.fits --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalyb/xdmat.fits.gz --lambda-abs LYA --mode desi_healpix --rej 0.99 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.963e-05 --fid-Om 0.315 --rebin-factor 3 --coef-binning-model 2"
date
srun  $command

date
