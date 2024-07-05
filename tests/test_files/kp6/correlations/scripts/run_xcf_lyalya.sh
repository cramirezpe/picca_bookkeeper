#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:40:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xcf_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_lyalya-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002



command="picca_xcf.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --drq /picca_bookkeeper/tests/test_files/dummy_catalog-bal.fits --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalya/xcf.fits.gz --lambda-abs LYA --mode desi_healpix --nside 16 --rp-min -200 --rp-max 200 --rt-max 200 --np 100 --nt 50 --fid-Or 7.963e-05 --fid-Om 0.315 --rebin-factor 3"
date
srun  $command

date