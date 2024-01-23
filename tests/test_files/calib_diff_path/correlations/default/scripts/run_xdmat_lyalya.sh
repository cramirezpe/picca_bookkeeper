#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xdmat_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/logs/xdmat_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/logs/xdmat_lyalya-%j.err

module load python
source activate picca
umask 0002



command="picca_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/results/lya/Delta --drq /picca_bookkeeper/tests/test_files/dummy_catalog.fits --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/correlations/default/results/qso_lyalya/xdmat.fits.gz --lambda-abs LYA --mode desi_healpix --nproc 256 --rej 0.99 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun  $command

date