#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name xdmat_lyalyb
#SBATCH --output /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/logs/xdmat_lyalyb-%j.out
#SBATCH --error /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/logs/xdmat_lyalyb-%j.err

module load python
conda activate picca
umask 0002



command="picca_xdmat.py --in-dir /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/deltas/results/lyb/Delta --drq /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/dummy_catalog.fits --out /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/results/qso_lyalyb/xdmat.fits.gz --lambda-abs LYA --mode desi_healpix --nproc 256 --rej 0.99 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
