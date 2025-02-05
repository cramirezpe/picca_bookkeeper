#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name xdmat_qso_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/calib_diff/correlations/logs/xdmat_qso_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/calib_diff/correlations/logs/xdmat_qso_lyalya-%j.err

module load python
conda activate picca
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/calib_diff/deltas/results/lya/Delta --drq /picca_bookkeeper/tests/test_files/dummy_catalog.fits --out /picca_bookkeeper/tests/test_files/output/results/calib_diff/correlations/results/qso_lyalya/xdmat.fits.gz --lambda-abs LYA --mode desi_healpix --nproc 256 --rej 0.99 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
