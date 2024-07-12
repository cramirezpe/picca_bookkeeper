#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name xcf_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_lyalya-%j.err

module load python
conda activate picca
umask 0002



command="picca_xcf.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --drq /picca_bookkeeper/tests/test_files/dummy_tracer_catalog.fits --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalya/xcf.fits.gz --lambda-abs LYA --mode desi_healpix --nproc 256 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date