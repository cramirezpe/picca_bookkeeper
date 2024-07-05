#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name dmat_lyalya_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/dmat_lyalya_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/dmat_lyalya_lyalyb-%j.err

module load python
source activate picca
umask 0002



command="picca_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalyb/dmat.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --nproc 256 --rej 0.99 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
