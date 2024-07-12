#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name cf_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_lyalyb_lyalya-%j.err

module load python
conda activate picca
umask 0002



command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalyb_lyalya/cf.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --nproc 256 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
