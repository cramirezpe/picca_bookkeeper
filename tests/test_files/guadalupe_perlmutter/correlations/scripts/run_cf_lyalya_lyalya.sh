#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cf_lyalya_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_lyalya_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/cf_lyalya_lyalya-%j.err
export OMP_NUM_THREADS=1

module load python
source activate picca
umask 0002



command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalya/cf.fits.gz --lambda-abs LYA --nproc 256 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun  $command

date