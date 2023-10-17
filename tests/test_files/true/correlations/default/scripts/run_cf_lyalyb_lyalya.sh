#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name cf_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/cf_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/cf_lyalyb_lyalya-%j.err

module load python
source activate picca_add_tests
umask 0002



command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/results/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/results/lyalyb_lyalya/cf.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/results/lya/Delta --nproc 256 --rp-min 0 --rp-max 200 --rt-max 200 --np 50 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3 --z-max-sources 3.79 --no-project "
srun  $command
