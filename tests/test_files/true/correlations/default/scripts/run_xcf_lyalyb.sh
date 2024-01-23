#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xcf_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/xcf_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/xcf_lyalyb-%j.err

module load python
source activate picca_add_tests
umask 0002



command="picca_xcf.py --in-dir /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/results/lyb/Delta --drq /global/cfs/cdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0/v9.0.0/desi-2.0-1000/zcat.fits --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/results/qso_lyalyb/xcf.fits.gz --lambda-abs LYA --nproc 256 --nside 16 --rp-min -200 --rp-max 200 --rt-max 200 --np 100 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3 --z-max-sources 3.79 --mode desi_mocks --no-project  --no-remove-mean-lambda-obs "
date
srun  $command

date