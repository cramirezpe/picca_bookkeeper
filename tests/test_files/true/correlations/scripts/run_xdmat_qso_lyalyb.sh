#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xdmat_qso_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xdmat_qso_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xdmat_qso_lyalyb-%j.err

module load python
conda activate picca_add_tests
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --drq /tmp/cdirs/desicollab/users/hiramk/desi/qq_mocks/iron/main/mock/london/v9.0.0/iron_main-0.134/qsocat_iron_main.fits --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalyb/xdmat.fits.gz --lambda-abs LYA --nproc 256 --rej 0.99 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3 --z-max-sources 3.79 --mode desi_mocks"
date
srun  $command

date
