#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name dmat_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/dmat_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/dmat_lyalyb_lyalya-%j.err

module load python
conda activate picca_add_tests
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalyb_lyalya/dmat.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --nproc 256 --rej 0.99 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3 --z-max-sources 3.79 --no-project "
date
srun  $command

date
