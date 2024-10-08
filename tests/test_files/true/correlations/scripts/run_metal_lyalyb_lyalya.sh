#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name metal_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/metal_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/metal_lyalyb_lyalya-%j.err

module load python
conda activate picca_add_tests
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_metal_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lyb/Delta --in-dir2 /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalyb_lyalya/metal.fits --lambda-abs LYA --nproc 256 --rej 0.999 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --rp-min 0 --rp-max 200 --rt-max 200 --np 50 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3 --z-max-sources 3.79"
date
srun  $command

date
