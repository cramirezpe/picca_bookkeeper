#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name xmetal_lyalyb
#SBATCH --output /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/logs/xmetal_lyalyb-%j.out
#SBATCH --error /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/logs/xmetal_lyalyb-%j.err

module load python
conda activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE


command="picca_metal_xdmat.py --in-dir /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/deltas/results/lyb/Delta --drq /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/dummy_catalog.fits --out /global/common/software/desi/users/cramirez/picca_bookkeeper_simple/tests/test_files/output/results/correlations/results/qso_lyalyb/xmetal.fits --lambda-abs LYA --mode desi_healpix --nproc 256 --rej 0.995 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
