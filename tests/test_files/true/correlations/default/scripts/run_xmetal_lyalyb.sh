#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xmetal_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/logs/xmetal_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/logs/xmetal_lyalyb-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/results/lyb/Delta --drq /global/cfs/cdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0/v9.0.0/desi-2.0-1000/zcat.fits --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/results/qso_lyalyb/xmetal.fits.gz --lambda-abs LYA --mode desi_healpix --nproc 128 --rej 0.995 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
