#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xmetal_lyb
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/xmetal_lyb-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/xmetal_lyb-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/deltas/lyb/Delta --drq /global/cfs/cdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0/v9.0.0/desi-2.0-1000/zcat.fits --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/correlations/lyalyb_qso/xmetal.fits.gz --mode desi_mocks"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
