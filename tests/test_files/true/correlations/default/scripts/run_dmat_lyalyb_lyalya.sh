#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name dmat_lyalyb_lyalya
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/dmat_lyalyb_lyalya-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/dmat_lyalyb_lyalya-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128


command="picca_dmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/deltas/lyb/Delta --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/correlations/lyalyb_lyalya/dmat.fits.gz ----lambda-abs LYA --mode desi_mocks --in-dir2 /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/deltas/lya/Delta --nproc 128 --rej 0.99 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-5"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
