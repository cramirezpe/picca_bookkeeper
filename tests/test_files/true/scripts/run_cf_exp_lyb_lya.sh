#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name cf_exp_lyb_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0.0.0_0/logs/cf_exp_lyb_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0.0.0_0/logs/cf_exp_lyb_lya-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=32

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0.0.0_0/correlations/lyalyb_lyalya/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0.0.0_0/correlations/lyalyb_lyalya/cf_exp.fits.gz --dmat /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0.0.0_0/correlations/lyalyb_lyalya/dmat.fits.gz"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
