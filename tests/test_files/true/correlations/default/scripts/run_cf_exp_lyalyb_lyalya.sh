#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name cf_exp_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/logs/cf_exp_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/logs/cf_exp_lyalyb_lyalya-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/results/lyalyb_lyalya/cf.fits.gz --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/results/lyalyb_lyalya/cf_exp.fits.gz --dmat /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0_0.0.0_0/correlations/default/results/lyalyb_lyalya/dmat.fits.gz"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
