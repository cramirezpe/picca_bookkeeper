#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:20:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_exp_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/xcf_exp_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/logs/xcf_exp_lyalyb-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=1

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/results/qso_lyalyb/xcf.fits.gz --out /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/True_0.0.0.0_0/correlations/default/results/qso_lyalyb/xcf_exp.fits.gz --blind-corr-type qsoxlya"
srun --nodes 1 --ntasks 1 --cpus-per-task 1 $command
