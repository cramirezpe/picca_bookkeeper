#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_exp_lyb
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/xcf_exp_lyb-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/logs/xcf_exp_lyb-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=1

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/correlations/lyalyb_qso/xcf.fits.gz --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/correlations/lyalyb_qso/xcf_exp.fits.gz --blind-corr-type qsoxlya --dmat /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/correlations/default/correlations/lyalyb_qso/xdmat.fits.gz"
srun --nodes 1 --ntasks 1 --cpus-per-task 1 $command
