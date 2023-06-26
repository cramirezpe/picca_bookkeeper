#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_exp_lyalyb
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/logs/xcf_exp_lyalyb-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/logs/xcf_exp_lyalyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=1

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_export.py --data /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/correlations/lyalyalyb_qso/xcf.fits.gz --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/correlations/lyalyalyb_qso/xcf_exp.fits.gz --blind-corr-type qsoxlya ----lambda-abs LYA --dmat /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/correlations/lyalyalyb_qso/xdmat.fits.gz"
srun --nodes 1 --ntasks 1 --cpus-per-task 1 $command
