#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xmetal_lyalyb
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0__reuse_calib/correlations/default/logs/xmetal_lyalyb-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0__reuse_calib/correlations/default/logs/xmetal_lyalyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0__reuse_calib/deltas/lyb/Delta --drq /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/dummy_catalog.fits --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0__reuse_calib/correlations/default/correlations/lyalyalyb_qso/xmetal.fits.gz --lambda-abs LYA"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
