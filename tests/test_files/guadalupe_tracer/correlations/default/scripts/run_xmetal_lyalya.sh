#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name xmetal_lyalya
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/xmetal_lyalya-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/xmetal_lyalya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/deltas/lya/Delta --drq /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/dummy_tracer_catalog.fits --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/correlations/lyalyalya_qso/xmetal.fits.gz ----lambda-abs LYA"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command