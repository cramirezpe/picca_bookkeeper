#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name xcf_lyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xcf_lyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xcf_lyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_xcf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/deltas/lyb/Delta --drq /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/catalogs/dummy_catalog.fits --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/correlations/lyalyb_qso/xcf.fits.gz --mode desi_healpix --nproc 32"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
