#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name xmetal_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xmetal_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xmetal_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/deltas/lya/Delta --drq /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/catalogs/dummy_catalog.fits --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/correlations/lyalya_qso/xmetal.fits.gz --mode desi_healpix --nproc 32 --rej 0.9999 --abs-igm CIV(eff) SiII(1260) SiIII(1207) SiII(1193) SiII(1190)"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
