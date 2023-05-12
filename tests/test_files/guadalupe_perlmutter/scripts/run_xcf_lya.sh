#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xcf_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/xcf_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="picca_xcf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/deltas/lya/Delta --drq /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/catalogs/dummy_catalog.fits --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/correlations/lyalya_qso/xcf.fits.gz --mode desi_healpix --nproc 128 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
