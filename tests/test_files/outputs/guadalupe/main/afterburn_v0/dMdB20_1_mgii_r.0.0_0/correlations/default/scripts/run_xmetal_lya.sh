#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xmetal_lya
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/outputs/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/xmetal_lya-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/outputs/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/xmetal_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128

export HDF5_USE_FILE_LOCKING=FALSE

command="picca_metal_xdmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/outputs/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/deltas/lya/Delta --drq /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/dummy_catalog.fits --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/outputs/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/correlations/lyalya_qso/xmetal.fits.gz --mode desi_healpix --nproc 128 --rej 0.995 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-05"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command