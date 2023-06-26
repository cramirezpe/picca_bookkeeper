#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name xcf_lyalyb
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/logs/xcf_lyalyb-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/logs/xcf_lyalyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="picca_xcf.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/deltas/lyb/Delta --drq /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/dummy_catalog-bal.fits --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.1.2_0/correlations/default/correlations/lyalyalyb_qso/xcf.fits.gz ----lambda-abs LYA --mode desi_healpix --nproc 128 --nside 16 --rp-min -300 --rp-max 300 --rt-max 200 --np 150 --nt 50 --fid-Or 7.975e-5"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
