#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name metal_lyb_lya
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0_mgii_r.0.0_0/correlations/default/logs/metal_lyb_lya-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0_mgii_r.0.0_0/correlations/default/logs/metal_lyb_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="picca_metal_dmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0_mgii_r.0.0_0/deltas/lyb/Delta --out /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0_mgii_r.0.0_0/correlations/default/correlations/lyalyb_lyalya/metal.fits.gz --lambda-abs LYA --in-dir2 /global/u2/c/cramirez/Codes/picca_bookkeeper_new/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0_mgii_r.0.0_0/deltas/lya/Delta --nproc 128 --rej 0.995 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-5"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
