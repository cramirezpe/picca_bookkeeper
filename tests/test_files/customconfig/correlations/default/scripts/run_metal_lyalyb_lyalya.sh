#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 10:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name metal_lyalyb_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/logs/metal_lyalyb_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/logs/metal_lyalyb_lyalya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=256


command="picca_metal_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/results/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/correlations/default/results/lyalyb_lyalya/metal.fits.gz --lambda-abs LYA --in-dir2 /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0.0_0/results/lya/Delta --nproc 128 --rej 0.995 --abs-igm SiII(1260) SiIII(1207) SiII(1193) SiII(1190) CIV(eff) --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
srun --nodes 1 --ntasks 1 --cpus-per-task 256 $command
