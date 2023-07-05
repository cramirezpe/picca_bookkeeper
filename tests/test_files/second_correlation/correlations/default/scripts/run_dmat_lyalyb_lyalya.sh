#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name dmat_lyalyb_lyalya
#SBATCH --output /global/u2/c/cramirez/Codes/picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/logs/dmat_lyalyb_lyalya-%j.out
#SBATCH --error /global/u2/c/cramirez/Codes/picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/logs/dmat_lyalyb_lyalya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_dmat.py --in-dir /global/u2/c/cramirez/Codes/picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/results/lyb/Delta --out /global/u2/c/cramirez/Codes/picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/correlations/default/results/lyalyb_lyalya/dmat.fits.gz --lambda-abs LYA --in-dir2 /global/u2/c/cramirez/Codes/picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/results/lya/Delta --nproc 128 --rej 0.99 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05 --rebin-factor 3"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
