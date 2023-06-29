#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name cf_lyalya_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/cf_lyalya_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/logs/cf_lyalya_lyalya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/deltas/lya/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1_mgii_r.0.0_0/correlations/default/correlations/lyalya_lyalya/cf.fits.gz --lambda-abs LYA --nproc 128 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-5 --rebin-factor 3"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
