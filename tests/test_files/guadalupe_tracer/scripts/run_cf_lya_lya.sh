#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name cf_lya_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/logs/cf_lya_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/logs/cf_lya_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/deltas/lya/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/correlations/lyalya_lyalya/cf.fits.gz --nproc 128 --rp-min 0 --rp-max 300 --rt-max 200 --np 75 --nt 50 --fid-Or 7.975e-05"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
