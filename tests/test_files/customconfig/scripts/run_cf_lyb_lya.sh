#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:12:12
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name cf_lyb_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0_0/logs/cf_lyb_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0_0/logs/cf_lyb_lya-%j.err
#SBATCH --mail-type fail
#SBATCH --mail-user user@host.com

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_cf.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0_0/deltas/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0_0/correlations/lyalyb_lyalya/cf.fits.gz --nproc 32 --in-dir2 /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_0.0.0_0/deltas/lya/Delta"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
