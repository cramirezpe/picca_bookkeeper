#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name metal_lyb_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.1.2_0/logs/metal_lyb_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.1.2_0/logs/metal_lyb_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_metal_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.1.2_0/deltas/lyb/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.1.2_0/correlations/lyalyb_lyalya/metal.fits.gz --nproc 32 --rej 0.9999 --abs-igm CIV(eff) SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --in-dir2 /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.1.2_0/deltas/lya/Delta"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
