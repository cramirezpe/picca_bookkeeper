#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name dmat_lya_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/logs/dmat_lya_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/logs/dmat_lya_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_dmat.py --in-dir /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/deltas/lya/Delta --out /picca_bookkeeper/tests/test_files/output/guadalupe/main/dummy_catalog2/dMdB20_1.0.0_0/correlations/lyalya_lyalya/dmat.fits.gz --nproc 32 --rej 0.99"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
