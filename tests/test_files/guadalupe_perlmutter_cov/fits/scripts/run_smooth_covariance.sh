#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name smooth_covariance
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/smooth_covariance-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/smooth_covariance-%j.err
export OMP_NUM_THREADS=1

module load python
conda activate picca
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="/global/cfs/cdirs/desicollab/science/lya/y1-kp6/iron-tests/correlations/scripts/write_smooth_covariance_flex_size.py --rt-max-auto 200 --rp-min-auto 0 --rp-max-auto 300 --np-auto 75 --nt-auto 50 --rt-max-cross 200 --rp-min-cross -300 --rp-max-cross 300 --np-cross 150 --nt-cross 50 --input-cov /picca_bookkeeper/tests/test_files/output/results/fits/results/full-covariance.fits --output-cov /picca_bookkeeper/tests/test_files/output/results/fits/results/full-covariance-smoothed.fits"
date
srun  $command

date
