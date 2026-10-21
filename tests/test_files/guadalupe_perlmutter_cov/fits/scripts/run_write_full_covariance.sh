#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name write_full_covariance
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/write_full_covariance-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/write_full_covariance-%j.err
export OMP_NUM_THREADS=1

module load python
conda activate picca
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python3 -m picca_bookkeeper.get_version picca)
echo -e '\n'

command="picca_write_full_covariance.py --correlations /picca_bookkeeper/tests/test_files/output/results/correlations/results/lyalya_lyalya/cf.fits.gz /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalya/xcf.fits.gz --output /picca_bookkeeper/tests/test_files/output/results/fits/results/full-covariance.fits"
date
srun --nodes 1 $command

date
