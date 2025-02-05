#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --cpus-per-task 128
#SBATCH --job-name xcf_exp_qso_lyalyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_exp_qso_lyalyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_exp_qso_lyalyb-%j.err

module load python
conda activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE



echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalyb/xcf.fits.gz --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalyb/xcf_exp.fits.gz --blind-corr-type qsoxlyb"
date
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command

date
