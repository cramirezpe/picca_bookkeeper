#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:15:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name xcf_exp_qso_lyalya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_exp_qso_lyalya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/correlations/logs/xcf_exp_qso_lyalya-%j.err
#SBATCH --cpus-per-task 1

module load python
conda activate picca
umask 0002

export HDF5_USE_FILE_LOCKING=FALSE



echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_export.py --data /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalya/xcf.fits.gz --out /picca_bookkeeper/tests/test_files/output/results/correlations/results/qso_lyalya/xcf_exp.fits.gz --blind-corr-type qsoxlya"
date
srun  $command

date
