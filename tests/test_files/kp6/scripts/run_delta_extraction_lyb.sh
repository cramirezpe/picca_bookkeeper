#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 01:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name delta_extraction_lyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/logs/delta_extraction_lyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/logs/delta_extraction_lyb-%j.err
#SBATCH --cpus-per-task 256

module load python
source activate picca
umask 0002



command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/Y1v4_1.ciii.4.2_0/configs/delta_extraction_lyb.ini"
date
srun  $command

date