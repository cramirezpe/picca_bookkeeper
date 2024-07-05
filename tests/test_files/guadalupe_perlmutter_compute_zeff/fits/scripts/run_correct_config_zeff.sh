#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name correct_config_zeff
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/correct_config_zeff-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/correct_config_zeff-%j.err

module load python
source activate picca
umask 0002



command="picca_bookkeeper_correct_config_zeff /picca_bookkeeper/tests/test_files/output/results/configs/bookkeeper_config.yaml"
date
srun  $command

date
