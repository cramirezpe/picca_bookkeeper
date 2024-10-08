#!/bin/bash -l

#SBATCH --qos shared
#SBATCH --nodes 1
#SBATCH --time 00:04:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name correct_config_zeff
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/fits/logs/correct_config_zeff-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/fits/logs/correct_config_zeff-%j.err
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5

module load python
conda activate picca
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_bookkeeper_correct_config_zeff /picca_bookkeeper/tests/test_files/output/results/configs/bookkeeper_config.yaml"
date
srun  $command

date
