#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/deltas/logs/delta_extraction_lya-%j.err

module load python
conda activate picca_add_tests
umask 0002




echo used picca_bookkeeper version: x.xx
echo using picca version: $(python -c "import importlib.metadata; print(importlib.metadata.version('picca'))")
echo -e '\n'

command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/results/deltas/configs/delta_extraction_lya.ini"
date
srun  $command

date
