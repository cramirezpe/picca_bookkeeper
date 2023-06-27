#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/logs/delta_extraction_lya-%j.err

module load python
source activate picca_add_tests
umask 0002
export OMP_NUM_THREADS=128


command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/true_0_0.0.0_0/configs/delta_extraction_lya.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
