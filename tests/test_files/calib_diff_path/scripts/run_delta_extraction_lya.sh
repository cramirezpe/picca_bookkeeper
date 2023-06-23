#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output2/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output2/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/delta_extraction_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output2/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/configs/delta_extraction_lya.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command