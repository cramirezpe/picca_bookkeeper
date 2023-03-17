#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:00:00
#SBATCH --constraint haswell
#SBATCH --account desi
#SBATCH --job-name delta_extraction_mgii_calib_step_2
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/delta_extraction_mgii_calib_step_2-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/logs/delta_extraction_mgii_calib_step_2-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=32


command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_1.0.0_0/configs/delta_extraction_mgii_calib_step_2.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 32 $command
