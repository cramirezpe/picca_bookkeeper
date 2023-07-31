#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name delta_extraction_lyb
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/logs/delta_extraction_lyb-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/logs/delta_extraction_lyb-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=256


command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0__reuse_calib/configs/delta_extraction_lyb.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 256 $command
