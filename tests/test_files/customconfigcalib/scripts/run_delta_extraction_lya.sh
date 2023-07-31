#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:16:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name delta_extraction_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/logs/delta_extraction_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/logs/delta_extraction_lya-%j.err
#SBATCH --mail-type fail
#SBATCH --mail-user user@host.com

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=256


command="picca_delta_extraction.py /picca_bookkeeper/tests/test_files/output/guadalupe/main/afterburn_v0/dMdB20_2.mgii_r.0.0_0/configs/delta_extraction_lya.ini"
srun --nodes 1 --ntasks 1 --cpus-per-task 256 $command
