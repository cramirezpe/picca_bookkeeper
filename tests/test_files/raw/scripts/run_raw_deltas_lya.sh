#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 03:00:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name raw_deltas_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/raw/LyaCoLoRe/raw_0_0.0.0_0/logs/raw_deltas_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/raw/LyaCoLoRe/raw_0_0.0.0_0/logs/raw_deltas_lya-%j.err

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=128


command="picca_convert_transmission.py --object-cat /global/cfs/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0/master.fits --in-dir /global/cfs/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0 --out-dir /picca_bookkeeper/tests/test_files/output/v9.0.0/raw/LyaCoLoRe/raw_0_0.0.0_0/deltas/lya/Delta --lambda-rest-min 1040.0 --lambda-rest-max 1200.0"
srun --nodes 1 --ntasks 1 --cpus-per-task 128 $command
