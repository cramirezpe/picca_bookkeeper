#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --job-name raw_deltas_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/raw_0.0.0.0_0/logs/raw_deltas_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/raw_0.0.0.0_0/logs/raw_deltas_lya-%j.err
#SBATCH --mail-type fail
#SBATCH --mail-user mail@host.es

module load python
source activate picca
umask 0002
export OMP_NUM_THREADS=256


command="picca_convert_transmission.py --object-cat /global/cfs/cdirs/desi/mocks/lya_forest/develop/london/qq_desi/v9.0/v9.0.0/desi-2.0-1000/zcat.fits --in-dir /global/cfs/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0 --out-dir /picca_bookkeeper/tests/test_files/output/v9.0.0/desi-2.0-1000/LyaCoLoRe/raw_0.0.0.0_0/results/lya/Delta --lambda-rest-min 1040.0 --lambda-rest-max 1200.0 --linear-spacing True --lambda-min 3600 --lambda-max 5500 --delta-lambda 0.8"
srun --nodes 1 --ntasks 1 --cpus-per-task 256 $command
