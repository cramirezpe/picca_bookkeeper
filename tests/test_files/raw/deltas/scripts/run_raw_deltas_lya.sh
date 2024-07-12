#!/bin/bash -l

#SBATCH --qos regular
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --constraint cpu
#SBATCH --account desi
#SBATCH --ntasks-per-node 1
#SBATCH --job-name raw_deltas_lya
#SBATCH --output /picca_bookkeeper/tests/test_files/output/results/deltas/logs/raw_deltas_lya-%j.out
#SBATCH --error /picca_bookkeeper/tests/test_files/output/results/deltas/logs/raw_deltas_lya-%j.err
#SBATCH --mail-type fail
#SBATCH --mail-user user@host.com

module load python
conda activate /global/common/software/desi/users/cramirez/conda/envs/durham2023
umask 0002



command="picca_convert_transmission.py --object-cat /tmp/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0/master.fits --in-dir /tmp/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0 --out-dir /picca_bookkeeper/tests/test_files/output/results/deltas/results/lya/Delta --lambda-rest-min 1040.0 --lambda-rest-max 1200.0 --lambda-min 3600 --lambda-max 5500 --delta-lambda 0.8 --linear-spacing True"
date
srun  $command

date
