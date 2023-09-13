#!/bin/sh
#SBATCH --partition=priority
#SBATCH --account=zhz18039
#SBATCH --array=3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00     

module purge
module load slurm/slurm/22.05.9 cuda/11.6
module load gcc/11.3.0 gsl/2.7 zlib/1.2.12 libffi/3.2.1 
module load gdal/3.6.0

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38   # replace with your own conda environment

cd /gpfs/scratchfs1/zhz18039/fah20002/pycold_running/pythoncode/

python3 landsat_download.py  --i_collect=$SLURM_ARRAY_TASK_ID  --path_and_row='010047' --start_date='2023-03-01'  --end_date='2023-04-01'