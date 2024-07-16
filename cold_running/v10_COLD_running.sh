#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --array=1-100
#SBATCH --time=12:00:00  

module purge
module load slurm/slurm/22.05.9
module load gcc/11.3.0 gsl/2.7 zlib/1.2.12 libffi/3.2.1 
module load gdal/3.6.0

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/LCM_diversity/pythoncode/COLD/

python3 v10_COLD_running.py   --n_cores=$SLURM_ARRAY_TASK_MAX  --rank=$SLURM_ARRAY_TASK_ID  --output_foldername='COLD_output_morechange'  --rootpath_scratch='/scratch/zhz18039/fah20002/LCM_diversity'
