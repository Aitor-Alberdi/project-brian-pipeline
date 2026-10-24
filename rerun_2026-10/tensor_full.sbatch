#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --account=brian
#SBATCH --nodelist=ori
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=180G
#SBATCH --time=6-00:00:00
#SBATCH --job-name=tensorProto
#SBATCH --output=/opt/nfs/neuroinformatics/BRIAN/tensorProto_2026-10/slurm_full_%j.out
source /opt/conda/etc/profile.d/conda.sh
conda activate simulation
cd /opt/nfs/neuroinformatics/BRIAN/tensorProto_2026-10
python scripts/cluster_tensor.py all --workers 30 --cpus 2
