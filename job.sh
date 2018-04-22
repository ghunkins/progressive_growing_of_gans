#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-00:10:00
#SBATCH --job-name=pgGAN
#SBATCH --mem=30GB 
#SBATCH --output=output_pgGAN_%j.txt
#SBATCH -e error_pgGAN_%j.txt
#SBATCH --gres=gpu:2

source activate pgGAN4
python import_example.py