#!/bin/bash
#
#SBATCH --job-name=preprocess
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16000
#SBATCH --output=out/preprocess.out
#SBATCH --error=out/preprocess.out

source activate xsl
python -u preprocess.py

