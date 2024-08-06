#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12                   # 12 cores (12 cores per GPU)
#$ -l h_rt=6:0:0                # runtime
#$ -l h_vmem=7.5G               # 7.5 * 12 = 90G total RAM
#$ -l gpu=1                     # request 1 GPU
#$ -l cluster=andrena           # use the Andrena nodes
#$ -m bea                       # Send email at the beginning and end of the job and if aborted
#$ -M haokun.tian@qmul.ac.uk    # The email address to notify

source ~/.bashrc
conda activate label_metric
python label_metric/lightning_modules.py
