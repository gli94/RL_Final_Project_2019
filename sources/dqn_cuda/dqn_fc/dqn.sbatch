#!/bin/bash
#SBATCH -J deep_q_learning           # job name
#SBATCH -o deep_q_learning.o%j       # expands to jobID
#SBATCH -N 1                # total number of node
#SBATCH -n 1                # total number of mpi tasks requested
#SBATCH -p gtx              # queue (partition) — normal, development, etc.
#SBATCH -t 24:00:00         # run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=lige@utexas.edu
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes


module load intel/17.0.4 python3/3.6.3
module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7

pip3 install --user torch torchvision
pip3 install --user gym
pip3 install --user gym[atari]
pip3 install --user numpy
pip3 install --user matplotlib
pip3 install --user scikit-image

python3 deep_q_learning_ram.py
