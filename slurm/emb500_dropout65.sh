#!/bin/bash 



#SBATCH --output=model_em500_drop65.out

#SBATCH -t 0-3:30 

#SBATCH --mail-type=ALL

#SBATCH --mail-user=neb330@nyu.edu



module load pytorch/intel/20170226



python main_dropout.py --epochs 6 --emsize 500 --nhid 500 --nlayers 2 --log-interval 1000