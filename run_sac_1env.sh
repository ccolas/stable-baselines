#!/bin/sh
#SBATCH --mincpus 20
#SBATCH -p court_sirocco
#SBATCH -t 4:00:00
#SBATCH -e ./run_sac_1env.sh.err
#SBATCH -o ./run_sac_1env.sh.out
export EXP_INTERP='/cm/shared/apps/intel/composer_xe/python3.5/intelpython3/bin/python3' ;
$EXP_INTERP sac_walker_test.py --logdir sac_walker_1env --n_env 1
wait