#!/usr/bin/python
# -*- coding: utf-8 -*-
PATH_TO_INTERPRETER = "~/anaconda3/envs/tfGPU/bin/python"  # plafrim
filename = "run_sac_walker_1_env.sh"
save_dir = '.'
trial_id = list(range(0, 20))

with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH -p court_sirocco \n')
    f.write('#SBATCH -t 4:00:00\n')
    f.write('#SBATCH -e ' + filename[1:] + '.err\n')
    f.write('#SBATCH -o ' + filename[1:] + '.out\n')
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)

    for seed in range(len(trial_id)):
        t_id = trial_id[seed]
        f.write("$EXP_INTERP sac_walker_test.py --logdir sac_walker_1_env/ --exp_name %s --n_env 1 --buffer_size 10000 --eval_freq 100 --total_timesteps 1000 --nb_eval_rollouts 10 &\n" % (t_id))
    f.write('wait')
