import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import SubprocVecEnv
import click
import os
#import cProfile

@click.command()
@click.option('--env_name', type=str, default='BipedalWalker-v2', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='sac_default_log/', help='the directory to where data should go.')
@click.option('--exp_name', type=str, default='default', help='the file to where data should go.')
@click.option('--n_envs', type=int, default=1, help='the number of vectorized envs')
@click.option('--buffer_size', type=int, default=1e6, help='buffer size')
@click.option('--eval_freq', type=int, default=100000, help='how often do we perform evaluation')
@click.option('--total_timesteps', type=int, default=100000, help='number of total environment steps')
@click.option('--nb_eval_rollouts', type=int, default=10, help='number of total eval episodes')
@click.option('--batch_size', type=int, default=64, help='number of batches per upgrade')
@click.option('--gpu_id', type=int, default=-1, help='specific gpu to focus computation on')
def main(env_name, logdir, n_envs, buffer_size, eval_freq, total_timesteps, nb_eval_rollouts, exp_name, batch_size, gpu_id):
    print("launching sac with env {}, logs in {}, n_envs {} bufsize {} eval_freq {} total_timesteps {} nb eval rols {} e_name {} btchsize {} gpu {}"
          .format(env_name, logdir, n_envs, buffer_size, eval_freq, total_timesteps, nb_eval_rollouts, exp_name, batch_size, gpu_id))

    if gpu_id != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    #
    # BipedalWalker-v2:
    #   n_timesteps: !!float 1e6
    #   policy: 'CustomSACPolicy'
    #   learning_rate: lin_3e-4
    #   buffer_size: 1000000
    #   batch_size: 64
    #   ent_coef: 0.005
    #   train_freq: 1
    #   gradient_steps: 1
    #   learning_starts: 1000

    # cp = cProfile.Profile()
    # cp.enable()

    env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
    eval_env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(n_envs)])

    model = SAC(MlpPolicy, env, eval_env, verbose=1, learning_starts=1000, tensorboard_log="./sac_walker_tensorboard/",
                replay_buffer=ReplayBuffer(buffer_size), eval_freq=eval_freq, nb_eval_rollouts=nb_eval_rollouts,
                learning_rate=0.0003, batch_size=batch_size, logdir=logdir+exp_name)
    model.learn(total_timesteps=total_timesteps)
    model.save("sac_walker")

    # cp.disable()
    # cp.dump_stats("test_sac_1env.cprof")

    # del model # remove to demonstrate saving and loading
    #
    # n_envs = 1
    # env = DummyVecEnv([lambda: gym.make(env_name)])
    # eval_env = DummyVecEnv([lambda: gym.make(env_name)])
    # model = SAC.load("sac_walker", env=env, eval_env=eval_env)
    #
    #
    # obs = env.reset()
    # r = 0
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     r+=rewards[0]
    #     if dones[0]:
    #         print(r)
    #         r = 0
    #     env.render()

if __name__ == "__main__":
    main()