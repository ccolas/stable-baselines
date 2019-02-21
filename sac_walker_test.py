import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import SubprocVecEnv
import click
#import cProfile

@click.command()
@click.option('--env', type=str, default='BipedalWalker-v2', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='sac_default_log/', help='the directory to where data should go.')
@click.option('--exp_name', type=str, default='default', help='the file to where data should go.')
@click.option('--n_env', type=int, default=1, help='the number of vectorized envs')
@click.option('--buffer_size', type=int, default=1e6, help='buffer size')
@click.option('--eval_freq', type=int, default=100000, help='how often do we perform evaluation')
@click.option('--total_timesteps', type=int, default=100000, help='number of total environment steps')
@click.option('--nb_eval_rollouts', type=int, default=10, help='number of total eval episodes')
@click.option('--batch_size', type=int, default=64, help='number of batches per upgrade')
def main(env, logdir, n_env, buffer_size, eval_freq, total_timesteps, nb_eval_rollouts, exp_name, batch_size):
    print("launching sac with env {}, logs in {}, n_env {} bufsize {} eval_freq {} total_timesteps {} nb eval rols {} e_name {} btchsize {}"
          .format(env, logdir, n_env, buffer_size, eval_freq, total_timesteps, nb_eval_rollouts, exp_name, batch_size))
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

    env = SubprocVecEnv([lambda: gym.make('BipedalWalker-v2') for _ in range(n_env)])
    eval_env = SubprocVecEnv([lambda: gym.make('BipedalWalker-v2') for _ in range(n_env)])

    model = SAC(MlpPolicy, env, eval_env, verbose=1, learning_starts=1000, tensorboard_log="./sac_walker_tensorboard/",
                replay_buffer=ReplayBuffer(buffer_size), eval_freq=eval_freq, nb_eval_rollouts=nb_eval_rollouts,
                learning_rate=0.0003, batch_size=batch_size, logdir=logdir+exp_name)
    model.learn(total_timesteps=total_timesteps)
    model.save("sac_walker")

    # cp.disable()
    # cp.dump_stats("test_sac_1env.cprof")

    # del model # remove to demonstrate saving and loading
    #
    # n_env = 1
    # env = DummyVecEnv([lambda: gym.make('BipedalWalker-v2')])
    # eval_env = DummyVecEnv([lambda: gym.make('BipedalWalker-v2')])
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