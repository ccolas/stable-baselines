import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import SubprocVecEnv
import click
#import cProfile
import time

@click.command()
@click.option('--env_name', type=str, default='BipedalWalker-v2', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--n_envs', type=int, default=1, help='the number of vectorized envs')

def main(env_name, n_envs):
    env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
    obs = env.reset()
    r = 0
    nb_steps = 0
    start = time.time()
    while True:
        actions = [env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, info = env.step(actions)
        nb_steps += n_envs
        if nb_steps % 10000 == 0:
            end = time.time()
            print("{}: {}".format(nb_steps, end-start))
            start = time.time()

if __name__ == "__main__":
    main()