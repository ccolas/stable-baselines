import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import SubprocVecEnv
import cProfile
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

cp = cProfile.Profile()
cp.enable()

n_env = 1
env = SubprocVecEnv([lambda: gym.make('BipedalWalker-v2') for _ in range(n_env)])
eval_env = SubprocVecEnv([lambda: gym.make('BipedalWalker-v2') for _ in range(n_env)])

model = SAC(MlpPolicy, env, eval_env, verbose=1, learning_starts=1000, tensorboard_log="./sac_walker_tensorboard/",
            replay_buffer=ReplayBuffer(100000), eval_freq=100000, nb_eval_rollouts=10, learning_rate=0.0003, batch_size=64)
model.learn(total_timesteps=100000)
model.save("sac_walker")
cp.disable()
cp.dump_stats("test_sac_1env.cprof")

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