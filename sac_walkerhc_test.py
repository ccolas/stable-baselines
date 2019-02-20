import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import SubprocVecEnv
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

# BipedalWalkerHardcore-v2:
#   n_timesteps: !!float 5e7
#   policy: 'CustomSACPolicy'
#   learning_rate: lin_3e-4
#   buffer_size: 2000000
#   batch_size: 64
#   ent_coef: 0.005
#   train_freq: 1
#   gradient_steps: 1
#   learning_starts: 1000

n_env = 4
env = SubprocVecEnv([lambda: gym.make('BipedalWalkerHardcore-v2') for _ in range(n_env)])
eval_env = SubprocVecEnv([lambda: gym.make('BipedalWalkerHardcore-v2') for _ in range(n_env)])

model = SAC(MlpPolicy, env, eval_env, verbose=1, learning_starts=1000, tensorboard_log="./sac_walker_tensorboard/",
            replay_buffer=ReplayBuffer(1000000), eval_freq=1000000, nb_eval_rollouts=20, learning_rate=0.0003, batch_size=64)
model.learn(total_timesteps=50000000, log_interval=10)
model.save("sac_walkerhc")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_walkerhc", env=env, eval_env=eval_env)

obs = env.reset()
r = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    r+=rewards[0]
    if dones:
        print(r)
        r = 0
    env.render()