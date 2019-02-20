import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import SAC

n_env = 4

env = SubprocVecEnv([lambda: gym.make('Pendulum-v0') for _ in range(n_env)])
eval_env = SubprocVecEnv([lambda: gym.make('Pendulum-v0') for _ in range(n_env)])

model = SAC(MlpPolicy, env, eval_env, verbose=1, eval_freq=1000, learning_starts=100, tensorboard_log="./sac_pendulum_logutestu")
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_pendulum")

# del model # remove to demonstrate saving and loading
#
# model = SAC.load("sac_pendulum", env=env, eval_env=eval_env)
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()