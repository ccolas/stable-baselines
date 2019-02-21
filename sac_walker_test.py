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
@click.option('--logdir', type=str, default='sac_default_log/log', help='the path to where data should go.')
@click.option('--n_env', type=int, default=1, help='the number of vectorized envs')
def main(env, logdir, n_env):
    print("launching sac with {} envs, logs in {}, env {}".format(n_env, logdir, env))
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
                replay_buffer=ReplayBuffer(200000), eval_freq=200000, nb_eval_rollouts=40, learning_rate=0.0003, batch_size=64,
                logdir="sac_walker_log/plaf20env")
    model.learn(total_timesteps=2000000)
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