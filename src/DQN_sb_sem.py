import gym 
# from gym import envs
# for env in envs.registry.all():
#     print(env)
import gym_sem

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DQN

layer_size = 32

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[layer_size, layer_size],
                                           layer_norm=False,
                                           feature_extraction="mlp")



env = make_vec_env('sem-v0', n_envs=1)
model = DQN(CustomDQNPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("/home/alexandre/sem-project-logs/0100k_test_sem1_3x4_dqn_32")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.5f} +/- {std_reward:.5f}")

# t = []
# t_step = 10
# t_counter = 1
# #while abs(t) < 11:
# for k in range(20):
#     t_temp = []
#     for h in range(2):
#         print(t_counter)
#         mean_reward_1, std_reward_1 = evaluate_policy(model, env, n_eval_episodes=t_step * t_counter)
#         mean_reward_2, std_reward_2 = evaluate_policy(model, env, n_eval_episodes=t_step * t_counter)
#         t_temp.append(abs( (mean_reward_1 - mean_reward_2) / (np.sqrt((std_reward_1 + std_reward_2) / 2) * np.sqrt(2/(t_step * t_counter))) ))
#         print(t)
#     t.append(sum(t_temp) / len(t_temp))
#     t_counter += 1
# print(f"mean_reward:{mean_reward_1:.5f} +/- {std_reward_1:.5f}")

# plt.plot(t)
# plt.show()
env.close()