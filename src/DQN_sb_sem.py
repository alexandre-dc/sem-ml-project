import gym
import gym_sem
import sem_game

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DQN

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

layer_size_lst = [64]
n_train_size_lst = [10000]
minimax_rate_lst = [0.8]
test_steps = 10


layer_size = 8
layer_size_cnn = 64

class CustomDQNPolicy_8(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_8, self).__init__(*args, **kwargs,
                                           layers=[8, 8],
                                           layer_norm=False,
                                           feature_extraction="mlp")
class CustomDQNPolicy_16(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_16, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")
class CustomDQNPolicy_32(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_32, self).__init__(*args, **kwargs,
                                           layers=[32, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")
class CustomDQNPolicy_64(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_64, self).__init__(*args, **kwargs,
                                           layers=[64, 64],
                                           layer_norm=False,
                                           feature_extraction="mlp")
class CustomDQNPolicy_128(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_128, self).__init__(*args, **kwargs,
                                           layers=[128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")

class CustomDQNPolicy_256(FeedForwardPolicy):
    def __init__(self,*args, **kwargs):
        super(CustomDQNPolicy_128, self).__init__(*args, **kwargs,
                                           layers=[128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")
                                        
class CustomDQNPolicy_Cnn(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy_Cnn, self).__init__(*args, **kwargs,
                                           layers=[layer_size_cnn, layer_size_cnn],
                                           layer_norm=False,
                                           feature_extraction="cnn")

# train_steps = 100000
# test_steps = 1000
# # save_file = str(int(train_steps/1000)) + "k_sem1_3x2_" + str(int(layer_size))
# #env = make_vec_env('sem-v0', n_envs=1)
# env = gym.make('sem-v0', _type='DQN')
# env_test = gym.make('sem-v0', _type='DQN_test')
# save_file = "policy2_sem" + str(MAX_MOVES) + "_" + str(BOARD_ROWS) + "_" + str(BOARD_COLS)
# env.agent_turn = -1
# env_test.agent_turn = -1
# #model = DQN.load("/home/alexandre/sem-project-logs/" + save_file, env)
# model = DQN(CustomDQNPolicy, env, verbose=1)
# t0 = time.clock()
# model.learn(total_timesteps=train_steps)
# print(time.clock() - t0)
# model.save("/home/alexandre/sem-project-logs/dqn/" + save_file)
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=test_steps)
# print(f"mean_reward:{mean_reward:.5f} +/- {std_reward:.5f}")

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


#env_test = gym.make('sem-v0', _type='DQN_test')
#print(len(env_test.minimax_player.model))
save_file = "policy2_sem" + str(MAX_MOVES) + "_" + str(BOARD_ROWS) + "x" + str(BOARD_COLS)
#env_test.agent_turn = -1



for minimax_rate in minimax_rate_lst:
    env = gym.make('sem-v0', _type='DQN', _minimax_rate=minimax_rate)
    env.agent_turn = -1
    for l_size in layer_size_lst:
        for n_train in n_train_size_lst:
            sum = 0
            for i in range(1):
                if l_size == 8:
                    model = DQN(CustomDQNPolicy_8, env, verbose=1, exploration_fraction=0.2, exploration_initial_eps=0.1)
                elif l_size == 16:
                    model = DQN(CustomDQNPolicy_16, env, verbose=1, exploration_fraction=0.2)
                elif l_size == 32:
                    model = DQN(CustomDQNPolicy_32, env, verbose=1, exploration_fraction=0.2)
                elif l_size == 64:
                    #model = DQN(CustomDQNPolicy_64, env, verbose=1, exploration_fraction=0.2, exploration_initial_eps=0.1)
                    #model = DQN(CustomDQNPolicy_Cnn, env, verbose=1, exploration_fraction=0.2)
                    model = DQN.load("/home/alexandre/sem-project-logs/dqn/" + save_file + "_4M", env, verbose=1, exploration_fraction=0.05, exploration_initial_eps=0.1)
                elif l_size == 128:
                    model = DQN(CustomDQNPolicy_128, env, verbose=1, exploration_fraction=0.2)
                elif l_size == 256:
                    model = DQN(CustomDQNPolicy_256, env, verbose=1, exploration_fraction=0.2)
                else:
                    model = DQN(CustomDQNPolicy_256, env, verbose=1, exploration_fraction=0.2)
                t0 = time.clock()
                model.learn(total_timesteps=n_train)
                time_learning = time.clock() - t0
                all_rewards = []
                for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    env_test = gym.make('sem-v0', _type='DQN_test', _minimax_rate=i)
                    env_test.agent_turn = -1
                    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=test_steps)
                    all_rewards.append(mean_reward)
                model.save("/home/alexandre/sem-project-logs/dqn/" + save_file + "_5M")
                print(minimax_rate)
                print(l_size)
                print(n_train)
                print(all_rewards)
                f = open("log.txt", "a")
                f.write(str(time_learning) + "\n")
                f.write(str(minimax_rate) + "\n")
                f.write(str(l_size) + "\n")
                f.write(str(n_train) + "\n")
                f.write(str(all_rewards) + "\n")
                f.write("\n")
                f.close()
    
fw = open('/home/alexandre/sem-project-logs/minimax/board_nextMoves_' + str(sem_game.MAX_MOVES) + "_" + str(sem_game.BOARD_ROWS) + "x" + str(sem_game.BOARD_COLS) + "_" + "MMPS", 'wb')
pickle.dump(env.minimax_player.model, fw)
fw.close()
env.close()
env_test.close()