import gym
import importlib  
#gym_semaphore = importlib.import_module("gym-project","gym-semaphore","gym_semaphore")
import gym_semaphore

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import time
import os

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DQN

from stable_baselines.common.callbacks import CheckpointCallback


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 64],
                                           layer_norm=False,
                                           feature_extraction="mlp")




def test(model):
    env1 = gym.make('semaphore-v0')
    obs = env1.reset()
    done = False
    while done == False:
        print(model.action_probability(obs))
        env1.render()
        action_p = model.action_probability(obs)
        to_plot = np.zeros((3, 4))
        for i in range(len(action_p)):
            to_plot[int(i/4), i%4] = action_p[i]
        plt.imshow(to_plot, cmap='gist_heat', interpolation='nearest', extent=[-0.5, 3.5, 2.5, -0.5])
        plt.colorbar()
        plt.show()

        action, _ = model.predict(obs)
        obs, r, done, _ = env1.step(action)

        if done == True:
            print("Game Over")
    






env = make_vec_env('semaphore-v0', n_envs=1)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                         name_prefix='rl_model')

model = DQN(CustomDQNPolicy, env, verbose=1)
model.save("semaphore1_p2_10_09_0k")

test_results = []

for i in range(1):

    i += 0
    epsilon = (20-i)/20

    t0= time.clock()
    #model = DQN.load("semaphore1_p2_11_08_" + str((i)*20) + "k", env=env, gamma=0.99, learning_rate=0.0001, exploration_initial_eps=0.8, exploration_fraction=0.2).learn(total_timesteps=100000, callback=checkpoint_callback)
    #model = DQN.load("semaphore1_p2_27_08_" + str((i)*400) + "k", env=env, gamma=0.99, learning_rate=0.00001, exploration_initial_eps=0.08, exploration_fraction=0.003, tensorboard_log="./DQN_semaphore_tensorboard/")
    model = DQN.load("semaphore1_p2_10_09_" + str((i)*100) + "k", env=env, gamma=0.99, learning_rate=0.0001, exploration_initial_eps=0.8, exploration_fraction=0.2)
    model.learn(total_timesteps=200000, tb_log_name="first_run")
    #model = DQN.load("semaphore1_p2_10_09_" + str((i+1)*5000) + "k", env=env)
    
    model.save("semaphore1_p2_10_09_" + str((i+1)*100) + "k")
    print(time.clock())
    print (i)

    #path = os.path.join(os.path.expanduser('~'), 'Documents', 'logs')
    #filelist = [ f for f in os.listdir(path)]
    #for f in filelist:
    #    os.remove(os.path.join(path, f))

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20000)
    print(f"mean_reward:{mean_reward:.5f} +/- {std_reward:.5f}")

    #test(model)

env.close()

#plt.plot(test_results)
#plt.show()

#export PYTHONPATH=$PYTHONPATH:/home/alexandre/open_spiel
#export PYTHONPATH=$PYTHONPATH:/home/alexandre/open_spiel/build/python