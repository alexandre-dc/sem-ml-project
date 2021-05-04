import gym
import gym_semaphore

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

import os, os.path

class Old_Models():
    #env = gym.make('semaphore-v0')
    def __init__(self):
        self.old_models_name = []
        self.old_models_policy = []
        

    def atualize(self):
        new_model_eval = 0
        path = os.path.join(os.path.expanduser('~'), 'Documents', 'logs')
        list_models = [name for name in os.listdir(path)]
        if len(list_models) > len(self.old_models_name):
            for model_name in list_models:
                if model_name not in self.old_models_name:
                    self.old_models_name.append(model_name)

                    if len(self.old_models_name) > 10:
                        new_model = DQN.load(os.path.join(path, model_name))
                        #new_model_eval = self.eval(new_model)
                        self.old_models_policy.append(new_model)

            while len(self.old_models_policy) > 20:
                self.old_models_policy.pop(0)
        
        return len(self.old_models_policy)

    def old_model_predict(self, id, obs):
        return self.old_models_policy[id].predict(obs)

    def eval(self, model):
        return evaluate_policy(model, env, n_eval_episodes=5000)[0]