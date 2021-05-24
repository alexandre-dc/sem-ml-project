import sem_game
from sem_game import Board, Game

import numpy as np

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class Q_Learning:
    def __init__(self, agent, adv):
        self.p1 = adv
        self.p2 = agent
        self.game = Game(self.p1, self.p2)
        #self.dict_canonic_states = {}

    def feed_reward (self, reward):
        self.p2.all_rewards.append(reward)
        for st in reversed(self.p2.states):
            if self.p2.states_value.get(st) is None:
                self.p2.states_value[st] = 0

            self.p2.states_value[st] += self.p2.lr * (reward)
            reward = reward * self.p2.gamma

    def train(self, steps=1000, progressive_lr=False):
        self.p1.all_rewards = []
        self.p2.all_rewards = []
        self.steps_made = 0
        log_interval = 5000
        self.last_log = 0

        a = self.p2.epsilon_min
        b = self.p2.epsilon
        c = steps * self.p2.epsilon_rate
        d = np.power(a/b, 1/c)

        print(b)
        #for i in range(rounds):
        while self.steps_made < steps:
            # if self.p2.epsilon > self.p2.epsilon_min:
            #     self.p2.epsilon *= 0.9999
            if self.p2.epsilon > self.p2.epsilon_min:
                self.p2.epsilon
                self.p2.epsilon *= d
                

            if self.steps_made - self.last_log >= log_interval:
                print("Steps {}".format(self.steps_made))
                #print("Players epsilon: {}".format(p2.epsilon))
                #print("Bot epsilon: {}".format(p1.epsilon))
                mean_p2 = sum(self.p2.all_rewards[-log_interval:])/log_interval
                self.p2.all_rewards_means.append(mean_p2)
                print("Mean round_r: {}".format(mean_p2))
                #mean_ep_len = sum(self.all_ep_len[-log_interval:])/log_interval
                #print("Mean ep_len: {}".format(mean_ep_len))

                if progressive_lr == True:
                    mean_test = sum(self.p2.all_rewards[-5000:])/5000
                    if mean_test > 0.95:
                        self.p2.lr = 0.000001
                    elif mean_test > 0.92:
                        self.p2.lr = 0.00001
                    elif mean_test > 0.8:
                        self.p2.lr = 0.0001
                    print(mean_test)
                print("Current learning-rate: {}".format(self.p2.lr))
                print( "Current epsilon: {}".format(self.p2.epsilon) )
                print()
                self.last_log = self.steps_made

            reward = self.game.play()
            self.feed_reward(reward)
            self.steps_made += len(self.p2.states)
            self.p2.reset()
            self.game.reset()

    def test(self, steps=1000):
        self.p2.set_test_mode(True)
        test_rewards = []
        print("Testing... ")
        for i in range(steps):
            reward = self.game.play()
            test_rewards.append(reward)
            self.p2.reset()
            self.game.reset()
        print(sum(test_rewards)/steps)
        self.p2.set_test_mode(False)