import small_sem
from small_sem import Board, Game, Player, Agent

import numpy as np

BOARD_ROWS = small_sem.BOARD_ROWS
BOARD_COLS = small_sem.BOARD_COLS
MAX_MOVES = small_sem.MAX_MOVES

class Q_learning:
    def __init__(self, agent, adv):
        self.p1 = adv
        self.p2 = agent
        self.game = Game(self.p1, self.p2)
        #self.dict_canonic_states = {}

    def feed_reward (self, reward):
        self.p2.all_rewards.append(reward)
        count = len(self.p2.states) - 1
        reversed_states = [st for st in reversed(self.p2.states)]
        next_st = None
        for st in reversed_states:
            a = self.p2.moves[count]
            if self.p2.states_value.get(st) is None:
                self.p2.states_value[st] = np.zeros((BOARD_ROWS, BOARD_COLS))
            if next_st == None:
                self.p2.states_value[st][a] += self.p2.lr * (reward)
            else:
                self.p2.states_value[st][a] += self.p2.lr * (self.p2.gamma * np.max(self.p2.states_value[next_st]) - self.p2.states_value[st][a])
            next_st = st
            #reward = reward * self.p2.gamma
            count -= 1

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
                    if mean_test > 0.96:
                        self.p2.lr = 0.000001
                    elif mean_test > 0.94:
                        self.p2.lr = 0.00001
                    elif mean_test > 0.9:
                        self.p2.lr = 0.0001
                    print(mean_test)
                print("Current learning-rate: {}".format(self.p2.lr))
                print( "Current epsilon: {}".format(self.p2.epsilon) )

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

    

agent = Agent(name="q_learning", epsilon=0.8, lr=0.001)
adv = Player()
q = Q_learning(agent, adv)
q.train(steps=100000, progressive_lr=True)
q.test(steps=10000)
agent.savePolicy()