import gym
import gym_sem
import sem_game
from sem_game import Board, Game

import numpy as np
import operator

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class Q_Learning:
    def __init__(self, agent, adv, minimax_rate):
        self.p1 = adv
        self.p2 = agent
        self.board = Board()
        self.game = Game(self.p1, self.p2)
        self.env = gym.make('sem-v0', _type='Q-learning', _minimax_rate=minimax_rate)
        self.env.agent_turn = -1

    def feed_reward (self, reward):
        self.p2.all_rewards.append(reward)
        s = self.p2.states.pop(-1)
        a = self.p2.moves.pop(-1)
        counter = len(self.p2.moves) - 1

        if type(self.p2.states_value.get(s)) == type(None):
            self.p2.states_value[s] = [0] * (BOARD_ROWS * BOARD_COLS)
        self.p2.states_value[s][a[0]*BOARD_COLS + a[1]] = reward

        next_s = s

        for s in reversed(self.p2.states):
            a = self.p2.moves[counter]
            if type(self.p2.states_value.get(s)) == type(None):
                self.p2.states_value[s] = [0] * (BOARD_ROWS * BOARD_COLS)

            max_idx, max_value = max(enumerate(self.p2.states_value[next_s]), key=operator.itemgetter(1))
            self.p2.states_value[s][a[0]*BOARD_COLS + a[1]] += self.p2.lr * ( self.p2.gamma * max_value - self.p2.states_value[s][a[0]*BOARD_COLS + a[1]] )
            next_s = s
            counter -= 1

    # def feed_reward (self, reward):
    #     self.p2.all_rewards.append(reward)
    #     for st in reversed(self.p2.states):
    #             # ....... Without canonic states ...............
    #         # if self.p2.states_value.get(st) is None:
    #         #     self.p2.states_value[st] = 0
    #         # self.p2.states_value[st] += self.p2.lr * (reward)
    #         # reward = reward * self.p2.gamma

    #             # ........ With canonic states ...............
    #         if st in self.p2.dict_canonic_states:
    #             key_state = self.p2.dict_canonic_states[st]
    #             self.p2.states_value[key_state] += self.p2.lr * (reward)
    #         else:
    #             canonic_state, all_symmetry = self.board.get_canonic_state(st)
    #             key_state = str(canonic_state)
    #             for sym in all_symmetry:
    #                 self.p2.dict_canonic_states[str(sym)] = key_state

    #             self.p2.states_value[key_state] = self.p2.lr * (reward)
    #         reward = reward * self.p2.gamma
            

    def train(self, steps=1000, progressive_lr=False):
        self.p1.all_rewards = []
        self.p2.all_rewards = []
        self.steps_made = 0
        self.games_made = 0
        self.steps_made_until_last_game = 0
        log_interval = 100
        self.last_log = 0

        a = self.p2.epsilon_min
        b = self.p2.epsilon
        c = steps * self.p2.epsilon_rate
        d = np.power(a/b, 1/c)

        while self.steps_made < steps:
            if self.p2.epsilon > self.p2.epsilon_min and self.steps_made > 0:
                self.p2.epsilon
                self.p2.epsilon *= np.power(d, int(self.steps_made - self.steps_made_until_last_game))
                self.steps_made_until_last_game = self.steps_made
                

            if self.games_made - self.last_log >= log_interval:
                print("Steps {}".format(self.steps_made))
                print("Games {}".format(self.games_made))
                #print("Players epsilon: {}".format(p2.epsilon))
                #print("Bot epsilon: {}".format(p1.epsilon))
                mean_p2 = sum(self.p2.all_rewards[-log_interval:])/log_interval
                self.p2.all_rewards_means.append(mean_p2)
                print("Mean round_r: {0:.1f}".format(mean_p2))
                #mean_ep_len = sum(self.all_ep_len[-log_interval:])/log_interval
                #print("Mean ep_len: {}".format(mean_ep_len))

                if progressive_lr == True:
                    mean_test = sum(self.p2.all_rewards[-5000:])/5000
                    if mean_test > 0.9:
                        self.p2.lr = 0.00002
                    elif mean_test > 0.8:
                        self.p2.lr = 0.00005
                    #elif mean_test > 0.8:
                    #    self.p2.lr = 0.0001
                    print(mean_test)
                print("Current learning-rate: {}".format(self.p2.lr))
                print( "Current epsilon: {}".format(self.p2.epsilon) )
                print()
                self.last_log = self.games_made


            s = self.env.reset()
            done = False
            while not done:
                self.p2.states.append(self.env.board.getHash())
                action = self.p2.choose_action(self.env.board)
                s, r, done, info = self.env.step(action)
                self.p2.moves.append(action)

            reward = r
            self.feed_reward(reward)
            self.steps_made += len(self.p2.states)
            self.p2.reset()
            self.game.reset()
            self.games_made += 1

    def test(self, steps=1000):
        self.p2.set_test_mode(True)
        test_rewards = []
        print("Testing... ")
        # for i in range(steps):
        #     s = self.env.reset()
        #     done = False
        #     while not done:
        #         self.p2.states.append(self.env.board.getHash())
        #         action = self.p2.choose_action(self.env.board)
        #         s, r, done, info = self.env.step(action)
        #         self.p2.moves.append(action)

        #     test_rewards.append(r)
        #     self.p2.reset()
        #     self.game.reset()
        # print(sum(test_rewards)/steps)
        # self.p2.set_test_mode(False)

        all_rewards = []
        for i in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            print("minimax_test_rate: " + str(i))
            rewards = 0
            env_test = gym.make('sem-v0', _type='Q-learning_test', _minimax_rate=i)
            env_test.agent_turn = -1
            for game in range(steps):
                s = env_test.reset()
                done = False
                while not done:
                    self.p2.states.append(env_test.board.getHash())
                    action = self.p2.choose_action(env_test.board)
                    s, r, done, info = env_test.step(action)
                    self.p2.moves.append(action)
                rewards += r
            all_rewards.append(rewards/steps)
        return all_rewards

    def play_game(self):
        state = self.env.reset()
        done = False
        turn = 1

        while not done:
            if turn == 1:
                try:
                    action = self.p1.choose_action(self.board)
                    s, r, done, _ = self.env.step(action = action)
                except:
                    s, r, done, _ = self.env.step()
            else:
                s, r, done, _ = self.env.step()
            
            if done:
                return r

            turn *= -1