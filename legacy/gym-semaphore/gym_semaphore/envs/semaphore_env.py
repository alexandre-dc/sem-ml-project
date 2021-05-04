import gym
from gym import spaces

import numpy as np
import random
import copy
import operator

from semaphore import Board, Player, Game
from gym_semaphore.old_models import Old_Models

import os, os.path
import pickle

class SemaphoreEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,3,4), dtype=np.uint8)

        self.rand_bot = Player(epsilon=1)
        self.pseudo_bot = Player(epsilon=1)
        #self.pseudo_bot.loadPolicy("/home/alexandre/Documents/gym/gym-semaphore/gym_semaphore/policy_321")
        self.state = Board()
        self.game = Game(self.rand_bot, self.rand_bot)

        self.done = False

        self.old_models_len = 0
        self.last_model_eval = 0
        self.adversary = -1
        self.lose_reward = -1

        self.game_log = []

        self.Old_Models = Old_Models()
        self.best_moves = self.getMinimaxPolicy()
        self.reset()

        
    def step(self, action):

        #return self.step_human(action)


        self.done = False
                                    # Agent move
        move_pos = (int(action/4), int(action%4))
        moveDone = self.state.make_move(move_pos)
        #self.render()
        #self.game_log.append(copy.deepcopy(self.state))

        reward = 0
        if moveDone == 0:
            reward = self.lose_reward
            self.done = True
            
            return self.one_hot_encode(self.state), reward, self.done, {}

        win = self.state.check_win()
        if win != -1:
            reward = 1
            self.done = True
            #print("////////////////////")
            
            return self.one_hot_encode(self.state), reward, self.done, {}

                                    # Adversary move
        if self.adversary == -1:
            positions = self.state.availablePositions()
            botMove = self.rand_bot.chooseAction(positions, self.state.board)
            #botMove = self.pseudo_bot.perfectMove(self.state.board)
            #botMove = self.game.minimax(self.state, 35, 1, self.best_moves)
        elif self.adversary == 0:
            botMove = self.game.minimax(self.state, 35, 1, self.best_moves)
        else:
            bot_action, _states = self.Old_Models.old_model_predict(self.adversary, self.one_hot_encode(self.state))
            botMove = (int(bot_action/4), int(bot_action%4))

        
        
        moveDone = self.state.make_move((botMove[0], botMove[1]))
        #self.render()
        #self.game_log.append(copy.deepcopy(self.state))

        win = self.state.check_win()
        if win != -1:
            reward = self.lose_reward
            self.done = True
            # for i in range(len(self.game_log)):
            #     self.state = self.game_log[i]
            #     self.render()
            #print("-------------------------------------")
            
            return self.one_hot_encode(self.state), reward, self.done, {}

        return self.one_hot_encode(self.check_mainBoard(self.state)), reward, self.done, {}


    def render(self):
        self.state.showBoard()

    def reset(self):
        self.state.reset()
        self.done = False
        self.game_log = []

        # if self.old_models_len == 0:    # Verificar se existem agents anteriores
        #     self.adversary = -1
        # else:
        #     if random.random() < 0.8:
        #         self.adversary = -1
        #     else:
        #         self.adversary = random.randrange(0, self.old_models_len)

        if random.random() < 1:
            self.adversary = -1
        else:
            self.adversary = 0

        if random.random() < 1:       # Decidir que jogador faz o primeiro move
            if self.adversary == -1:
                positions = self.state.availablePositions()
                botMove = self.rand_bot.chooseAction(positions, self.state.board)
                #botMove = self.pseudo_bot.perfectMove(self.state.board)
                #botMove = self.game.minimax(self.state, 35, 1, self.best_moves)
            elif self.adversary == 0:
                positions = self.state.availablePositions()
                botMove = self.rand_bot.chooseAction(positions, self.state.board)
            else:
                bot_action, _states = self.Old_Models.old_model_predict(self.adversary, self.one_hot_encode(self.state.board))
                botMove = (int(bot_action/4), int(bot_action%4))
            moveDone = self.state.make_move((botMove[0], botMove[1]))
        
        #self.render()
        #self.game_log.append(copy.deepcopy(self.state))
        
        #////////////////////////////////////////////////////////////////////////////
        while 1:
            rand = random.choice([0])
            temp_state = copy.deepcopy(self.state)
            while rand > 0:
                positions = self.state.availablePositions()
                botMove = self.rand_bot.chooseAction(positions, self.state.board)
                moveDone = self.state.make_move((botMove[0], botMove[1]))

                rand -= 1
            if self.state.check_win() == -1:
                break 
            self.state = temp_state
        
        #if random.random() < 0.01:
            #self.getPolicies()
            #if self.last_model_eval > 0.95:
            #    self.lose_reward *= 1
        #self.render()
        return self.one_hot_encode(self.check_mainBoard(self.state))

    def close(self):
        pass


                                    # Function para testar o bot vs human
    def step_human(self, action):
        move_pos = (int(action/4), int(action%4))
        moveDone = self.state.make_move(move_pos)
        self.render()

        reward = 0

        win = self.state.check_win()
        if win != -1:
            reward = 1
            self.done = True
            
            return self.one_hot_encode(self.state.board), reward, self.done, {}

        move1 = int(input("Move1: "))
        move2 = int(input("Move2: "))

        moveDone = self.state.make_move((move1, move2))
        self.render()

        win = self.state.check_win()
        if win != -1:
            reward = -1
            self.done = True
            
            return self.one_hot_encode(self.state.board), reward, self.done, {}

        return self.one_hot_encode(self.state.board), reward, self.done, {}

    def one_hot_encode(self, state):
        board = state.board
        one_hot_board = np.zeros((3, 3, 4))
        for i in range(3):
            for j in range(4):
                if board[i, j] == 1:
                    one_hot_board[0, i, j] = 1
                elif board[i, j] == 2:
                    one_hot_board[1, i, j] = 1
                elif board[i, j] == 3:
                    one_hot_board[2, i, j] = 1

        #print(one_hot_board)
        #print("__________")
        return one_hot_board

    def check_mainBoard(self, state):
        value = [0, 0, 0, 0]

        #while value1 == value2 or value1 == value3 or value1 == value4 or value2 == value3 or value2 == value4 or value3 == value4:
        value[0] = self.board_value(state)
        value[1] = self.board_value(self.Inv_Vertical(state))
        value[2]= self.board_value(self.Inv_Horizontal(state))
        value[3] = self.board_value(self.Inv_Horizontal(self.Inv_Vertical(state)))

        # for x in range(3):
        #     for y in range(x+1, 4):
        #         if value[x] == value[y]:
        #             #state.showBoard()
        #             # for i in range(100):
        #             #     print()

        idx, value = max(enumerate(value), key=operator.itemgetter(1))
        
        if idx == 0:
            #state.showBoard()
            return state
        if idx == 1:
            #self.Inv_Vertical(state).showBoard()
            return self.Inv_Vertical(state)
        if idx == 2:
            #self.Inv_Horizontal(state).showBoard()
            return self.Inv_Horizontal(state)
        if idx == 3:
            #self.Inv_Horizontal(self.Inv_Vertical(state)).showBoard()
            return self.Inv_Horizontal(self.Inv_Vertical(state))


    def board_value(self, state):
        matrix_value = [13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        board = state.board
        board_line = []
        for i in range(3):
            for j in range(4):
                board_line.append(board[i, j])

        total_value = 0
        for i in range(len(matrix_value)):
            total_value += board_line[i] * matrix_value[i]

        return total_value


    def Inv_Vertical(self, state):
        new_state = copy.deepcopy(state)
        board = new_state.board
        temp_board = np.zeros((3, 4))

        for i in range(3):
            for j in range(4):
                x = ((j + 1) - 4) * (-1)

                temp_board[i, x] = board[i, j]
        
        new_state.board = temp_board
        return new_state

    def Inv_Horizontal(self, state):
        new_state = copy.deepcopy(state)
        board = new_state.board
        temp_board = np.zeros((3, 4))

        for i in range(3):
            for j in range(4):
                x = ((i + 1) - 3) * (-1)

                temp_board[x, j] = board[i, j]
        
        new_state.board = temp_board
        return new_state

    def getPolicies (self):
        self.old_models_len = self.Old_Models.atualize()

    def getMinimaxPolicy(self):
        path = os.path.join(os.path.expanduser('~'), 'gym-project')
        file_name = [name for name in os.listdir(path) if name == 'policy_111']
        fr = open(os.path.join(path,file_name[0]), 'rb')
        best_moves = pickle.load(fr)
        fr.close()
        return best_moves