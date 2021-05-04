import gym
from gym import spaces

import numpy as np
from tensorflow.keras.models import Sequential

from ttt import Board, Player

class tttEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,3,3), dtype=np.uint8)

        self.rand_bot = Player()

        self.board = Board()
        self.done = False

        self.reset()

    def step(self, action = -1, player = 1, model = None):

        reward = 0
        if player == 1:
            if type(action) == int:
                movePos = (int(action/3), int(action%3))
            else:
                movePos = action

            moveDone = self.board.make_move(movePos, player)

            if moveDone == 0:
                reward = -2
                self.done = True
                
                return self.board.state, reward, self.done, {}

            win = self.board.check_win()
            if win != 0:
                if win == 2:
                    reward = 0
                    self.done = True
                else:
                    reward = player
                    self.done = True
                
                return self.board.state, reward, self.done, {}



            positions = self.board.availablePositions()
            movePos = self.rand_bot.choose_action(positions)
            moveDone = self.board.make_move(movePos, -1)

            win = self.board.check_win()
            if win != 0:
                if win == 2:
                    reward = 0
                    self.done = True
                else:
                    reward = -1
                    self.done = True
                
                return self.board.state, reward, self.done, {}
        
        # else:
        #     if action == 'random':
        #         positions = self.board.availablePositions()
        #         movePos = self.rand_bot.choose_action(positions)
        #         moveDone = self.board.make_move(movePos, -1)
        #     else:
        #         if type(action) == int:
        #             movePos = (int(action/3), int(action%3))
        #         else:
        #             movePos = action
        #         moveDone = self.board.make_move(movePos, -1)

        #     if moveDone == 0:
        #         reward = 0
        #         self.done = True
                
        #         return self.board.state, reward, self.done, {}

        #     win = self.board.check_win()
        #     if win != 0:
        #         if win == 2:
        #             reward = 0
        #             self.done = True
        #         else:
        #             reward = -1
        #             self.done = True
                
        #         return self.board.state, reward, self.done, {}

        return self.board.state, reward, self.done, {}

        #------------------
        # if model == None:
        #     positions = self.board.availablePositions()
        #     movePos = self.rand_bot.choose_action(positions)
        #     moveDone = self.board.make_move(movePos, -1)
        # else:
        #     s = self.one_hot_encode(self.board.state)
        #     qvals_s = model.predict(s.reshape(1,18))
        #     a = np.argmax(qvals_s)
        #     movePos = (int(a/3), int(a%3))
        #     moveDone = self.board.make_move(movePos, -1)

        # win = self.board.check_win()
        # if win != 0:
        #     if win == 2:
        #         reward = 0
        #         self.done = True
        #     else:
        #         reward = -1
        #         self.done = True
            
        #     return self.one_hot_encode(self.board.state), reward, self.done, {}


        #----------------- Monte Carlo's Step ------------------------------

        # reward = 0

        # if action != -1:
        #     if type(action) == int:
        #         movePos = (int(action/3), int(action%3))
        #     else:
        #         movePos = action
        #     moveDone = self.board.make_move(movePos, player)
        #     win = self.board.check_win()
        #     if win != 0:
        #         if win == 2:
        #             reward = 0
        #             self.done = True
        #         else:
        #             reward = 1
        #             self.done = True
                
        #         return self.one_hot_encode(self.board.state), reward, self.done, {}
        
        # else:

        #     positions = self.board.availablePositions()
        #     movePos = self.rand_bot.choose_action(positions)
        #     moveDone = self.board.make_move(movePos, player)

        #     win = self.board.check_win()
        #     if win != 0:
        #         if win == 2:
        #             reward = 0
        #             self.done = True
        #         else:
        #             reward = player
        #             self.done = True
                
        #         return self.board.state, reward, self.done, {}

        # return self.board.state, reward, self.done, {}

        


    def render(self):
        self.board.showBoard()

    def reset(self):
        self.board.reset()
        self.done = False

        # if np.random.rand()  < 0.5:
        #     positions = self.board.availablePositions()
        #     movePos = self.rand_bot.choose_action(positions)
        #     moveDone = self.board.make_move(movePos, -1)

        return self.board.state

    def close(self):
        pass

    def one_hot_encode(self, state):
        one_hot_board = np.zeros((2, 3, 3))
        for i in range(3):
            for j in range(3):
                if state[i, j] == 1:
                    one_hot_board[0, i, j] = 1
                elif state[i, j] == -1:
                    one_hot_board[1, i, j] = -1

        return one_hot_board

    def possible_move_boards(self, player = 1):
        positions = self.board.availablePositions()
        possible_Boards = []

        temp_state = self.board.state


        for p in positions:
            temp_state[p] = player
            possible_Boards.append((list(temp_state.reshape(3*3)), p))
            temp_state[p] = 0
                

        return possible_Boards

    def set_state(self, state):
        self.board.state = state
