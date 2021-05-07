import gym

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

import sem_game
from sem_game import Board
from Player import Player

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class SemEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(BOARD_ROWS * BOARD_COLS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS, MAX_MOVES), dtype=np.uint8)

        self.rand_bot = Player(_name="board_nextMoves", _player_type="Minimax")
        self.rand = Player()
        self.agent_turn = -1

        self.board = Board()
        self.done = False

        self.reset()


    def step(self, action = -1):
        reward = 0
        #----------------- Monte Carlo's Step ------------------------------

        # if action != -1:
        #     if type(action) == int:
        #         movePos = (int(action/BOARD_COLS), int(action%BOARD_COLS))
        #     else:
        #         movePos = action
        #     moveDone = self.board.make_move(movePos)

        #     win = self.board.check_win()
        #     if win >= 0:
        #         reward = self.board.turn
        #         self.done = True
                
        #         return self.board.getHash(), reward, self.done, {}
        
        # else:
        #     positions = self.board.availablePositions()
        #     movePos = self.rand_bot.choose_action(positions)
        #     moveDone = self.board.make_move(movePos)

        #     win = self.board.check_win()
        #     if win >= 0:
        #         reward = self.board.turn
        #         self.done = True
                
        #         return self.board.getHash(), reward, self.done, {}

        # return self.board.getHash(), reward, self.done, {}

        #-------------------- DQN SB's Step -----------------------------------

            #---------- Agent Move -----------------------------
        move_pos = (int(action / BOARD_COLS), int(action % BOARD_COLS))
        moveDone = self.board.make_move(move_pos)

        if moveDone == 0:
            reward = -2
            self.done = True
            
            return self.board.get_one_hot(True), reward, self.done, {}
        
        win = self.board.check_win()
        if win != -1:
            reward = 1
            self.done = True
            
            return self.board.get_one_hot(), reward, self.done, {}

            #--------- Random Bot Move -------------------------
        if np.random.rand() < 0.2:
            positions = self.board.availablePositions()
            botMove = self.rand_bot.choose_action(positions, self.board, player = -self.agent_turn)
            moveDone = self.board.make_move((botMove[0], botMove[1]))
        else:
            positions = self.board.availablePositions()
            botMove = self.rand.choose_action(positions, self.board, player = -self.agent_turn)
            moveDone = self.board.make_move((botMove[0], botMove[1]))

        win = self.board.check_win()
        if win != -1:
            reward = -1
            self.done = True
            
            return self.board.get_one_hot(), reward, self.done, {}

        # return self.board.state, reward, self.done, {}
        # return self.one_hot_encode(self.board.state), reward, self.done, {}
        return self.board.get_one_hot(), reward, self.done, {}
        #return canonic_state, reward, self.done, {}

    def reset(self):
        self.board.reset()
        self.done = False

        if self.agent_turn == -1:
            positions = self.board.availablePositions()
            botMove = self.rand.choose_action(positions, self.board, player = -self.agent_turn)
            moveDone = self.board.make_move((botMove[0], botMove[1]))

        # if np.random.choice(range(2)) == 1:
        # if self.agent_turn == -1:
        #     positions = self.board.availablePositions()
        #     botMove = self.rand_bot.choose_action(positions, self.board, player = -self.agent_turn)
        #     moveDone = self.board.make_move((botMove[0], botMove[1]))

        # hash_state = self.board.getHash()
        # canonic_state = self.get_canonic_state(hash_state)
        # canonic_state = self.board.get_array_from_flat(canonic_state[0])
        return self.board.get_one_hot()

    def render(self):
        self.board.showBoard()

    def possible_move_boards(self):
        positions = self.board.availablePositions()
        possible_Boards = []

        for p in positions:
            self.board.make_move(p)
            possible_Boards.append((self.board.getHash(), p))
            self.board.undo_move(p)
                
        return possible_Boards

    def set_state(self, state):
        return self.board.set_state(state)

    def get_symmetry(self, board_hash = -1):
        return self.board.get_symmetry(board_hash)

    def get_canonic_score(self, board_flat = -1):
        return self.board.get_canonic_score(board_flat)

            # Returns always the same state for all any symmetric states and a list of all those symmetric states
    def get_canonic_state(self, board_hash = -1):
        all_symmetry = self.get_symmetry(board_hash)
        all_canonic_score = [self.get_canonic_score(b) for b in all_symmetry]
        canonic_board_index = all_canonic_score.index(max(all_canonic_score))
        return all_symmetry[canonic_board_index], all_symmetry

    def get_canonic_state_mask (self, board_flat = -1):
        return self.board.get_canonic_state_mask(board_flat)

    def one_hot_encode(self, state):
        one_hot_board = np.zeros((MAX_MOVES, BOARD_ROWS, BOARD_COLS))
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for m in range(MAX_MOVES + 1):
                    if state[i, j] == m + 1:
                        one_hot_board[m, i, j] = 1
                        break
        # print("------------------")
        # print(state)
        # print(one_hot_board)
        #padded_state = np.pad(one_hot_board, ((0, 0), (31, 30), (31, 31)),mode='constant', constant_values=(0))
        return one_hot_board

    def pad (self, state):
        padded_state = np.pad(state, ((0, 0), (31, 30), (30, 30)),mode='constant', constant_values=(0))
        #print(padded_state)
        return padded_state