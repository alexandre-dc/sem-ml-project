import gym

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from numpy import random
from tensorflow.python.ops.gen_array_ops import broadcast_to_eager_fallback

import sem_game
from sem_game import Board
from Player import Player

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class SemEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, _type='DQN', _minimax_rate=0):
        self._type = _type
        self._minimax_rate = _minimax_rate

        self.action_space = spaces.Discrete(BOARD_ROWS * BOARD_COLS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(BOARD_ROWS, BOARD_COLS, MAX_MOVES), dtype=np.uint8)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, MAX_MOVES), dtype=np.uint8)

        self.minimax_player = Player(_name="board_nextMoves_3_3_4_mmps", _player_type="Minimax")
        self.rand = Player()
        self.agent_turn = 1

        self.board = Board()
        self.done = False
        self.padding = False
        self.minimax_test = False

        self.counter = 0

        self.reset()


    def step(self, action = -1):
        reward = 0
        #----------------- Monte Carlo's Step ------------------------------
        if self._type == 'Monte Carlo':
            if action != -1:
                print("here")
                if type(action) == int:
                    movePos = (int(action/BOARD_COLS), int(action%BOARD_COLS))
                else:
                    movePos = action
                moveDone = self.board.make_move(movePos)

                win = self.board.check_win()
                if win >= 0:
                    reward = -1
                    self.done = True
                    #print(reward)
                    
                    return self.board.getHash(), reward, self.done, {}

                # if self.minimax_test:
                #     self.board.showBoard()
            
            else:
                if self.minimax_test:
                    botMove = self.minimax_player.choose_action(self.board, player = -self.agent_turn)
                    #print("here")
                else:
                    botMove = self.rand.choose_action(self.board, player = self.board.turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))

                win = self.board.check_win()
                if win >= 0:
                    reward = self.board.turn
                    self.done = True
                    #print(reward)
                    return self.board.getHash(), reward, self.done, {}

                # if self.minimax_test:
                #     self.board.showBoard()

            self.board.turn *= -1
            return self.board.getHash(), reward, self.done, {}

        #-------------------- Q-learning Step ----------------------------------
        if self._type == "Q-learning":
            #---------- Agent Move -----------------------------
            move_pos = (int(action / BOARD_COLS), int(action % BOARD_COLS))
            moveDone = self.board.make_move(move_pos)

            if moveDone == 0:
                reward = -2
                self.done = True
                
                return self.board.getHash(), reward, self.done, {}
            
            win = self.board.check_win()
            if win != -1:
                reward = 1
                self.done = True
                
                return self.board.getHash(), reward, self.done, {}

            #--------- Random Bot Move -------------------------
            if np.random.rand() < 0:
                positions = self.board.availablePositions()
                botMove = self.minimax_player.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))
            else:
                positions = self.board.availablePositions()
                botMove = self.rand.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))

            win = self.board.check_win()
            if win != -1:
                reward = -1
                self.done = True
                
                return self.board.getHash(), reward, self.done, {}

            return self.board.getHash(), reward, self.done, {}

        #-------------------- DQN SB's Step -----------------------------------
        if self._type == "DQN":
            #---------- Agent Move -----------------------------
            move_pos = (int(action / BOARD_COLS), int(action % BOARD_COLS))
            moveDone = self.board.make_move(move_pos)

            if moveDone == 0:
                reward = -2
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}
            
            win = self.board.check_win()
            if win != -1:
                reward = 1
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}

            #--------- Random Bot Move -------------------------
            if np.random.rand() < self._minimax_rate:
                #print("here")
                #self.board.showBoard()
                positions = self.board.availablePositions()
                botMove = self.minimax_player.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))
                #self.board.showBoard()
            else:
                positions = self.board.availablePositions()
                botMove = self.rand.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))

            win = self.board.check_win()
            if win != -1:
                reward = -1
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}

            # return self.board.state, reward, self.done, {}
            # return self.one_hot_encode(self.board.state), reward, self.done, {}
            return self.board.get_one_hot(self.padding), reward, self.done, {}
            #return canonic_state, reward, self.done, {}

        #-------------------- DQN SB's Step Test -----------------------------------
        if self._type == "DQN_test":
            #---------- Agent Move -----------------------------
            move_pos = (int(action / BOARD_COLS), int(action % BOARD_COLS))
            moveDone = self.board.make_move(move_pos)

            if moveDone == 0:
                reward = -2
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}
            
            win = self.board.check_win()
            if win != -1:
                reward = 1
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}

            #--------- Random Bot Move -------------------------
            if np.random.rand() < 0.5:
                positions = self.board.availablePositions()
                botMove = self.minimax_player.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))
            else:
                positions = self.board.availablePositions()
                botMove = self.rand.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))

            win = self.board.check_win()
            if win != -1:
                reward = -1
                self.done = True
                
                return self.board.get_one_hot(self.padding), reward, self.done, {}

            # return self.board.state, reward, self.done, {}
            # return self.one_hot_encode(self.board.state), reward, self.done, {}
            return self.board.get_one_hot(self.padding), reward, self.done, {}
            #return canonic_state, reward, self.done, {}

    def reset(self):
        self.board.reset()
        self.done = False

        if self.agent_turn == -1:
            if random.rand() < 0.5 and self._type == "DQN" and self.counter < 5000:
                self.get_init_board()
            else:
                botMove = self.rand.choose_action(self.board, player = -self.agent_turn)
                moveDone = self.board.make_move((botMove[0], botMove[1]))

            self.board.turn = -1

        if self.minimax_test:
            self.board.showBoard()

        # if np.random.choice(range(2)) == 1:
        # if self.agent_turn == -1:
        #     positions = self.board.availablePositions()
        #     botMove = self.rand_bot.choose_action(self.board, player = -self.agent_turn)
        #     moveDone = self.board.make_move((botMove[0], botMove[1]))

        # hash_state = self.board.getHash()
        # canonic_state = self.get_canonic_state(hash_state)
        # canonic_state = self.board.get_array_from_flat(canonic_state[0])
        return self.board.get_one_hot(self.padding)

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
        return padded_state.reshape(-1, BOARD_ROWS, BOARD_COLS, MAX_MOVES)

    def reshape_cnn(self, state):
        return state.reshape(-1, BOARD_ROWS, BOARD_COLS, MAX_MOVES)

    def get_init_board(self):
        done = False
        turn = 1
        moves_lst = []
        while not done:
            if np.random.rand() < 0.1:
                botMove = self.minimax_player.choose_action(self.board, player = turn)
            else:
                botMove = self.rand.choose_action(self.board, player = turn)
            moveDone = self.board.make_move((botMove[0], botMove[1]))

            moves_lst.append(botMove)

            win = self.board.check_win()
            if win != -1:
                if turn == -1:
                    possible_steps_back = []
                    for i in range(len(moves_lst)):
                        if i%2 != 0:
                            possible_steps_back.append(i)
                    steps_back = random.choice([1])
                    reversed_moves_lst = []
                    for move in reversed(moves_lst):
                        reversed_moves_lst.append(move)
                    #print(len(moves_lst))
                    #print(moves_lst)
                    if turn == 1:
                        steps_back += 1
                    for i in range(steps_back):
                        self.board.undo_move(reversed_moves_lst[i])
                    done = True
                else:
                    self.board.reset()
                    moves_lst = []
                    turn = 1
                

            turn *= -1
        #self.board.showBoard()
        self.counter += 1
        return self.board.get_one_hot(self.padding)
