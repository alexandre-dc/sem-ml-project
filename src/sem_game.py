import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from stable_baselines import DQN

import tkinter as tk
import time
import copy
import operator

from Agent import Agent

BOARD_ROWS = 3
BOARD_COLS = 4
MAX_MOVES = 3

class Board:                                
    def __init__(self):                     # Inicialização do board
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.turn = 1
        self.movesMade = 0
        self.all_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.all_positions.append((i, j))
        self.sym_num = []
        for i in range(12):
            self.sym_num.append(10**i)

        self.num_board_rows = BOARD_ROWS
        self.num_board_cols = BOARD_COLS
        self.num_max_moves = MAX_MOVES
        
        #self.symmetry_masks = [(self.get_array_from_flat(sym[0]), sym[1]) for sym in self.get_symmetry_mask(self.sym_num)]
        
        
                            
    def getHash(self):                  # Hash do board para usar como key num dic            
        boardHash = str(self.state.reshape(BOARD_ROWS * BOARD_COLS))
        return boardHash

    def get_flat_from_hash(self, board_hash):                  # Hash do board para usar como key num dic            
        board_flat = []
        for c in board_hash:
            try:
                c = int(c)
                board_flat.append(c)
            except:
                ...
        return board_flat

    def getFlat(self):
        boardFlat = np.zeros(BOARD_ROWS * BOARD_COLS)
        for i in range(0, BOARD_ROWS):
            for j in range(0, BOARD_COLS):       
                boardFlat[i*BOARD_COLS + j] = self.state[i, j]
        # boardFlat = self.state.reshape(BOARD_ROWS * BOARD_COLS)
        return boardFlat

    def get_array_from_flat(self, board_flat):
        if type(board_flat) == type([]):
            board = np.zeros(BOARD_ROWS * BOARD_COLS)
            for i in range(len(board_flat)):
                board[i] = board_flat[i]
        else:
            board = board_flat
        board = board.reshape(BOARD_ROWS, BOARD_COLS)
        return board

    def get_one_hot(self, padded = False):
        one_hot_board = np.zeros((BOARD_ROWS, BOARD_COLS, MAX_MOVES))
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for m in range(MAX_MOVES + 1):
                    if self.state[i, j] == m + 1:
                        one_hot_board[i, j, m] = 1
                        break

        if padded:
            half_cols = int(BOARD_COLS / 2)
            one_hot_board = np.pad(one_hot_board, ((31, 30), (30, 30), (0, 0)),mode='constant', constant_values=(0))
            #one_hot_board = one_hot_board.reshape(-1, 64, 64, MAX_MOVES)
        return one_hot_board

    def check_win (self, return_line=False):               # Verifica se existe um vencedor - return (-1) se não existir, outro numero se existir
        if BOARD_ROWS >= 3:
            for j in range(BOARD_COLS):
                for lim in range(BOARD_ROWS - 2):
                    if self.state[lim,j] != 0:
                        if self.state[lim,j] == self.state[lim + 1,j] and self.state[lim,j] == self.state[lim + 2,j]:
                            if return_line:
                                win_line = [(lim,j), (lim + 1,j), (lim + 2,j)]
                                return self.state[lim,j], win_line
                            else:
                                return self.state[lim,j]
        
        if BOARD_COLS >= 3:
            for i in range(BOARD_ROWS):
                for lim in range(BOARD_COLS - 2):
                    if self.state[i,lim] != 0:
                        if self.state[i,lim] == self.state[i,lim + 1] and self.state[i,lim] == self.state[i,lim + 2]:
                            if return_line:
                                win_line = [(i,lim), (i,lim + 1), (i,lim + 2)]
                                return self.state[i,lim], win_line
                            else:
                                return self.state[i,lim]

        if BOARD_ROWS >= 3 and BOARD_COLS >= 3:
            for lim_cols in range(BOARD_COLS - 2):
                for lim_rows in range(BOARD_ROWS - 2):
                    if self.state[lim_rows + 1,lim_cols + 1] != 0:
                        if self.state[lim_rows,lim_cols] == self.state[lim_rows + 1,lim_cols + 1] and self.state[lim_rows,lim_cols] == self.state[lim_rows + 2,lim_cols + 2]:
                            if return_line:
                                win_line = [(lim_rows,lim_cols), (lim_rows + 1,lim_cols + 1), (lim_rows + 2,lim_cols + 2)]
                                return self.state[lim_rows + 1,lim_cols + 1], win_line
                            else:
                                return self.state[lim_rows + 1,lim_cols + 1]
                        if self.state[lim_rows,lim_cols + 2] == self.state[lim_rows + 1,lim_cols + 1] and self.state[lim_rows,lim_cols + 2] == self.state[lim_rows + 2,lim_cols]:
                            if return_line:
                                win_line = [(lim_rows,lim_cols + 2), (lim_rows + 1,lim_cols + 1), (lim_rows + 2,lim_cols)]
                                return self.state[lim_rows + 1,lim_cols + 1], win_line
                            else:
                                return self.state[lim_rows + 1,lim_cols + 1]
        if return_line:
            return -1, [((-1, -1), (-1, -1), (-1, -1))]
        else:
            return -1

    def availablePositions(self):       # Devolve uma lista de tuples com todas as positions onde é possivel jogar
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.state[i, j] < MAX_MOVES:
                    positions.append((i, j)) 
        return positions    

    def make_move (self, move_pos):     # Realiza move se este for possivel
        if self.state[move_pos] < MAX_MOVES:
            self.state[move_pos] += 1
            self.movesMade += 1
            return 1
        return 0

    def undo_move (self, move_pos):     # Desfaz move se este for possivel
        if self.state[move_pos] > 0:
            self.state[move_pos] -= 1
            self.movesMade -= 1
            return 1
        return 0

    def get_symmetry_mask(self, sym_num):
        sym = self.get_symmetry(sym_num)
        sym_to_return = []
        for i in range(len(sym)):
            sym_to_return.append((sym[i], i))
        return sym_to_return

    def get_canonic_score(self, board_flat = -1):
        current_state = np.copy(self.state)
        if type(board_flat) is int:
            ...
        else:
            board_state = self.get_array_from_flat(board_flat)
            self.set_state(board_state)

        score = 0
        flat_board = self.getFlat()
        for i in range(len(flat_board)):
            score += flat_board[i] * (10 ** (11 - i))
            #score *= flat_board[i] * self.primes[i]

        self.set_state(current_state)
        return score

    def get_symmetry(self, board_hash = -1, mask = False):
        all_symmetry = []
        if mask == False:
            current_state = np.copy(self.state)
            if board_hash == -1:
                ...
            else:
                board_flat = self.get_flat_from_hash(board_hash)
                board_state = self.get_array_from_flat(board_flat)
                self.set_state(board_state)


            all_symmetry.append(self.getFlat())
            
            self.state = np.flipud(self.state)
            all_symmetry.append(self.getFlat())
            
            self.state = np.fliplr(self.state)
            all_symmetry.append(self.getFlat())

            self.state = np.flipud(self.state)
            all_symmetry.append(self.getFlat())

            #self.state = np.fliplr(self.state)

            self.set_state(current_state)
        else:
            for sym_mask in self.symmetry_masks:
                all_symmetry.append()

        return all_symmetry

    def get_canonic_state(self, board_hash = -1):
        all_symmetry = self.get_symmetry(board_hash)
        all_canonic_score = [self.get_canonic_score(b) for b in all_symmetry]
        canonic_board_index = all_canonic_score.index(max(all_canonic_score))
        return all_symmetry[canonic_board_index], all_symmetry

    def get_canonic_state_mask(self, board_hash = -1):
        max_symmetry_score = -1
        best_symmetry_index = 0
        current_state = np.copy(self.state)

        if board_hash == -1:
                ...
        else:
            board_flat = self.get_flat_from_hash(board_hash)
            board_state = self.get_array_from_flat(board_flat)
            self.set_state(board_state)

        for sym_mask in self.symmetry_masks:
            if max_symmetry_score < np.tensordot(sym_mask[0], self.state):
                max_symmetry_score = np.tensordot(sym_mask[0], self.state)
                best_symmetry_index = sym_mask[1]
        
        if best_symmetry_index == 0:
            canonic_board = self.getFlat()
        elif best_symmetry_index == 1:
            self.state = np.flipud(self.state)
            canonic_board = self.getFlat()
        elif best_symmetry_index == 2:
            self.state = np.flipud(self.state)
            self.state = np.fliplr(self.state)
            canonic_board = self.getFlat()
        else:
            self.state = np.fliplr(self.state)
            canonic_board = self.getFlat()

        self.set_state(current_state)
        return canonic_board


    def showBoard(self):                # Imprime uma representação do board atual na consola
        for i in range(BOARD_ROWS):
            print(' ---' * BOARD_COLS)
            out = '| '
            for j in range(BOARD_COLS):
                out += str(int(self.state[i, j])) + ' | '
            print(out)
        print(' ---' * BOARD_COLS)
        print()

    def reset(self):                    # Reset ao estado inicial do board
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.turn = 1
        self.movesMade = 0

    def set_state(self, state):
        self.state = state

class Game:
    def __init__(self, p1, p2):
        self.board = Board()
        self.root = tk.Tk()
        #self.gui = VisualGame()
        self.p1 = p1
        self.p2 = p2
        self.done = False


    
    def play(self):
        while not self.done:
            # positions = self.board.availablePositions()
            positions = self.board.all_positions
            all_positions = self.board.all_positions
            if self.board.turn == 1:    # Player 1 play
                action = self.p1.choose_action(self.board)
                moveMade = self.board.make_move(action)

                # self.p2.addState(self.board.getHash())

                if moveMade == 0:   # Check in move tried was valid
                    return -1

                if self.board.check_win() != -1:    # Check win
                    #print("wins p1!")
                    self.done = True
                    #self.board.showBoard()
                    return -1

                # if type(self.p2) != type(Agent()):
                #     self.p1.addState(self.board.getHash())

            else:                       # Player 2 play
                #print("...............")
                #self.board.showBoard()
                self.p2.addState(self.board.getHash())
                
                action = self.p2.choose_action(self.board)
                moveMade = self.board.make_move(action)

                self.p2.addMove(action)
                # print(self.p2.states)
                # print(self.p2.moves)

                if moveMade == 0:   # Check in move tried was valid
                    return -1
                    
                if self.board.check_win() != -1:    # Check win
                    #print("wins p2!")
                    self.done = True
                    #self.p2.addState(self.board.getHash())
                    #self.board.showBoard()
                    return 1
            #self.board.showBoard()
            self.board.turn *= -1

    # def play_gui(self):
    #     self.root.geometry("410x350")
    #     VisualGame(self.root, self.p2).pack(fill="both", expand=True)
    #     self.root.mainloop()
    
    def reset(self):
        self.board.reset()
        self.done = False
    

# class Agent:
#     def __init__(self, name="___", epsilon=0.8, epsilon_rate=0.2, epsilon_min=0.02, lr=0.001, gamma=0.99, model = None):
#         self.name = name
#         self.states = []  # record all positions taken
#         self.moves = []
#         self.lr = lr
#         self.epsilon = epsilon
#         self.epsilon_drop_rate = 0
#         self.epsilon_rate = epsilon_rate
#         self.epsilon_min = epsilon_min
#         self.gamma = gamma
#         self.states_value = {}  # q-value for each explored state
#         self.all_rewards = []
#         self.all_rewards_means = []
#         self.epsilon_drop_rate = 0

#         self.test_mode = False
#         self.model = model

#     def set_test_mode (self, test_mode):
#         self.test_mode = test_mode

#     def choose_action(self, positions, current_board):
#         #print(self.test_mode)
#         if self.test_mode:                                  # Choose action while being tested
#             #print(current_board.all_positions)
#             action = self.predict(current_board.all_positions,current_board)
#         else:                                               # Choose action while being trained
#             if np.random.uniform(0, 1) <= self.epsilon:         # Random action
#                 idx = np.random.choice(len(current_board.all_positions))
#                 action = current_board.all_positions[idx]
#             else:                                               # Greedy action
#                 action = self.predict(current_board.all_positions,current_board)

#         return action

#     def predict(self, positions, current_board):
#         possible_state_values = 0
#         board_hash = current_board.getHash()
#         state_moves_values = self.states_value.get(board_hash)
#         #state_moves_values = (None if self.states_value.get(board_hash) is None else self.states_value.get(current_board.getHash()))
#         #print("........")
#         #current_board.showBoard()
#         #print(state_moves_values)
#         if type(state_moves_values) == type(None):
#             idx = np.random.choice(len(positions))
#         else:
#             #print("here")
#             state_moves_values_list = []
#             for p in positions:
#                 state_moves_values_list.append(state_moves_values[p])
#             #print(state_moves_values_list)
#             idx, value = max(enumerate(state_moves_values_list), key=operator.itemgetter(1))

#         return positions[idx]

#     def q_values(self, positions, current_board):
#         possible_state_values = []
        
#         for p in positions:
#             next_board = current_board.copy()
#             current_board.make_move(p)
#             next_boardHash = current_board.getHash()
#             possible_state_values.append(0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash))
#             current_board.undo_move(p)

#         return possible_state_values

#     def print_heatmap(self, state):
#         state.showBoard()
#         q_heatmap = self.q_values(state.all_positions, state.board)
#         print(q_heatmap)
#         a = np.zeros((BOARD_ROWS, BOARD_COLS))
#         for i in range(len(q_heatmap)):
#             if q_heatmap[i] == 0:
#                 continue
#             a[int(i/BOARD_COLS), i%BOARD_COLS] = q_heatmap[i][2]

#         #print (state.showBoard())
#         print(a)
        
#         plt.imshow(a, cmap='gist_heat', interpolation='nearest')
#         plt.clim(-1, 1)
#         plt.show()

#     def perfectMove(self, state):
#         best_move = self.states_value.get(state.getHash())
#         return (best_move[0], best_move[1])

#     # append a hash state
#     def addState(self, state):
#         self.states.append(state)

#     def addMove(self, move):
#         #print("here")
#         self.moves.append(move)
#         #print(len(self.moves))

#     # at the end of game, backpropagate and update states value
#     def feedReward(self, reward):
#         self.all_rewards.append(reward)
#         count = len(self.states) - 1
#         #print(len(self.moves))
#         #print(len(self.states))
#         for st in reversed(self.states):
#             if self.states_value.get(st) is None:
#                 self.states_value[st] = np.zeros((3, 2))
#             #print(type(st))
#             #print(self.moves[count])
#             self.states_value[st][self.moves[count]] += self.lr * (self.gamma * reward - self.states_value[st][self.moves[count]])
#             reward = np.max(self.states_value[st])
#             #reward = max(self.states_value[st].iteritems(), key=operator.itemgetter(1))[0]
#             count -= 1

#     def reset(self):
#         self.states = []
#         self.moves = []

#     def rand_stateValues(self, all_states):
#         for st in all_states:
#             self.states_value[st] = random()

#     def savePolicy(self, name=''):
#         if name == '':
#             name = self.name

#         fw = open('policy_' + str(name), 'wb')
#         pickle.dump(self.states_value, fw)
#         fw.close()

#     def loadPolicy(self, file):
#         fr = open(file, 'rb')
#         self.states_value = pickle.load(fr)
#         fr.close()

# class Player:
#     def __init__(self, _name="___", _player_type = "Random"):
#         self._name = _name
#         self._player_type = _player_type    # Random / Human / DQN
#         self.model = None
#         if self._player_type == "DQN":
#             self.model = DQN.load(_name)
#         if self._player_type == "Minimax":
#             fr = open("/home/alexandre/sem-project-logs/" + self._name, 'rb')
#             self.model = pickle.load(fr)
#             fr.close()
#         if self._player_type == "Q_learning":
#             fr = open("policy_q_learning", 'rb')
#             self.agent = Agent()
#             self.agent.states_value = pickle.load(fr)
#             fr.close()

#     def choose_action(self, positions, board = None, player = 1):  # Corrigir depois
#         if self._player_type == "Human":
#             while True:
#                 row = input("Input your action row:")
#                 col = input("Input your action col:")
#                 try:
#                     row = int(row)
#                     col = int(col)

#                     action = (row, col)
#                     if action in positions:
#                         return action

#                 except:
#                     print("Wrong move format")
#         elif self._player_type == "DQN":
#             action, _states = self.model.predict(board.get_one_hot())
#             action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
#         elif self._player_type == "Minimax":
#             minimax_move = Minimax.minimax_main_pruning_sym(board, 35, -100, 100, player, self.model)
#             action = (minimax_move[0], minimax_move[1])
#         elif self._player_type == "Q_learning":
#             action, _states = self.agent.choose_action(positions, board)
#             action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
#         elif self._player_type == "Monte_Carlo":
#             minimax_move = Minimax.minimax_main_pruning_sym(board, 35, -100, 100, 1, self.model)
#             action = (minimax_move[0], minimax_move[1])
#         else:
#             idx = np.random.choice(len(positions))
#             action = positions[idx]

#         return action

# class VisualGame(tk.Frame):
#     def __init__(self, parent, p2):
#         self.board = Board()
#         self.p2 = p2
#         self.parent = parent

#         tk.Frame.__init__(self, parent)

#         self.canvas = tk.Canvas(self)
#         self.canvas.create_line(100, 5, 100, 300, width=3)
#         self.canvas.create_line(200, 5, 200, 300, width=3)
#         self.canvas.create_line(300, 5, 300, 300, width=3)

#         self.canvas.create_line(5, 100, 400, 100, width=3)
#         self.canvas.create_line(5, 200, 400, 200, width=3)
#         self.canvas.create_rectangle(5, 5, 400, 300,
#             width=5)

#         self.canvas.pack(fill="both", expand=1)
#         self.canvas.bind("<Button-1>", self.player_move)

#         self.reset_b = tk.Button(text="Reset", command=self.reset)
#         self.reset_b.pack(side = "bottom")


#     def player_move(self, event):
#         x = int(event.x / 100)
#         y = int(event.y / 100)
#         print(x, y)
#         moveMade = self.board.make_move((y, x))
#         print(moveMade)
#         #if moveMade == 1:
#         self.atualizeVisual()
#         print(self.board.check_win())
        
#         self.waithere()

#         self.bot_move()
    
#     def bot_move(self):
#         time.sleep(0.5)
#         positions = self.board.availablePositions()
#         action = self.p2.choose_action(positions, self.board)
#         moveMade = self.board.make_move(action)
#         print(moveMade)
#         # if moveMade == 1:
#         #     self.atualizeVisual()
#         #     print(self.board.check_win())

#     def waithere(self):
#         var = 1
#         self.parent.after(1000, var, 1)
#         print("waiting...")
#         self.parent.wait_variable(var)

#     def atualizeVisual(self):
#         for i in range(3):
#             for j in range(4):
#                 if self.board.state[i, j] == 1:
#                     self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
#                         fill="green", width=3, tags="move")
#                 elif self.board.state[i, j] == 2:
#                     self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
#                         fill="yellow", width=3, tags="move")
#                 elif self.board.state[i, j] == 3:
#                     self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
#                         fill="red", width=3, tags="move") 
    
#     def reset(self):
#         self.board.reset()
#         self.canvas.delete("move")


board = Board()
for i in range(MAX_MOVES):
    board.make_move((0, 0))
board.showBoard()


