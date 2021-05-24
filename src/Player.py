#from sem_game import 
import sem_game
from Minimax import Minimax
from Monte_Carlo import Monte_Carlo
from Agent import Agent
from stable_baselines import DQN

import numpy as np
import pickle

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

#minimax = Minimax("MMPS", 35, -100, 100, board_nextMoves, force_best_move=True)

class Player:
    def __init__(self, _name="___", _player_type = "Random"):
        self._name = _name
        self._player_type = _player_type    # Random / Human / DQN
        self.model = None
        if self._player_type == "DQN":
            self.model = DQN.load("/home/alexandre/sem-project-logs/dqn/" + self._name)
        elif self._player_type == "Minimax":
            fr = open("/home/alexandre/sem-project-logs/minimax/" + self._name, 'rb')
            self.model = pickle.load(fr)
            self.minimax = Minimax("MMPS", 35, -100, 100, self.model, force_best_move=True)
            fr.close()
        elif self._player_type == "Q-learning":
            fr = open("/home/alexandre/sem-project-logs/q_learning/" + self._name, 'rb')
            self.agent = Agent()
            self.agent.set_test_mode(True)
            print("here")
            self.agent.states_value = pickle.load(fr)
            print("here1")
            fr.close()
        elif self._player_type == "Monte Carlo":
            self.monte_carlo = Monte_Carlo()
            f_results_file = open("/home/alexandre/sem-project-logs/monte_carlo/" + self._name + "_f_results", 'rb')
            dcs_file = open("/home/alexandre/sem-project-logs/monte_carlo/" + self._name + "_dcs", 'rb')
            self.f_results = pickle.load(f_results_file)
            self.dcs = pickle.load(dcs_file)
            f_results_file.close()
            dcs_file.close()

    def choose_action(self, board = None, player = 1):  # Corrigir depois
        positions = board.availablePositions()
        if self._player_type == "Human":
            while True:
                row = input("Input your action row:")
                col = input("Input your action col:")
                try:
                    row = int(row)
                    col = int(col)

                    action = (row, col)
                    if action in positions:
                        return action

                except:
                    print("Wrong move format")
        elif self._player_type == "DQN":
            action, s= self.model.predict(board.get_one_hot())
            print(action)
            action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
        elif self._player_type == "Minimax":
            minimax_move = self.minimax.run_search(board, player)
            action = (minimax_move[0], minimax_move[1])
        elif self._player_type == "Q-learning":
            action = self.agent.choose_action(board)
            #action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
        elif self._player_type == "Monte Carlo":
            action = self.monte_carlo.run_test(self.f_results, self.dcs, player)
        else:
            idx = np.random.choice(len(positions))
            action = positions[idx]

        return action