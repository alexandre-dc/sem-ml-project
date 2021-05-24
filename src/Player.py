#from sem_game import 
import sem_game
from Minimax import Minimax
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
        if self._player_type == "Minimax":
            fr = open("/home/alexandre/sem-project-logs/minimax/" + self._name, 'rb')
            self.model = pickle.load(fr)
            self.minimax = Minimax("MMPS", 35, -100, 100, self.model, force_best_move=True)
            fr.close()
        if self._player_type == "Q-learning":
            fr = open("/home/alexandre/sem-project-logs/q_learning/" + self._name, 'rb')
            print(self._name)
            self.agent = Agent()
            self.agent.states_value = pickle.load(fr)
            print(self.agent.states_value)
            fr.close()

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
            action, _states = self.model.predict(board.get_one_hot())
            action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
        elif self._player_type == "Minimax":
            minimax_move = self.minimax.run_search(board, player)
            action = (minimax_move[0], minimax_move[1])
        elif self._player_type == "Q_learning":
            action, _states = self.agent.choose_action(board)
            action = (int(action / BOARD_COLS), int(action % BOARD_COLS))
        elif self._player_type == "Monte_Carlo":
            minimax_move = Minimax.minimax_main_pruning_sym(board, 35, -100, 100, 1, self.model)
            action = (minimax_move[0], minimax_move[1])
        else:
            idx = np.random.choice(len(positions))
            action = positions[idx]

        return action


print(1)