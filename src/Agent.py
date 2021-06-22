import numpy as np
import operator
import pickle


class Agent:
    def __init__(self, name="___", epsilon=0.8, epsilon_rate=0.2, epsilon_min=0.02, lr=0.0001, gamma=0.99, model = None):
        self.name = name
        self.states = []
        self.moves = []
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_drop_rate = 0
        self.epsilon_rate = epsilon_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.states_value = {}
        self.dict_canonic_states = {}
        self.all_rewards = []
        self.all_rewards_means = []
        self.epsilon_drop_rate = 0

        self.test_mode = False
        self.model = model

    def set_test_mode (self, test_mode):
        self.test_mode = test_mode

    def choose_action(self, current_board):
        positions = current_board.availablePositions()
        if self.test_mode:                                
            action = self.predict(positions, current_board)
        else:                                              
            if np.random.rand() <= self.epsilon:        
                idx = np.random.choice(len(positions))
                action = positions[idx]
            else:                                         
                action = self.predict(positions,current_board)

        return action

    def predict(self, positions, current_board):
        all_positions = current_board.all_positions


        board_hash = current_board.getHash()
        state_moves_values = (None if self.states_value.get(board_hash) is None else self.states_value.get(current_board.getHash()))

        if type(state_moves_values) == type(None):
            idx = np.random.choice(len(all_positions))
        else:
            idx, value = max(enumerate(state_moves_values), key=operator.itemgetter(1))
        return all_positions[idx]

    def q_values(self, positions, current_board):
        possible_state_values = []
        
        for p in positions:
            next_board = current_board.copy()
            current_board.make_move(p)
            next_boardHash = current_board.getHash()
            possible_state_values.append(0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash))
            current_board.undo_move(p)

        return possible_state_values

    def print_heatmap(self, state):
        state.showBoard()
        q_heatmap = self.q_values(state.all_positions, state.board)
        print(q_heatmap)
        a = np.zeros((BOARD_ROWS, BOARD_COLS))
        for i in range(len(q_heatmap)):
            if q_heatmap[i] == 0:
                continue
            a[int(i/BOARD_COLS), i%BOARD_COLS] = q_heatmap[i][2]
        
        plt.imshow(a, cmap='gist_heat', interpolation='nearest')
        plt.clim(-1, 1)
        plt.show()

    def perfectMove(self, state):
        best_move = self.states_value.get(state.getHash())
        return (best_move[0], best_move[1])

    def addState(self, state):
        self.states.append(state)

    def addMove(self, move):
        self.moves.append(move)

    def reset(self):
        self.states = []
        self.moves = []

    def rand_stateValues(self, all_states):
        for st in all_states:
            self.states_value[st] = random()

    def savePolicy(self, name=''):
        if name == '':
            name = self.name

        fw = open('/home/alexandre/sem-project-logs/q_learning/' + str(name) + '_st_values', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
        fw = open('/home/alexandre/sem-project-logs/q_learning/' + str(name) + '_dcs', 'wb')
        pickle.dump(self.dict_canonic_states, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open("/home/alexandre/sem-project-logs/q_learning/" + file + "_st_values", 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
        fr = open("/home/alexandre/sem-project-logs/q_learning/" + file + "_dcs", 'rb')
        self.dict_canonic_states = pickle.load(fr)
        fr.close()