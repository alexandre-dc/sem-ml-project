import numpy as np
import operator
import pickle


class Agent:
    def __init__(self, name="___", epsilon=0.8, epsilon_rate=0.2, epsilon_min=0.02, lr=0.0001, gamma=0.99, model = None):
        self.name = name
        self.states = []  # record all positions taken
        self.moves = []
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_drop_rate = 0
        self.epsilon_rate = epsilon_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.states_value = {}  # q-value for each explored state
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
        if self.test_mode:                                  # Choose action while being tested
            action = self.predict(positions, current_board)
        else:                                               # Choose action while being trained
            if np.random.rand() <= self.epsilon:         # Random action
                idx = np.random.choice(len(positions))
                action = positions[idx]
            else:                                               # Greedy action
                action = self.predict(positions,current_board)

        return action

    def predict(self, positions, current_board):
            # ....... Without canonic states ...............
        # possible_states_values = []
        # for p in positions:
        #     current_board.make_move(p)
        #     board_hash = current_board.getHash()
        #     if board_hash in self.states_value.keys():
        #         possible_states_values.append( self.states_value.get(board_hash) )
        #     else:
        #         possible_states_values.append( -2 )
        #     current_board.undo_move(p)
        # idx, value = max(enumerate(possible_states_values), key=operator.itemgetter(1))

            # ........ With canonic states ...............
        possible_states_values = []
        for p in positions:
            current_board.make_move(p)
            canonic_state, all_symmetry = current_board.get_canonic_state(current_board.getHash())
            key_state = str(canonic_state)
            if key_state in self.states_value.keys():
                possible_states_values.append( self.states_value.get(key_state) )
            else:
                possible_states_values.append( -2 )
            current_board.undo_move(p)
        #print(possible_states_values)
        idx, value = max(enumerate(possible_states_values), key=operator.itemgetter(1))

        # current_board.showBoard()
        # print(positions)
        #print(possible_states_values)
        # print(idx)
        # print(positions[idx])
        #print(len(self.states_value))
        #print(self.states_value.keys())
        #state_moves_values = (None if self.states_value.get(board_hash) is None else self.states_value.get(current_board.getHash()))
        #print("........")
        #current_board.showBoard()
        #print(state_moves_values)
        # if type(state_moves_values) == type(None):
        #     idx = np.random.choice(len(positions))
        # else:
        #     #print("here")
        #     state_moves_values_list = []
        #     for p in positions:
        #         state_moves_values_list.append(state_moves_values[p])
        #     #print(state_moves_values_list)
        #     idx, value = max(enumerate(state_moves_values_list), key=operator.itemgetter(1))

        return positions[idx]

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

        #print (state.showBoard())
        print(a)
        
        plt.imshow(a, cmap='gist_heat', interpolation='nearest')
        plt.clim(-1, 1)
        plt.show()

    def perfectMove(self, state):
        best_move = self.states_value.get(state.getHash())
        return (best_move[0], best_move[1])

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    def addMove(self, move):
        #print("here")
        self.moves.append(move)
        #print(len(self.moves))

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        self.all_rewards.append(reward)
        count = len(self.states) - 1
        #print(len(self.moves))
        #print(len(self.states))
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = np.zeros((3, 2))
            #print(type(st))
            #print(self.moves[count])
            self.states_value[st][self.moves[count]] += self.lr * (self.gamma * reward - self.states_value[st][self.moves[count]])
            reward = np.max(self.states_value[st])
            #reward = max(self.states_value[st].iteritems(), key=operator.itemgetter(1))[0]
            count -= 1

    def reset(self):
        self.states = []
        self.moves = []

    def rand_stateValues(self, all_states):
        for st in all_states:
            self.states_value[st] = random()

    def savePolicy(self, name=''):
        if name == '':
            name = self.name

        fw = open('/home/alexandre/sem-project-logs/q_learning/' + str(name) + 'st_values', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
        fw = open('/home/alexandre/sem-project-logs/q_learning/' + str(name) + 'dcs', 'wb')
        pickle.dump(self.dict_canonic_states, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open("/home/alexandre/sem-project-logs/q_learning/" + file + "st_values", 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
        fr = open("/home/alexandre/sem-project-logs/q_learning/" + file + "dcs", 'rb')
        self.dict_canonic_states = pickle.load(fr)
        fr.close()