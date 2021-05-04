import numpy as np
import operator
import pickle
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate as interpld
import copy
import time
import random

import timeit

BOARD_ROWS = 3
BOARD_COLS = 4

class Board:                                
    def __init__(self):                     # Inicialização do board
        self.board = np.zeros((3, 4))
        self.boardHash = None
        self.turn = 1
        self.movesMade = 0
        self.all_positions = []
        for i in range(3):
            for j in range(4):
                self.all_positions.append((i, j))
        
                            
    def getHash(self):                  # Hash do board para usar como key num dic            
        self.boardHash = str(self.board.reshape(4 * 3))
        return self.boardHash

            
    def check_win (self):               # Verifica se existe um vencedor - return (-1) se não existir, outro numero se existir
        for j in range(4):
            if self.board[0,j] != 0:
                if self.board[0,j] == self.board[1,j] and self.board[0,j] == self.board[2,j]:
                    return self.board[0,j]
        
        for i in range(3):
            if self.board[i,1] != 0:
                if self.board[i,0] == self.board[i,1] and self.board[i,0] == self.board[i,2]:
                    return self.board[i,1]
                if self.board[i,1] == self.board[i,2] and self.board[i,1] == self.board[i,3]:
                    return self.board[i,1]

        if self.board[1,1] != 0:
            if self.board[0,0] == self.board[1,1] and self.board[0,0] == self.board[2,2]:
                return self.board[1,1]
            if self.board[0,2] == self.board[1,1] and self.board[0,2] == self.board[2,0]:
                return self.board[1,1]

        if self.board[1,2] != 0:
            if self.board[0,1] == self.board[1,2] and self.board[0,1] == self.board[2,3]:
                return self.board[1,2]
            if self.board[0,3] == self.board[1,2] and self.board[0,3] == self.board[2,1]:
                return self.board[1,2]

        return -1

    def availablePositions(self):       # Devolve uma lista de tuples com todas as positions onde é possivel jogar
        positions = []
        for i in range(3):
            for j in range(4):
                if self.board[i, j] < 1:
                    positions.append((i, j)) 
        return positions    

    def make_move (self, move_pos):     # Realiza move se este for possivel
        if self.board[move_pos] < 1:
            self.board[move_pos] += 1
            self.movesMade += 1
            return 1
        return 0

    def showBoard(self):                # Imprime uma representação do board atual na consola
        for i in range(0, 3):
            print('-----------------')
            out = '| '
            for j in range(0, 4):
                out += str(int(self.board[i, j])) + ' | '
            print(out)
        print('-----------------')
        print()

    def reset(self):                    # Reset ao estado inicial do board
        self.board = np.zeros((3, 4))
        self.boardHash = None
        self.turn = 1
        self.movesMade = 0


class Node:

    def __init__(self, state, depth, parent=None, player=None, score=0):
        self.state = state
        self.parent = parent
        self.player = player
        self.depth = depth
        self.score = score


# Representação de um confronto entre dois (2) jogadores. Vários jogos distintos podem ser jogados a partir deste mesmo confronto
class Game:                     
    def __init__(self, p1, p2):
        self.state = Board()
        self.p1 = p1
        self.p2 = p2
    
    # Atribuição de rewards aos dois (2) jogadores no final de um jogo.
    def giveReward(self, result):   
                
                # Vitoria p1
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(-1)

                # Vitoria p2
        elif result == -1:
            self.p1.feedReward(-1)
            self.p2.feedReward(1)

                # Empate
        else:
            self.p1.feedReward(-1)
            self.p2.feedReward(-1)

    # Treinar um agent com Q-learning
    def train(self, rounds=1000, progressive_lr=False):
        self.p1.all_rewards = []
        self.p2.all_rewards = []
        self.p1 = Player(epsilon=1)
        log_interval = 500

        a = self.p2.epsilon_min
        b = self.p2.epsilon
        c = rounds * self.p2.epsilon_rate
        d = np.power(a/b, 1/c)

        print(b)
        for i in range(rounds):
            #if p1.epsilon > p1.epsilon_min:
                #p1.epsilon *= 0.9999
            if p2.epsilon > p2.epsilon_min:
                p2.epsilon
                p2.epsilon *= d

            if i % log_interval == 0 and i != 0:
                print("Rounds {}".format(i))
                print("Players epsilon: {}".format(p2.epsilon))
                print("Bot epsilon: {}".format(p1.epsilon))
                mean_p1 = sum(p2.all_rewards[-log_interval:])/log_interval
                self.p2.all_rewards_means.append(mean_p1)
                print("Mean round_r: {}".format(mean_p1))
                #mean_ep_len = sum(self.all_ep_len[-log_interval:])/log_interval
                #print("Mean ep_len: {}".format(mean_ep_len))

                if progressive_lr == True:
                    mean_test = sum(p2.all_rewards[-5000:])/5000
                    if mean_test > 0.92:
                        self.p2.lr = 0.0000001
                    elif mean_test > 0.85:
                        self.p2.lr = 0.000001
                    elif mean_test > 0.5:
                        self.p2.lr = 0.00001
                    print(mean_test)
                print("Current learning-rate: {}".format(self.p2.lr))


                print()

            for ep_len in range(50):
                if self.state.turn == 1:
                    # Player 1
                    positions = self.state.availablePositions()
                    p1_action = self.p1.chooseAction(positions, self.state.board)
                    # take action and upate board state
                    self.state.make_move(p1_action)
                    board_hash = self.state.getHash()
                    self.p1.addState(board_hash)
                    # check board status if it is end

                elif self.state.turn == -1:
                    # Player 2
                    #positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(self.state.all_positions, self.state.board)
                    moveMade = self.state.make_move(p2_action)
                    self.p2.addMove(p2_action)
                    board_hash = self.state.getHash()
                    self.p2.addState(board_hash)

                    if moveMade == 0:
                        self.giveReward(1)
                        self.p1.reset()
                        self.p2.reset()
                        self.state.reset()
                        #self.all_ep_len.append(ep_len)
                        break

                if self.state.check_win() != -1:
                    # self.showBoard()
                    self.giveReward(self.state.turn)
                    print(len(self.p2.moves))
                    print("here1")
                    self.p1.reset()
                    self.p2.reset()
                    self.state.reset()
                    #self.all_ep_len.append(ep_len)
                    break

                if ep_len == 49:
                    self.giveReward(0)
                    self.p1.reset()
                    self.p2.reset()
                    self.state.reset()
                    #self.all_ep_len.append(ep_len)
                
                self.state.turn *= -1
        
    # Testar um agent contra um jogador que joga de forma aleatoria
    def test (self, rounds=1000):
        self.p1.all_rewards = []
        self.p2.all_rewards = []
        #rand_bot = Player(epsilon=1)
        log_interval = 1000
        for i in range(rounds):
            movesMade = []
            if i % log_interval == 0 and i != 0:
                print("Rounds {}".format(i))
                mean_p1 = sum(p2.all_rewards)/i
                print("Mean 1: {}".format(mean_p1))
            for ep_len in range(100):
                #self.showBoard()
                if self.state.turn == 1:
                    # Player 1
                    #positions = self.availablePositions()
                    #p1_action = self.p1.perfectMove(self.board)
                    positions = self.state.availablePositions()
                    p1_action = self.p1.chooseAction(positions, self.state.board)
                    # take action and upate board state
                    self.state.make_move(p1_action)
                    movesMade.append(p1_action)
                    # check board status if it is end

                    if self.state.check_win() != -1:
                        # self.showBoard()
                        # ended with p1 either win or draw

                        #self.showBoard()
                        #print(movesMade)

                        self.p1.all_rewards.append(1)
                        self.p1.reset()
                        self.p2.reset()
                        self.state.reset()
                        #print("Player1 won")
                        break

                elif self.state.turn == -1:
                    # Player 2
                    #positions = self.availablePositions()
                    #p2_action = self.p2.chooseAction(positions, self.board)
                    p2_action = self.p2.predict(self.state.all_positions, self.state.board)
                    self.state.make_move(p2_action)
                    movesMade.append(p2_action)
                    if self.state.check_win() != -1:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.p2.all_rewards.append(1)
                        self.p1.reset()
                        self.p2.reset()
                        self.state.reset()
                        #print("Rand_bot won")
                        break

                if ep_len == 49:
                    #self.giveReward(0)
                    self.p1.reset()
                    self.p2.reset()
                    self.state.reset()
                    #self.all_ep_len.append(ep_len)
                    print("here")
                    break
                
                self.state.turn *= -1

        mean_p1 = sum(p2.all_rewards)/rounds
        print("Win rate against random-bots: {}".format(mean_p1))
        return mean_p1

    def playHuman(self):
        depth = 35
        bestMoves = self.p1.states_value
        #bestValues = {}
        print(len(bestMoves))
        for ep_len in range(50):
            if self.state.turn == 1:
                # Player 2
                positions = self.state.availablePositions()
                p1_action = self.p2.chooseAction(positions, self.state.board)
                # p1_action = self.minimax(self.state, depth, -1)
                # p1_action = (p1_action[0], p1_action[1])
                # take action and upate board state

                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p2.name, "wins!")
                    self.state.reset()
                    break

            else:
                # Player 1
                #self.p1.print_heatmap(self.state)
                #p1_action = self.minimax(self.state, depth, 1, bestMoves)
                p1_action = self.minimax_simple(self.state, depth, 1)
                #p1_action = self.minimax_largura(self.state, 1, bestMoves)
                p1_action = (p1_action[0], p1_action[1])
                # positions = self.availablePositions()
                # p2_action = self.p1.chooseAction(positions)

                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p1.name, "wins!")
                    self.state.reset()
                    break
                
            self.state.turn *= -1
            #print(self.state.turn)
        return bestMoves

    def doubleLearn(self):
        depth = 35
        bestMoves = self.p1.states_value
        #bestValues = {}
        print(len(bestMoves))
        for ep_len in range(50):
            if self.state.turn == -1:
                # Player 2
                p1_action = self.minimax(self.state, depth, -1, bestMoves)
                p1_action = (p1_action[0], p1_action[1])
                # positions = self.availablePositions()
                # p2_action = self.p1.chooseAction(positions)

                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p2.name, "wins! 2")
                    self.state.reset()
                    break

            else:
                # Player 1
                p1_action = self.minimax(self.state, depth, 1, bestMoves)
                p1_action = (p1_action[0], p1_action[1])
                # positions = self.availablePositions()
                # p2_action = self.p1.chooseAction(positions)

                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p1.name, "wins! 1")
                    self.state.reset()
                    break
                
            self.state.turn *= -1
            #print(self.state.turn)
        return bestMoves

    def test_RL(self):
        depth = 9
        for ep_len in range(50):
            if self.state.turn == -1:
                # Player 2
                p1_action = self.p2.perfectMove(self.state)
                p1_action = (p1_action[0], p1_action[1])
                # take action and upate board state
                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p2.name, "wins! 2")
                    self.state.reset()
                    break

            else:
                # Player 1
                p1_action = self.p1.perfectMove(self.state)
                p1_action = (p1_action[0], p1_action[1])
                #positions = self.availablePositions()
                #p2_action = self.p1.chooseAction(positions)

                self.state.make_move(p1_action)
                #self.state.showBoard()

                if self.state.check_win() != -1:
                    print(self.p1.name, "wins! 1")
                    self.state.reset()
                    break
                
            self.state.turn *= -1
            #print(self.state.turn)
                
    def minimax(self, state, depth, player, board_nextMoves):       # Algoritmo Minimax em profundidade (com pŕe-pesquisa de um (1) depth em largura)
        if player == 1:
            best = [-1, -1, -10]
        else:
            best = [-1, -1, +10]

        if state.check_win() != -1:
            return [-1, -1, -player]

        # if depth == 0:
        #     return [-1, -1, 0]

        for pos in state.availablePositions():
            x, y = pos[0], pos[1]
            state.board[x, y] += 1
            if state.check_win() != -1:
                if depth == 35:
                    state.board[x, y] -= 1
                    return [x, y, player]
                else:
                    state.board[x, y] -= 1
                    return [-1, -1, player]
            state.board[x, y] -= 1

        for pos in state.availablePositions():
            score = [-1,-1,0]
            x, y = pos[0], pos[1]
            state.board[x, y] += 1
            standard_board_form, is_old = check_newBoard(state.board)
            if is_old == 0:
                standard_states.append(standard_board_form)

            if board_nextMoves.get(state.getHash()) == None:
                score = self.minimax(state, depth - 1, -player, board_nextMoves)
                score[2] *= 0.99
            else:
                score = board_nextMoves.get(state.getHash())
            state.board[x, y] -= 1
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

            if score[2] == player:
                board_nextMoves[state.getHash()] = best
                return best

        board_nextMoves[state.getHash()] = best
        return best

    def check_mainBoard(self, state, standard_states):
        value = [0, 0, 0, 0]

        #while value1 == value2 or value1 == value3 or value1 == value4 or value2 == value3 or value2 == value4 or value3 == value4:
        value[0] = self.board_value(state)
        value[1] = self.board_value(Inv_Vertical(state))
        value[2]= self.board_value(Inv_Horizontal(state))
        value[3] = self.board_value(Inv_Horizontal(Inv_Vertical(state)))

        idx, value = max(enumerate(value), key=operator.itemgetter(1))
        
        if idx == 0:
            return state
        if idx == 1:
            return Inv_Vertical(state)
        if idx == 2:
            return Inv_Horizontal(state)
        if idx == 3:
            return Inv_Horizontal(Inv_Vertical(state))


    def board_value(self, state):
        matrix_value = [13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        board_line = []
        for i in range(3):
            for j in range(4):
                board_line.append(board[i, j])

        total_value = 0
        for i in range(len(matrix_value)):
            total_value += board_line[i] * matrix_value[i]

        return total_value

    def Inv_Vertical(self, state):
        board = state.board
        temp_board = np.zeros((3, 4))

        for i in range(3):
            for j in range(4):
                x = ((j + 1) - 4) * (-1)

                temp_board[i, x] = board[i, j]
        
        return temp_board

    def Inv_Horizontal(self, state):
        board = state.board
        temp_board = np.zeros((3, 4))

        for i in range(3):
            for j in range(4):
                x = ((i + 1) - 3) * (-1)

                temp_board[x, j] = board[i, j]
        
        return temp_board

    def minimax_largura(self, state, player, board_nextMoves):      # Algoritmo Minimax em largura
        open_nodes = [Node(state, 0, player=player, score=-player)]
        best = (-1,-1)
        while open_nodes != []:
            node = open_nodes.pop(0)

            if node.state.check_win() != -1:
                #print("here")
                #print(node.state.board)
                node_temp = copy.deepcopy(node)

                if node.player == -player:
                    node.score = 1
                    
                else:
                    node.score = -1

                while node.depth != 0:
                    if node.parent.player == player:
                        if node.parent.score < node.score:
                            node.parent.score = node.score
                    else:
                        if node.parent.score > node.score:
                            node.parent.score = node.score

                    if node.depth == 1:
                        break
                        
                    node = node.parent
                    
                if node.score == player:
                    pos = node.state.board - node.parent.state.board
                    #print(pos)
                    for i in range(3):
                        for j in range(4):
                            if pos[i, j] == 1:
                                best = (i, j)
                                print(best)
                
                node = node_temp
                    
            else:
                for pos in node.state.availablePositions():
                    node_temp = copy.deepcopy(node)
                    node_temp.state.board[pos] += 1
                    if node_temp.state.check_win() != -1 and node.player == -player:
                        break
                    open_nodes.append(Node(node_temp.state, node.depth + 1, parent=node, player=(-1)*node.player, score=node.player))
                    #print(node.depth)

        return best

    def minimax_simple(self, state, depth, player):
        if player == 1:
            best = [-1, -1, -10]
        else:
            best = [-1, -1, +10]

        if state.check_win() != -1:
            return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        #if depth == 8:
        #    print("here")

        for pos in state.availablePositions():
            x, y = pos[0], pos[1]
            state.board[x, y] += 1
            score = self.minimax_simple(state, depth - 1, -player)
            score[2] *= 0.99
            state.board[x, y] -= 1
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        return best


class Player:
    def __init__(self, name="___", epsilon=0.8, epsilon_rate=0.2, epsilon_min=0.02, lr=0.00001, gamma=0.99, model = None):
        self.name = name
        self.states = []  # record all positions taken
        self.moves = []
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_drop_rate = 0
        self.epsilon_rate = epsilon_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.states_value = {}  # state -> value
        self.all_rewards = []
        self.all_rewards_means = []
        self.epsilon_drop_rate = 0

        self.model = model

    def getHash(self, board):
        boardHash = str(board.reshape(4 * 3))
        return boardHash

    def chooseAction(self, positions, current_board):
        if np.random.uniform(0, 1) <= self.epsilon:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            action = self.predict(positions,current_board)

        return action

    def predict(self, positions, current_board):

        possible_state_values = []
        
        for p in positions:
            next_board = current_board.copy()
            if next_board[p] < 3:
                next_board[p] += 1
            next_boardHash = self.getHash(next_board)
            possible_state_values.append(0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash))

        idx, value = max(enumerate(possible_state_values), key=operator.itemgetter(1))

        return positions[idx]

    def q_values(self, positions, current_board):
        possible_state_values = []
        
        for p in positions:
            next_board = current_board.copy()
            if next_board[p] < 3:
                next_board[p] += 1
            next_boardHash = self.getHash(next_board)
            possible_state_values.append(0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash))
        
        return possible_state_values

    def print_heatmap(self, state):
        state.showBoard()
        q_heatmap = self.q_values(state.all_positions, state.board)
        print(q_heatmap)
        a = np.zeros((3, 4))
        for i in range(len(q_heatmap)):
            if q_heatmap[i] == 0:
                continue
            a[int(i/4), i%4] = q_heatmap[i][2]

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
        print("here")
        self.moves.append(move)
        print(len(self.moves))

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        self.all_rewards.append(reward)
        count = len(self.states) - 1
        print(len(self.moves))
        print(len(self.states))
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = np.zeros((3, 4))
            self.states_value[st, [self.moves[count]]] += self.lr * (self.gamma * reward - self.states_value[st, [self.moves[count]]])
            reward = np.max(self.states_value[st])
            #reward = max(self.states_value[st].iteritems(), key=operator.itemgetter(1))[0]
            count -= 1

    def reset(self):
        self.states = []
        #self.moves = []

    def rand_stateValues(self, all_states):
        for st in all_states:
            self.states_value[st] = random()

    def savePolicy(self, name=''):
        if name == '':
            name = self.name

        fw = open('policy_' + str(name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board):
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

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass

def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

def savePolicy(states_value, name):
        fw = open('policy_' + str(name), 'wb')
        pickle.dump(states_value, fw)
        fw.close()




if __name__ == "__main__":
    # training
    #t0 = time.clock()

    #p2 = Player("p2")
    #p1.loadPolicy('policy_221')
    #p1 = Player(epsilon=1)

    #st = Game(p1, p2)
    test_results = []
    to_plot = {}
    for rounds in range(1):
        p1 = Player(epsilon=1)
        p2 = Player("p2")
        #p2.lr *= np.power(0.1, rounds+1)
        st = Game(p1, p2)
        print("training... " + str(rounds))
        st.train(500000)
        # #if rounds%2 == 1:
        # p2.lr = 0.0001
        # st.train(20000, progressive_lr=True)
        # #else:
        #     #st.train(20000)

    #     to_plot[rounds] = p2.all_rewards_means
    #     x = range(len(to_plot))
    #     to_plot = sp.polyfit(x, to_plot, deg=50)
    #     to_plot = sp.polyval(to_plot, x)
    #     if rounds == 0:
    #         plt.plot(to_plot, "b")
    #     elif rounds == 1:
    #         plt.plot(to_plot, "r")
    #     elif rounds == 2:
    #         plt.plot(to_plot, "g")
    #     elif rounds == 3:
    #         plt.plot(to_plot, "k")
    #     elif rounds == 4:
    #         plt.plot(to_plot, "y")
        
        print("testing... " + str(rounds))
        test_results.append(st.test(rounds=5000))
        print("------------")
        st.p2 = copy.deepcopy([p1])[0]

    #     #p2.lr *= 0.1
    # plt.show()
    # #p1.savePolicy()
    # p2.savePolicy()
    #plt.plot(test_results)
    #plt.show()

    # p1 = Player("p1")
    # p2 = Player(epsilon=1)
    # p3 = HumanPlayer("human")
    
    # num_games = 1000
    # time_m = 0
    # for i in range(num_games):
    #     p1 = Player("p1")
    #     p2 = Player(epsilon=1)
    #     st = Game(p1, p2)
        
    #     t0 = time.clock()
    #     #p1.loadPolicy('policy_212')
    #     policy = st.playHuman()
    #     time_m += time.clock() - t0
    #     print(time.clock() - t0)
    #     #savePolicy(policy, '252')

    # print(time_m/num_games)

    # p1.loadPolicy('policy_112')
    # st = Game(p1, p3)
    # while 1:
    #     rand = random.choice([1])
    #     temp_state = copy.deepcopy(st.state)
    #     while rand > 0:
    #         positions = st.state.availablePositions()
    #         botMove = p2.chooseAction(positions, st.state.board)
    #         moveDone = st.state.make_move((botMove[0], botMove[1]))

    #         rand -= 1
    #     if st.state.check_win() == -1:
    #         break 
    #     st.state = temp_state

    # st.state.showBoard()
    # q_heatmap = p1.q_values(st.state.all_positions, st.state.board)
    # print(q_heatmap)
    # a = np.zeros((3, 4))
    # for i in q_heatmap:
    #     if i == 0:
    #         continue
    #     a[i[0], i[1]] = i[2]

    # print (st.state.showBoard())
    # print(a)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()
        

    print(time.clock())
