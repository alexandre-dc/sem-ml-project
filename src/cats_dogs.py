import numpy as np
import operator
import pickle
import matplotlib.pyplot as plt
import copy
import time


BOARD_ROWS = 4
BOARD_COLS = 4


class Board:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.turn = 1
        self.moves = 0
        self.all_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.all_positions.append((i, j))
        
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def permitedPos(self, player):
        central_pos = [1, 2]
        posList = []
        if self.moves == 0:
            posList.append((central_pos[0], central_pos[0]))
            posList.append((central_pos[1], central_pos[0]))
            posList.append((central_pos[0], central_pos[1]))
            posList.append((central_pos[1], central_pos[1]))

            return posList

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    if i+1 >= BOARD_ROWS or self.board[i+1, j] != -player:
                        if i-1 < 0 or self.board[i-1, j] != -player:
                            if j+1 >= BOARD_ROWS or self.board[i, j+1] != -player:
                                if j-1 < 0 or self.board[i, j-1] != -player:
                                    if self.moves == 1:
                                        if (i, j) != (central_pos[0], central_pos[0]) and (i, j) != (central_pos[1], central_pos[0]):
                                            if (i, j) != (central_pos[0], central_pos[1]) and (i, j) != (central_pos[1], central_pos[1]):
                                                posList.append((i, j))
                                    else:
                                        posList.append((i, j))
        
        return posList
                

    def check_win (self, player):
        p1 = self.permitedPos(1)
        p2 = self.permitedPos(-1)

        for pos in p1:
            if pos in p2:
                return 0
        
        if len(p1) > len(p2):
            return 1

        if len(p1) < len(p2):
            return -1

        return -player

    def make_move (self, move_pos, player):
        if move_pos in self.permitedPos(player):
            self.board[move_pos] = player
            self.moves += 1
            return 1
        return 0

    def showBoard(self):
        for i in range(0, BOARD_ROWS):
            print('-----------' * 3)
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(int(self.board[i, j])) + ' | '
            print(out)
        print('-----------' * 3)
        print()

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.turn = 1

    


class Game:
    def __init__(self, p1, p2):
        self.state = Board()
        self.p1 = p1
        self.p2 = p2
        self.counter = 0
        self.bestMoves = {}

    def play_HvH(self):
        for ep_len in range(50):
            win = self.state.check_win(self.state.turn)
            if win != 0:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print(self.p2.name, "wins!")
                self.state.reset()
                break

            if self.state.turn == 1:
                # Player 1
                positions = self.state.permitedPos(self.state.turn)
                p1_action = self.p1.chooseAction(positions)

                self.state.make_move(p1_action, self.state.turn)

            else:
                # Player 2
                positions = self.state.permitedPos(self.state.turn)
                p2_action = self.p2.chooseAction(positions)

                self.state.make_move(p2_action, self.state.turn)
                

            self.state.showBoard()  
                
            self.state.turn *= -1
            print(self.state.turn)

    def testMM(self):
        depth = 7
        best = self.minimax(self.state.board, depth, self.state.turn)
        for ep_len in range(50):
            print(bestMoves)
            self.counter = 0
            win = self.state.check_win(self.state.turn)
            if win != 0:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print(self.p2.name, "wins!")
                self.state.reset()
                break

            if self.state.turn == 1:
                
                p1_action = self.minimax(self.state, depth, self.state.turn, self.bestMoves)
                #p1_action = bestMoves.get(self.state.getHash())
                p1_action = (p1_action[0], p1_action[1])

                print(p1_action)

                self.state.make_move(p1_action, self.state.turn)

            else:
                # Player 2
                p1_action = self.minimax(self.state, depth, self.state.turn, bestMoves)
                #p1_action = bestMoves.get(self.state.getHash())
                p1_action = (p1_action[0], p1_action[1])

                print(p1_action)

                self.state.make_move(p1_action, self.state.turn)
                
            self.state.showBoard()
            self.state.turn *= -1
            print(self.state.turn)
            print(ep_len)
    
    def minimax(self, state, depth, player):
        if player == 1:
            best = [-1, -1, -10]
        else:
            best = [-1, -1, +10]

        if state.check_win(player) != 0:
            return [-1, -1, -state.check_win(player)]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 6:
            self.counter += 1
            print("Iter: " + str(self.counter))

        for pos in state.permitedPos(player):
            x, y = pos[0], pos[1]
            state[x, y] = player
            score = minimax(state, depth - 1, -player)
            # if bestMoves.get(state.getHash()) == None:
            # score, bestMoves = minimax(state, depth - 1, -player, bestMoves)
            # else:
            #     score = bestMoves.get(self.state.getHash())
            state[x, y] = 0
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        print(bestMoves)
        #bestMoves[state.getHash()] = best
        return best

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass





p1 = HumanPlayer("p1")
p2 = HumanPlayer("p2")

st = Game(p1, p2)
for i in range(1):
    st.testMM()