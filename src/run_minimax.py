import sem_game
from sem_game import Board
from Minimax import Minimax

import pickle
import time

board = Board()
# list_alg = ["MM", "MMP", "MMPS"]
list_alg = ["MMPS"]
logs = {}
#minimax = Minimax("MMPS", 35, -100, 100, board_nextMoves, force_best_move=True)      #           MMPS            /        MMP        /    MM     /        MSP           /      MS
                                                                                      # Main_Prunning_Symmetric   /   Main_Prunning   /   Main    /   Simple_Prunning    /    Simple
for _ in range(1):
    for alg in list_alg:
        for i in range(3, 4):
            board_nextMoves = {}
            minimax = Minimax(alg, 35, -100, 100, board_nextMoves, force_best_move=True)
            done = False  
            board.reset()
            sem_game.MAX_MOVES = i
            print("here")
            t0 = time.clock()
            while not done:
                if board.turn == 1:
                    minimax_move = minimax.run_search(board, 1)
                    move = (minimax_move[0], minimax_move[1])
                    board.make_move(move)

                    if board.check_win() != -1:
                        done = True
                        print("Win 1")
                        board.showBoard()
                else:
                    minimax_move = minimax.run_search(board, -1)
                    move = (minimax_move[0], minimax_move[1])
                    board.make_move(move) 

                    if board.check_win() != -1:
                        done = True
                        print("Win 2")
                        board.showBoard()

                board.turn *= -1

                print()
                

            t1 = time.clock() - t0
            #print()
            print(alg + "_" + str(i))
            print(t1)
            if alg + str(i) in logs:
                logs[alg + str(i)].append(t1)
            else:
                logs[alg + str(i)] = [t1]

            fw = open('/home/alexandre/sem-project-logs/minimax/board_nextMoves_' + str(sem_game.MAX_MOVES) + "_" + str(sem_game.BOARD_ROWS) + "x" + str(sem_game.BOARD_COLS) + "_" + alg, 'wb')
            pickle.dump(board_nextMoves, fw)
            fw.close()

for key in logs.keys():
    logs[key].append (sum(logs[key]) / len(logs[key]))
    print(logs[key])
