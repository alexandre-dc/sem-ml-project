from sem_game import Board
from Minimax import Minimax

import pickle
import time

board = Board()
board_nextMoves = {}
minimax = Minimax("MMPS", 35, -100, 100, board_nextMoves, force_best_move=True)      #           MMPS            /        MMP        /    MM     /        MSP           /      MS
done = False                                                                                # Main_Prunning_Symmetric   /   Main_Prunning   /   Main    /   Simple_Prunning    /    Simple

t0 = time.clock()
while not done:
    if board.turn == 1:
        minimax_move = minimax.run_search(board, 1)
        move = (minimax_move[0], minimax_move[1])
        board.make_move(move)

        if board.check_win() != -1:
            done = True
            print("Win 1")
    else:
        minimax_move = minimax.run_search(board, -1)
        move = (minimax_move[0], minimax_move[1])
        board.make_move(move) 

        if board.check_win() != -1:
            done = True
            print("Win 2")

    board.turn *= -1

    board.showBoard()

t1 = time.clock() - t0
print(t1)

fw = open('/home/alexandre/sem-project-logs/minimax/board_nextMoves_3_4_3_mmps' + str(''), 'wb')
pickle.dump(board_nextMoves, fw)
fw.close()

