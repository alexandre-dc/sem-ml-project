from small_sem import Board
from Minimax import Minimax

import pickle
import time

board = Board()
minimax = Minimax()
done = False
board_next_move = {}

t0 = time.clock()
while not done:
    if board.turn == 1:
        # minimax_move = minimax.minimax_simple(board, 35, 1)
        # minimax_move = minimax.minimax_simple_pruning(board, 35, -100, 100, 1)
        # minimax_move = minimax.minimax_main(board, 35, 1, board_next_move)
        # minimax_move = minimax.minimax_main_pruning(board, 35, -100, 100, 1, board_next_move)
        minimax_move = minimax.minimax_main_pruning_sym(board, 35, -100, 100, 1, board_next_move)
        move = (minimax_move[0], minimax_move[1])
        board.make_move(move) 

        if board.check_win() != -1:
            done = True
            print("Win 1")
    else:
        # minimax_move = minimax.minimax_simple(board, 35, -1)
        # minimax_move = minimax.minimax_simple_pruning(board, 35, -100, 100, -1)
        # minimax_move = minimax.minimax_main(board, 35, -1, board_next_move)
        # minimax_move = minimax.minimax_main_pruning(board, 35, -100, 100, -1, board_next_move)
        minimax_move = minimax.minimax_main_pruning_sym(board, 35, -100, 100, -1, board_next_move)
        move = (minimax_move[0], minimax_move[1])
        board.make_move(move) 

        if board.check_win() != -1:
            done = True
            print("Win 2")

    board.turn *= -1

    board.showBoard()

t1 = time.clock() - t0
print(t1)

fw = open('board_next_move' + str(''), 'wb')
pickle.dump(board_next_move, fw)
fw.close()

