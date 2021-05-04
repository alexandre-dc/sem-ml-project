from small_sem import Board
import small_sem

import numpy as np
import time
import pickle

BOARD_ROWS = small_sem.BOARD_ROWS
BOARD_COLS = small_sem.BOARD_COLS
MAX_MOVES = small_sem.MAX_MOVES

class Minimax:
    def __init__(self):
            pass
    def minimax_main_pruning_sym(self, board, depth, alpha, beta, player, board_nextMoves):       # Minimax with memory, 1 level breath searsh, alpha-beta pruning and symmetries
        if player == 1:
            best = [-1, -1, -10]
            value = -100
        else:
            best = [-1, -1, +10]
            value = 100

        # if board.check_win() != -1:
        #     return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 34:
            print("here")

        for pos in board.availablePositions():      # Verificar se algum dos proximos moves dá vitoria imediata
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            if board.check_win() != -1:
                if depth == 35:
                    board.undo_move((x, y))
                    return [x, y, player]
                else:
                    board.undo_move((x, y))
                    return [-1, -1, player]
            board.undo_move((x, y))

        for pos in board.availablePositions():
            score = [-1,-1,0]
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            for b_flat in board.get_symmetry():
                b_hash = board.get_array_from_flat(b_flat)
                b_hash = str(b_hash.reshape(BOARD_ROWS * BOARD_COLS))
                if board_nextMoves.get(b_hash) != None:
                    score = board_nextMoves.get(b_hash)
                    break
            else:
                score = self.minimax_main_pruning_sym(board, depth - 1, alpha, beta, -player, board_nextMoves)
                score[2] *= 0.9
            board.undo_move((x, y))
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
                alpha = max(value, best[2])
                if alpha >= beta:
                    break
            else:
                if score[2] < best[2]:
                    best = score  # min value
                beta = min(value, best[2])
                if alpha >= beta:
                    break

            if score[2] == player:           # Verificar se esta arvore já têm uma jogada optima
                board_nextMoves[board.getHash()] = best
                return best

        if board_nextMoves.get(board.getHash()) == None:
            board_nextMoves[board.getHash()] = best     # Adicionar ao dicionario o melhor move para este estado do board

        return best


            # Minimax with memory, 1 level breath searsh and alpha-beta pruning
    def minimax_main_pruning(self, board, depth, alpha, beta, player, board_nextMoves):       
        if player == 1:
            best = [-1, -1, -10]
            value = -100
        else:
            best = [-1, -1, +10]
            value = 100

        # if board.check_win() != -1:
        #     return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 34:
            print("here")

        for pos in board.availablePositions():      # Verificar se algum dos proximos moves dá vitoria imediata
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            if board.check_win() != -1:
                if depth == 35:
                    board.undo_move((x, y))
                    return [x, y, player]
                else:
                    board.undo_move((x, y))
                    return [-1, -1, player]
            board.undo_move((x, y))

        for pos in board.availablePositions():
            score = [-1,-1,0]
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            if board_nextMoves.get(board.getHash()) == None:
                score = self.minimax_main_pruning(board, depth - 1, alpha, beta, -player, board_nextMoves)
                #score[2] *= 0.9
            else:
                score = board_nextMoves.get(board.getHash())
            board.undo_move((x, y))
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
                alpha = max(value, best[2])
                if alpha >= beta:
                    break
            else:
                if score[2] < best[2]:
                    best = score  # min value
                beta = min(value, best[2])
                if alpha >= beta:
                    break

            if score[2] == player:           # Verificar se esta arvore já têm uma jogada optima
                board_nextMoves[board.getHash()] = best
                return best

        if board_nextMoves.get(board.getHash()) == None:
            board_nextMoves[board.getHash()] = best     # Adicionar ao dicionario o melhor move para este estado do board

        return best

    def minimax_main(self, board, depth, player, board_nextMoves):

        if player == 1:
            best = [-1, -1, -10]
        else:
            best = [-1, -1, +10]

        # if board.check_win() != -1:
        #     return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 34:
            print("here")

        for pos in board.availablePositions():      # Verificar se algum dos proximos moves dá vitoria imediata
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            if board.check_win() != -1:
                if depth == 35:
                    board.undo_move((x, y))
                    return [x, y, player]
                else:
                    board.undo_move((x, y))
                    return [-1, -1, player]
            board.undo_move((x, y))

        for pos in board.availablePositions():
            score = [-1,-1,0]
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            if board_nextMoves.get(board.getHash()) == None:
                score = self.minimax_main(board, depth - 1, -player, board_nextMoves)
                #score[2] *= 0.9
            else:
                score = board_nextMoves.get(board.getHash())
            board.undo_move((x, y))
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

            if score[2] == player:           # Verificar se esta arvore já têm uma jogada optima
                board_nextMoves[board.getHash()] = best
                return best

        if board_nextMoves.get(board.getHash()) == None:
            board_nextMoves[board.getHash()] = best     # Adicionar ao dicionario o melhor move para este estado do board

        return best

    def minimax_simple_pruning(self, board, depth, alpha, beta, player):
        if player == 1:
            best = [-1, -1, -10]
            value = -100
        else:
            best = [-1, -1, +10]
            value = 100

        if board.check_win() != -1:
            return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 34:
            print("here")

        for pos in board.availablePositions():
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            #state.board[x, y] += 1
            score = self.minimax_simple_pruning(board, depth - 1, alpha, beta, -player)
            #score[2] *= 0.9
            #print(score)
            #state.board[x, y] -= 1
            board.undo_move((x, y))
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
                alpha = max(value, best[2])
                if alpha >= beta:
                    break
            else:
                if score[2] < best[2]:
                    best = score  # min value
                beta = min(value, best[2])
                if alpha >= beta:
                    break

        return best

    def minimax_simple(self, board, depth, player):
        if player == 1:
            best = [-1, -1, -10]
        else:
            best = [-1, -1, +10]

        if board.check_win() != -1:
            return [-1, -1, -player]

        if depth == 0:
            return [-1, -1, 0]

        if depth == 34:
            print("here")

        for pos in board.availablePositions():
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            #state.board[x, y] += 1
            score = self.minimax_simple(board, depth - 1, -player)
            #score[2] *= 0.9
            #print(score)
            #state.board[x, y] -= 1
            board.undo_move((x, y))
            score[0], score[1] = x, y

            if player == 1:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        return best

# board = Board()
# done = False
# board_next_move = {}

# t0 = time.clock()
# while not done:
#     if board.turn == 1:
#         # minimax_move = minimax_simple(board, 35, 1)
#         # minimax_move = minimax_simple_pruning(board, 35, -100, 100, 1)
#         # minimax_move = minimax_main(board, 35, 1, board_next_move)
#         # minimax_move = Minimax().minimax_main_pruning(board, 35, -100, 100, 1, board_next_move)
#         minimax_move = Minimax().minimax_main_pruning_sym(board, 35, -100, 100, 1, board_next_move)
#         move = (minimax_move[0], minimax_move[1])
#         board.make_move(move) 

#         if board.check_win() != -1:
#             done = True
#             print("Win 1")
#     else:
#         # minimax_move = minimax_simple(board, 35, -1)
#         # minimax_move = minimax_simple_pruning(board, 35, -100, 100, -1)
#         # minimax_move = minimax_main(board, 35, -1, board_next_move)
#         # minimax_move = Minimax().minimax_main_pruning(board, 35, -100, 100, -1, board_next_move)
#         minimax_move = Minimax().minimax_main_pruning_sym(board, 35, -100, 100, -1, board_next_move)
#         move = (minimax_move[0], minimax_move[1])
#         board.make_move(move) 

#         if board.check_win() != -1:
#             done = True
#             print("Win 2")

#     board.turn *= -1

#     board.showBoard()

# t1 = time.clock() - t0
# print(t1)

# fw = open('board_next_move' + str(''), 'wb')
# pickle.dump(board_next_move, fw)
# fw.close()
