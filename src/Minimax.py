from sem_game import Board
import sem_game

import numpy as np
import time
import pickle

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class Minimax:
    def __init__(self, _minimax_type, depth, alpha, beta, board_nextMoves, force_best_move = False):
            self._minimax_type = _minimax_type
            #self.board = board
            self.depth = depth
            self.alpha = alpha
            self.beta = beta
            self.board_nextMoves = board_nextMoves
            self.force_best_move = force_best_move
            self.check_loss = True
            self.value_loss = []

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

        # if depth == 34:
        #     print("- ", end="")

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
                # if self.force_best_move:
                #     score[2] *= 0.9
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
                #best.append(depth)

                return best

            # if depth == 35:
            #     self.value_loss.append(best[2])
            # if len(self.value_loss) == len(board.availablePositions()):
            #     print(self.value_loss)
            #     for v in self.value_loss:
            #         if self.value_loss[0] != v:
            #             self.check_loss = False
            #     print(self.check_loss)

            if self.force_best_move:
                #print("best move")
                if depth == 35:
                    self.value_loss.append(best[2])
                if len(self.value_loss) == len(board.availablePositions()):
                    #print(self.value_loss)
                    for v in self.value_loss:
                        if self.value_loss[0] != v:
                            self.check_loss = False
                    #print(self.check_loss)
                    if self.check_loss == True and self.value_loss[0] == -1:
                        #board.showBoard()
                        idx = np.random.choice(len(board.availablePositions()))
                        action = board.availablePositions()[idx]
                        best[0] = action[0]
                        best[1] = action[1]

        #best.append(depth)
        if board_nextMoves.get(board.getHash()) == None:
            board_nextMoves[board.getHash()] = best     # Adicionar ao dicionario o melhor move para este estado do board

        if depth == 35:
            self.check_loss = True
            self.value_loss = []

        #print(best)
        return best


            # Minimax with memory, 1 level breath searsh and alpha-beta pruning
    def minimax_main_pruning(self, board, depth, alpha, beta, player, board_nextMoves, force_best_move = False):       
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

        # if depth == 34:
        #     print("here")

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
                if self.force_best_move:
                    score[2] *= 0.9
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

        # if depth == 34:
        #     print("here")

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
                if self.force_best_move:
                    score[2] *= 0.9
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

        # if depth == 34:
        #     print("here")

        for pos in board.availablePositions():
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            #state.board[x, y] += 1
            score = self.minimax_simple_pruning(board, depth - 1, alpha, beta, -player)
            if self.force_best_move:
                score[2] *= 0.9
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

        # if depth == 34:
        #     print("here")

        for pos in board.availablePositions():
            x, y = pos[0], pos[1]
            board.make_move((x, y))
            #state.board[x, y] += 1
            score = self.minimax_simple(board, depth - 1, -player)
            if self.force_best_move:
                score[2] *= 0.9
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

    def run_search (self, board, player):
        if self._minimax_type == "MMPS":
            return self.minimax_main_pruning_sym(board, self.depth, self.alpha, self.beta, player, self.board_nextMoves)
        elif self._minimax_type == "MMP":
            return self.minimax_main_pruning(board, self.depth, self.alpha, self.beta, player, self.board_nextMoves)
        elif self._minimax_type == "MM":
            return self.minimax_main(board, self.depth, player, self.board_nextMoves)
        elif self._minimax_type == "MSP":
            return self.minimax_simple_pruning(board, self.depth, self.alpha, self.beta, player)
        elif self._minimax_type == "MS":
            return self.minimax_simple(board, self.depth, player)
        else:
            return 0

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
