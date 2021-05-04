from small_sem import Board
import numpy as np

all_states = []
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29, 31]

def get_canonic_score(board):
    score = 0
    for i in range(len(board)):
        score += board[i] * primes[i]
    return score

def get_symmetry(board):
    all_symmetry = []

    all_symmetry.append(board.getFlat())

    board.state = np.rot90(board.state, 2)
    all_symmetry.append(board.getFlat())
    #print(board.state)

    board.state = np.rot90(board.state, 2)
    
    board.state = np.flipud(board.state)
    all_symmetry.append(board.getFlat())
    
    board.state = np.fliplr(board.state)
    all_symmetry.append(board.getFlat())

    board.state = np.flipud(board.state)
    all_symmetry.append(board.getFlat())

    for hash in all_symmetry:
        all_states.append(hash)
        #b.showBoard()

    board.state = np.fliplr(board.state)

    return all_symmetry


def allGameStates (board):
    total = 0
    error_total = 0

    if board.check_win() >= 0:
        all_symmetry = get_symmetry(board)
        all_symmetry_scores = []
        for sym_board in all_symmetry:
            all_symmetry_scores.append(get_canonic_score(list(sym_board)))
        for i in range(len(all_symmetry_scores) - 1):
            for j in range(i + 1, len(all_symmetry_scores)):
                if all_symmetry_scores[i] == all_symmetry_scores[j]:
                    if all_symmetry[i].any() != all_symmetry[j].any():
                        print(all_symmetry[i])
                        print(all_symmetry[j])
                        error_total += 1
        return 1, error_total

    for pos in board.availablePositions():
        board.make_move(pos)
        r, error = allGameStates(board)
        total += r
        error_total += error
        board.undo_move(pos)
        

    #get_symmetry(board)
    #print (board.movesMade)
    if board.movesMade == 2:
        print(total)
        board.showBoard()
        print(get_canonic_score(board.getFlat()))
        print(error_total)

    all_symmetry = get_symmetry(board)
    all_symmetry_scores = []
    for sym_board in all_symmetry:
        all_symmetry_scores.append(get_canonic_score(list(sym_board)))
    for i in range(len(all_symmetry_scores) - 1):
        for j in range(i, len(all_symmetry_scores)):
            if all_symmetry_scores[i] == all_symmetry_scores[j]:
                if all_symmetry[i].any() != all_symmetry[j].any():
                    error_total += 1

    return total, error_total


board = Board()
print(allGameStates(board))