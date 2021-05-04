import numpy as np
from ttt import Board

error = 0
all_states = []

def get_symmetry(board):
    all_symmetry = []

    for i in range(0, 4):
        board.state = np.rot90(board.state, 1)
        all_symmetry.append(board.getHash())
        #print(board.state)

    board.state = np.flipud(board.state)
    all_symmetry.append(board.getHash())
    
    board.state = np.fliplr(board.state)
    all_symmetry.append(board.getHash())

    board.state = np.flipud(board.state)
    all_symmetry.append(board.getHash())

    for hash in all_symmetry:
        all_states.append(hash)
        #b.showBoard()

    board.state = np.fliplr(board.state)



def allGameStates (board):
    total = 0

    if board.movesMade == 10:
        return 0

    if board.getHash() in all_states:
        return 0
        

    if board.check_win() != 0:
        return 1

    for pos in board.availablePositions():
        if board.movesMade % 2 == 0:
            board.make_move(pos, 1)
        else:
            board.make_move(pos, -1)
        total += allGameStates(board)
        test = board.undo_move(pos)
        

    get_symmetry(board)
    #print (board.movesMade)
    if board.movesMade == 1:
        print(total)
        board.showBoard()

    return total


board = Board()
# board.make_move((0, 0), 1)
# board.make_move((1, 2), -1)
# get_symmetry(board)
# print([b for b in all_states])

print(allGameStates(board))
print(len(all_states))
