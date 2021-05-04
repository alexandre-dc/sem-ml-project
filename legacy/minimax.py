import numpy as np

# Versão atual do minimax em profundidade (com um nivel de largura)
def minimax_main(self, state, player, board_nextMoves):

    if player == 1:
        best = [-1, -1, -10]
    else:
        best = [-1, -1, +10]

    #if state.check_win() != -1:
    #    return [-1, -1, -player]

    if depth == 0:
        return [-1, -1, 0]

    for pos in state.availablePositions():      # Verificar se algum dos proximos moves dá vitoria imediata
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
        if board_nextMoves.get(state.getHash()) == None:
            score = self.minimax(state, depth - 1, -player, board_nextMoves)
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

        if score[2] == player:           # Verificar se esta arvore já têm uma jogada optima
            board_nextMoves[state.getHash()] = best
            return best

    if board_nextMoves.get(state.getHash()) == None:
        board_nextMoves[state.getHash()] = best     # Adicionar ao dicionario o melhor move para este estado do board
    
    return best

# Versão simplificada do minimax em profundidade (tempo de processamento extremamente altos)
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
        score = 0.95 * self.minimax(state, depth - 1, -player)
        state.board[x, y] -= 1
        score[0], score[1] = x, y

        if player == 1:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value

    return best


class Node:

    def __init__(self, state, depth, parent=None, player=None, score=0):
        self.state = state
        self.parent = parent
        self.player = player
        self.depth = depth
        self.score = score

# Versão inicial do minimax em largura
def minimax_largura_v3(self, state, player, board_nextMoves):
        open_nodes = [Node(state, 0, player=player, score=-player)]
        while open_nodes != []:
            node = open_nodes.pop(0)

            if node.state.check_win() != -1:
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
                    print(pos)
                    for i in range(3):
                        for j in range(4):
                            if pos[i, j] == 1:
                                return (i, j)
                
                node = node_temp
                    
            else:
                for pos in node.state.availablePositions():
                    node_temp = copy.deepcopy(node)
                    node_temp.state.board[pos] += 1
                    if node_temp.state.check_win() != -1 and node.player == -player:
                        break
                    open_nodes.append(Node(node_temp.state, node.depth + 1, parent=node, player=(-1)*node.player, score=node.player))
                    #print(node.depth)
            
        return (2,3)

def minimax_largura_v1(self, state, player, board_nextMoves):      # Algoritmo Minimax em largura
    open_nodes = [Node(state, 0, player=player)]
    while open_nodes != []:
        node = open_nodes.pop(0)
        
        if node.state.check_win() != -1:

            if node.player == -player:

                while node.parent.parent != None:
                    node = node.parent
                
                pos = node.state.board - node.parent.state.board
                print(pos)
                for i in range(3):
                    for j in range(4):
                        if pos[i, j] == 1:
                            return (i, j)
        else:
            for pos in node.state.availablePositions():
                node_temp = copy.deepcopy(node)
                node_temp.state.board[pos] += 1
                if node_temp.state.check_win() != -1 and node.player == -player:
                    break
                open_nodes.append(Node(node_temp.state, node.depth + 1, parent=node, player=(-1)*node.player))
                #print(node.depth)

# Uma outra tentativa de aplicar minimax em largura
def minimax_largura_v2(self, state, player, board_nextMoves):
        open_nodes = [Node(state, 0, player=player)]
        dead_end = False
        while open_nodes != []:
            node = open_nodes.pop(0)
            for pos in node.state.availablePositions():
                x, y = pos[0], pos[1]
                node_temp = copy.deepcopy(node)
                node_temp.state.board[x, y] += 1

                if node_temp.state.check_win() != -1:
                    print("here")
                    if node_temp.player == player:
                        print("here 1")

                        if node_temp.depth < 2:
                            while node_temp.parent.parent != None:
                                node_temp = node_temp.parent
                            

                            pos = node_temp.state.board - node_temp.parent.state.board
                            print(pos)
                            for i in range(3):
                                for j in range(4):
                                    if pos[i, j] == 1:
                                        return (i, j)
                        else:
                            return pos
                            
                    else:
                        dead_end = True
                        break
            
            if dead_end:
                dead_end = False
                continue

            for pos in node.state.availablePositions():
                x, y = pos[0], pos[1]
                node_temp = copy.deepcopy(node)
                node_temp.state.board[x, y] += 1
                #node.state.showBoard()
                open_nodes.append(Node(node_temp.state, node.depth + 1, parent=node, player=(-1)*node.player))

                #node.state.board[x, y] -= 1


# Primeira versão de minimax aplicada (não está em uso)
def old_minimax(board, IsMaximazing, depth, board_nextMoves, board_values, counter=0):
    bestMove = []
    if check_win(board) != -1:
        if IsMaximazing:
            return -1, counter
        else:
            return 1, counter

    if depth == 0:
        return 0, counter
        
    if IsMaximazing:
        max_v=-2
        for i in range(3):
            for j in range(3):
                if board[i,j] < 3:
                    board[i,j] += 1
                    #if board_nextMoves.get(getHash(board)) == None or type(board_nextMoves[getHash(board)]) != tuple:
                    if board_nextMoves.get(getHash(board)) == None:
                        value, counter = main_minimax(board, False, depth-1, board_nextMoves, board_values, counter)
                    else:
                        value = board_values.get(getHash(board))
                    if value > max_v:
                        max_v = value
                        bestMove = (i, j)
                    board[i,j] += -1

                    # alpha = max(alpha, value)
                    # if beta <= alpha:
                    #     break
        #print(bestMove)
        #print(board_nextMoves.get(getHash(board)))
        #print("------------")
        #if type(bestMove) == tuple:
        board_nextMoves[getHash(board)] = bestMove
        board_values[getHash(board)] = max_v
        
        return max_v, counter
    
    else:
        min_v=2
        for i in range(3):
            for j in range(3):
                if board[i,j] < 3:
                    board[i,j] += 1
                    #if board_nextMoves.get(getHash(board)) == None or type(board_nextMoves[getHash(board)]) != tuple:
                    if board_nextMoves.get(getHash(board)) == None:
                        value, counter = main_minimax(board, True, depth-1, board_nextMoves, board_values, counter)
                    else:
                        #print("here")
                        value = board_values.get(getHash(board))

                    if value < min_v:
                        min_v = value
                        bestMove = (i, j)

                    board[i,j] += -1

                    # beta = max(beta, value)
                    # if beta <= alpha:
                    #     break
        #print(bestMove)
        #print(board_nextMoves.get(getHash(board)))
        #print("------------")
        #if type(bestMove) == tuple:
        board_nextMoves[getHash(board)] = bestMove
        #print(board_nextMoves.get(getHash(board)))
        #print(type(board_nextMoves.get(getHash(board))) == tuple)
        board_values[getHash(board)] = min_v
        return min_v, counter





# Rotação e inversão matriz
def check_newBoard(state, standard_states):
    if Inv_Vertical(state) in standard_states:
        return Inv_Vertical(state)
    if Inv_Horizontal(state) in standard_states:
        return Inv_Horizontal(state)
    if Inv_Horizontal(Inv_Vertical(state)) in standard_states:
        return Inv_Horizontal(Inv_Vertical(state))
    
    return state


def Inv_Vertical(board):
    temp_board = np.zeros((3, 4))

    for i in range(3):
        for j in range(4):
            x = ((j + 1) - 4) * (-1)

            temp_board[i, x] = board[i, j]
    
    return temp_board

def Inv_Horizontal(board):
    temp_board = np.zeros((3, 4))

    for i in range(3):
        for j in range(4):
            x = ((i + 1) - 3) * (-1)

            temp_board[x, j] = board[i, j]
    
    return temp_board


































