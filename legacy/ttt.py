import numpy as np


class Board:                                
    def __init__(self):                     # Inicialização do board
        self.state = np.zeros((3, 3))
        self.boardHash = None
        self.turn = 1
        self.movesMade = 0
        self.all_positions = []
        for i in range(3):
            for j in range(3):
                self.all_positions.append((i, j))
        
                            
    def getHash(self):                  # Hash do board para usar como key num dic            
        self.boardHash = str(self.state.reshape(3 * 3))
        return self.boardHash

            
    def check_win (self):               # Verifica se existe um vencedor - return (-1) se não existir, outro numero se existir
        for i in range(3):
            if self.state[i,0] == self.state[i,1] and self.state[i,0] == self.state[i,2]:
                if self.state[i,0] != 0:
                    return int(self.state[i,0])
            if self.state[0,i] == self.state[1,i] and self.state[0,i] == self.state[2,i]:
                if self.state[0, i] != 0:
                    return int(self.state[0,i])
        
        if self.state[0,0] == self.state[1,1] and self.state[1,1] == self.state[2,2]:
            if self.state[1,1] != 0:
                return int(self.state[1,1])
        if self.state[2,0] == self.state[1,1] and self.state[1,1] == self.state[0,2]:
            if self.state[1,1] != 0:
                return int(self.state[1,1])

        if self.movesMade == 9:
            return 2

        return 0

    def availablePositions(self):       # Devolve uma lista de tuples com todas as positions onde é possivel jogar
        positions = []
        for pos in self.all_positions:
            if self.state[pos] == 0:
                positions.append(pos) 

        return positions    

    def make_move (self, move_pos, player):     # Realiza move se este for possivel
        if self.state[move_pos] == 0:
            self.state[move_pos] = player
            self.movesMade += 1
            return 1
        return 0

    def undo_move (self, move_pos):
        if self.state[move_pos] != 0:
            self.state[move_pos] = 0
            self.movesMade -= 1
            return 1
        return 0

    def showBoard(self):                # Imprime uma representação do board atual na consola
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if self.state[i, j] == 1:
                    out += "X" + ' | '
                elif self.state[i, j] == -1:
                    out += "O" + ' | '
                else:
                    out += " " + ' | '
                
            print(out)
        print('-------------')
        print()

    def reset(self):                    # Reset ao estado inicial do board
        self.state = np.zeros((3, 3))
        self.boardHash = None
        self.turn = 1
        self.movesMade = 0



class Game:
    def __init__(self, p1, p2):
        self.board = Board()
        self.p1 = p1
        self.p2 = p2

    def play_game(self):
        for _ in range(50):
            if self.board.turn == 1:
                position = self.board.availablePositions()
                a = self.p1.choose_action(position)
            else:
                position = self.board.availablePositions()
                a = self.p2.choose_action(position)

            self.board.make_move(a, self.board.turn)
            #self.board.showBoard()

            if self.board.check_win() != 0:
                print(str(self.board.turn) + " won!!")
                break

            if self.board.movesMade == 9:
                print("It's a draw!")
                break

            self.board.turn *= -1
            


class Player:
    def __init__(self, name="___", isHuman = False):
        self.name = name
        self.isHuman = isHuman

    def choose_action(self, positions):
        if self.isHuman:
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
        else:
            idx = np.random.choice(len(positions))
            action = positions[idx]

        return action


if __name__ == "__main__":
    #p1 = Player(isHuman=True)
    p1 = Player()
    p2 = Player()
    
    for i in range(10):
        game = Game(p1, p2)
        game.play_game()