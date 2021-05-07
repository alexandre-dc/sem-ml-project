import tkinter as tk

from sem_game import Board
import sem_game
from Player import Player

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class VisualGame(tk.Frame):
    def __init__(self, parent):
        self.board = Board()
        self.p2 = Player(_name="/home/alexandre/sem-project-logs/100k_mm_sem3_3x4_dqn_32", _player_type="DQN")    # DQN
        #self.p2 = Player(_name="board_nextMoves", _player_type="Minimax")
        #print(self.p2.agent.states_value)
        #print(sorted(self.p2.agent.states_value))
        #print(len(self.p2.agent.states_value))
        #self.p2.agent.set_test_mode(True)

        tk.Frame.__init__(self, parent)

        self.canvas = tk.Canvas(self)
        self.canvas.create_line(100, 5, 100, 300, width=3)
        self.canvas.create_line(200, 5, 200, 300, width=3)
        self.canvas.create_line(300, 5, 300, 300, width=3)

        self.canvas.create_line(5, 100, 400, 100, width=3)
        self.canvas.create_line(5, 200, 400, 200, width=3)
        self.canvas.create_rectangle(5, 5, 400, 300,
            width=5)

        self.canvas.pack(fill="both", expand=1)
        self.canvas.bind("<Button-1>", self.makeMove)

        self.reset_b = tk.Button(text="Reset", command=self.reset)
        self.reset_b.pack(side = "bottom")

        self.bot_turn = -1
        if self.bot_turn == 1:
            self.bot_move()


    def makeMove(self, event):
        if self.board.check_win() == -1:
            x = int(event.x / 100)
            y = int(event.y / 100)
            print(x, y)
            moveMade = self.board.make_move((y, x))
            print(moveMade)
            if moveMade == 1:
                self.atualizeVisual()
                print(self.board.check_win())
                
                self.bot_move()
        
        

        

    def bot_move(self):
        positions = self.board.availablePositions()
        action = self.p2.choose_action(positions, self.board, player = self.bot_turn)
        print(action)
        moveMade = self.board.make_move(action)
        print(moveMade)
        if moveMade == 1:
            self.atualizeVisual()
            print(self.board.check_win())

    def atualizeVisual(self):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board.state[i, j] == 1:
                    self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                        fill="green", width=3, tags="move")
                elif self.board.state[i, j] == 2:
                    self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                        fill="yellow", width=3, tags="move")
                elif self.board.state[i, j] == 3:
                    self.canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                        fill="red", width=3, tags="move") 
    
    def reset(self):
        self.board.reset()
        self.canvas.delete("move")

        if self.bot_turn == 1:
            self.bot_move()

                
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("410x350")
    VisualGame(root).pack(fill="both", expand=True)
    root.mainloop()