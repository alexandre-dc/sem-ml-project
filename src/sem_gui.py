import tkinter as tk
import numpy as np
from tkinter import font

from sem_game import Board
import sem_game
from Player import Player

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

class VisualGame(tk.Frame):
    def __init__(self, parent):
        self.frame = tk.Frame.__init__(self, parent)
        self.root = parent
        

        #self.p2 = Player(_name="/home/alexandre/sem-project-logs/100k_mm_sem3_3x4_dqn_32", _player_type="DQN")    # DQN
        #self.p2 = Player(_name="board_nextMoves_3_4_3_b", _player_type="Minimax")
        #print(self.p2.agent.states_value)
        #print(sorted(self.p2.agent.states_value))
        #print(len(self.p2.agent.states_value))
        #self.p2.agent.set_test_mode(True)

        self.canvas = tk.Canvas(self, bg= "grey")
        self.canvas.create_line(100, 5, 100, 300, width=3)
        self.canvas.create_line(200, 5, 200, 300, width=3)
        self.canvas.create_line(300, 5, 300, 300, width=3)

        self.canvas.create_line(5, 100, 400, 100, width=3)
        self.canvas.create_line(5, 200, 400, 200, width=3)
        self.canvas.create_rectangle(5, 5, 400, 300,
            width=5)

        self.canvas.pack(fill="both", expand=1)
        self.canvas.bind("<Button-1>", self.makeMove)

        self.tkvar = tk.StringVar(root)
        self.choices = { 'Minimax','Monte Carlo','Q-learning','DQN'}
        self.tkvar.set('Minimax') # set the default option

        popupMenu = tk.OptionMenu(root, self.tkvar, *self.choices)
        # tk.Label(root, text="Choose a dish")
        popupMenu.pack(side="bottom")

        self.reset_b = tk.Button(text="Reset", command=self.reset)
        self.reset_b.pack(side = "bottom")

        #self.txt_frame = tk.Frame(root, )
        self.winner_txt = tk.Label(self.canvas, height=1, width=50, bg="grey")
        self.winner_txt.config(font=("Arial",15))
        self.winner_txt.pack(side="bottom", padx=0, pady=15)
        #self.winner_txt.place(x = 200, y = 400)

        self.board = Board()
        self.bot_turn = -1
        #.bot_type = "Q-learning"

        # if self.bot_type == "DQN":
        #     self.p2 = Player(_name="200k_mm_sem1_3x4_32", _player_type="DQN")
        # elif self.bot_type == "Minimax":
        #     self.p2 = Player(_name="board_nextMoves_3_4_3_mmps", _player_type="Minimax")
        # elif self.bot_type == "Q-Learning":
        #     self.p2 = Player(_name="policy_sem1_3_2_20k", _player_type="Q-Learning")

        self.reset()
        

    def makeMove(self, event):
        print(self.bot_type)
        if self.check_win() == -1 and self.bot_type != None:
            y = int(event.x / 100)
            x = int(event.y / 100)
            print(x, y)
            moveMade = self.board.make_move((x, y))
            print(moveMade)
            if moveMade == 1:
                self.update_visual()
                print(self.check_win(return_line=True))
                if self.board.check_win() == -1:
                    self.root.after(np.random.randint(500, 1000), self.bot_move)
                    #self.bot_move()

        
    def bot_move(self):
        positions = self.board.availablePositions()
        action = self.p2.choose_action(self.board, player = self.bot_turn)
        print(action)
        moveMade = self.board.make_move(action)
        print(moveMade)
        if moveMade == 1:
            self.update_visual()
            print(self.check_win(return_line=True))

    def update_visual(self):
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
    
    def update_bot_type(self):
        try:
            if self.bot_type == "DQN":
                self.p2 = Player(_name="20k_sem1_3x2_32", _player_type="DQN")
            elif self.bot_type == "Minimax":
                self.p2 = Player(_name="board_nextMoves_3_4_3_mmps", _player_type="Minimax")
            elif self.bot_type == "Q-learning":
                print('here')
                self.p2 = Player(_name="policy2_sem1_3_2_20k", _player_type="Q-learning")
                print(self.p2.agent.states_value)
            elif self.bot_type == "Monte Carlo":
                print("here")
                self.p2 = Player(_name="policy_sem1_3_2_SCM_1k", _player_type="Monte Carlo")
        except:
            print("Error trying to load bot's data")

    def check_win(self, return_line=False):
        if return_line:
            win_flag, win_line = self.board.check_win(return_line=True)
            if win_flag != -1:
                if self.bot_turn == 1:
                    if self.board.movesMade % 2 == 0:
                        txt = "You Won!"
                    else:
                        txt = "You Lost..."
                else:
                    if self.board.movesMade % 2 != 0:
                        txt = "You Won!"
                    else:
                        txt = "You Lost..."
                self.winner_txt["text"] = txt

                line_dir =  (win_line[2][0] - win_line[0][0], win_line[2][1] - win_line[0][1])
                if line_dir[0] == 0:
                    self.canvas.create_line(30 + 100*win_line[0][1], 50 + 100*win_line[0][0], 70 + 100*win_line[2][1], 50 + 100*win_line[2][0], width=12, tags="win_line")
                elif line_dir[1] == 0:
                    self.canvas.create_line(50 + 100*win_line[0][1], 30 + 100*win_line[0][0], 50 + 100*win_line[2][1], 70 + 100*win_line[2][0], width=12, tags="win_line")
                elif line_dir[0] > 0 and line_dir[1] > 0:
                    self.canvas.create_line(40 + 100*win_line[0][1], 40 + 100*win_line[0][0], 60 + 100*win_line[2][1], 60 + 100*win_line[2][0], width=12, tags="win_line")
                else:
                    self.canvas.create_line(60 + 100*win_line[0][1], 40 + 100*win_line[0][0], 40 + 100*win_line[2][1], 60 + 100*win_line[2][0], width=12, tags="win_line")
        else:
            win_flag = self.board.check_win(return_line=False)
            if win_flag != -1:
                if self.board.movesMade % 2 == 0:
                    txt = "You Won!"
                else:
                    txt = "You Lost..."
                self.winner_txt["text"] = txt
        
        return win_flag

    def reset(self):
        self.board.reset()
        self.canvas.delete("move")
        self.canvas.delete("win_line")
        self.winner_txt["text"] = ""

        self.bot_type = self.tkvar.get()
        self.update_bot_type()

        if self.bot_turn == 1:
            self.bot_move()

    def change_dropdown(self, *args):
        self.bot_type = self.tkvar.get()
        self.update_bot_type()
        print( self.tkvar.get() )
                
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Semáforo")
    root.geometry("410x420")
    VisualGame(root).pack(fill="both", expand=True)
    root.mainloop()