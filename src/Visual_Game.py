import tkinter as tk

from sem_game import Board
import sem_game
from Player import Player

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES


def makeMove(event):
    print(bot_type)
    if board.check_win() == -1 and bot_type != None:
        x = int(event.x / 100)
        y = int(event.y / 100)
        print(x, y)
        moveMade = board.make_move((y, x))
        print(moveMade)
        if moveMade == 1:
            atualizeVisual()
            print(board.check_win())
            if board.check_win() == -1:
                bot_move()

    
def bot_move():
    positions = board.availablePositions()
    action = p2.choose_action(board, player = bot_turn)
    print(action)
    moveMade = board.make_move(action)
    print(moveMade)
    if moveMade == 1:
        atualizeVisual()
        print(board.check_win())

def atualizeVisual():
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if board.state[i, j] == 1:
                canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                    fill="green", width=3, tags="move")
            elif board.state[i, j] == 2:
                canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                    fill="yellow", width=3, tags="move")
            elif board.state[i, j] == 3:
                canvas.create_oval(10 + 100*j, 10 + 100*i, 90 + 100*j, 90 + 100*i,
                    fill="red", width=3, tags="move") 

def reset(self):
    board.reset()
    canvas.delete("move")
    

    if bot_turn == 1:
        bot_move()


def change_dropdown(*args):
    
    print( tkvar.get() )

def update_bot_type(bot_type):




if __name__ == "__main__":
    root = tk.Tk()
    root.title("Semáforo")

    canvas = tk.Canvas(root, height=350, width=510, bg="grey")
    canvas.create_line(100, 5, 100, 300, width=3)
    canvas.create_line(200, 5, 200, 300, width=3)
    canvas.create_line(300, 5, 300, 300, width=3)

    canvas.create_line(5, 100, 400, 100, width=3)
    canvas.create_line(5, 200, 400, 200, width=3)
    canvas.create_rectangle(5, 5, 400, 300,
        width=5)

    canvas.pack(fill="both", expand=1)
    canvas.bind("<Button-1>", makeMove)

    reset_b = tk.Button(text="Reset", command=reset)
    reset_b.pack(side = "bottom")

    board = Board()
    bot_type = "Q-Learning"

    if bot_type == "DQN":
        p2 = Player(_name="200k_mm_sem1_3x4_32", _player_type="DQN")
    elif bot_type == "Minimax":
        p2 = Player(_name="board_nextMoves_3_4_3_mmps", _player_type="Minimax")
    elif bot_type == "Q-Learning":
        p2 = Player(_name="policy_sem1_3_2_20k", _player_type="Q-Learning")

    bot_turn = 1
    if bot_turn == 1:
        bot_move()

    tkvar = tk.StringVar(root)
    choices = { 'Minimax','Monte Carlo','Q-Learning','DQN'}
    tkvar.set('Minimax') # set the default option

    popupMenu = tk.OptionMenu(root, tkvar, *choices)
    # tk.Label(root, text="Choose a dish")
    popupMenu.pack(side="bottom")



    root.mainloop()