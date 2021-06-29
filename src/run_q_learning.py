import pickle
import sem_game
from Player import Player
from Agent import Agent
from Q_Learning import Q_Learning

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

train_steps = 50000
test_steps = 5000
agent = Agent(name="policy2_sem" + str(MAX_MOVES) + "_" + str(BOARD_ROWS) + "x" + str(BOARD_COLS), epsilon=0.8, lr=0.0001)
#adv = Player(_name="board_nextMoves_3_4_1_mmps", _player_type="Minimax")
adv = Player()
q = Q_Learning(agent, adv)
q.train(steps=train_steps, progressive_lr=True)
q.test(steps=test_steps)
agent.savePolicy()