import pickle
import time
import sem_game
from Player import Player
from Agent import Agent
from Q_Learning import Q_Learning

BOARD_ROWS = sem_game.BOARD_ROWS
BOARD_COLS = sem_game.BOARD_COLS
MAX_MOVES = sem_game.MAX_MOVES

train_steps = 10000
test_steps = 1000
minimax_rates = 0.8
for minimax_rate in minimax_rates:
    for steps in train_steps:
        agent = Agent(name="policy2_sem" + str(MAX_MOVES) + "_" + str(BOARD_ROWS) + "x" + str(BOARD_COLS), epsilon=1, lr=0.0002)
        #adv = Player(_name="board_nextMoves_3_4_1_mmps", _player_type="Minimax")
        adv = Player()
        q = Q_Learning(agent, adv, minimax_rate)
        t0 = time.clock()
        q.train(steps=steps, progressive_lr=False)
        time_learning = time.clock() - t0
        all_rewards = q.test(steps=test_steps)
        agent.savePolicy()
        f = open("log_q_learning.txt", "a")
        f.write(str(time_learning) + "\n")
        f.write(str(minimax_rate) + "\n")
        #f.write(str(l_size) + "\n")
        f.write(str(train_steps) + "\n")
        f.write(str(all_rewards) + "\n")
        f.write("\n")
        f.close()