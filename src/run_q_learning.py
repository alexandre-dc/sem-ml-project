import pickle

from Player import Player
from Agent import Agent
from Q_Learning import Q_Learning

train_steps = 20000
test_steps = 10000
agent = Agent(name="policy2_sem1_3_2_" + str(int(train_steps/1000)) + "k", epsilon=0.8, lr=0.001)
adv = Player()
q = Q_Learning(agent, adv)
q.train(steps=train_steps, progressive_lr=True)
q.test(steps=test_steps)
agent.savePolicy()