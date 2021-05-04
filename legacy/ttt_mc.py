import gym
import gym_ttt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

env = gym.make('ttt-v0')

def monte_carlo_search(steps, main_State = None):

    resultes = {}
    log = 1000

    for i in range(steps):

        if i % log == 0:
            print(i)

        s = env.reset()
        done = False
        turn = 1
        s_list = []

        while not done:
            s, r, done, _ = env.step(player = turn)
            s = s.reshape(3*3)
            s_list.append((list(s), r))
            #print(s_list)
            #print()

            turn *= -1
        
        #env.render()
        G = 0
        #print(s_list)
        s_list.reverse()
        for st in s_list:
            G = G*0.95 + st[1]
            if str(st[0]) in resultes:
                resultes[str(st[0])].append(G)
            else:
                resultes[str(st[0])] = [G]
        
    f_resultes = {}
    for key in resultes:
        print(key)
        print(resultes[key])
        f_resultes[key] = np.mean(resultes[key])
    
    return f_resultes




def q_learning(steps, epsilon = 0.8):
    q_Table = {}
    log = 1000

    for i in range(steps):
        if i % log == 0:
            print(epsilon)
            print(i)

        s = env.reset()
        done = False
        turn = 1
        s_list = []

        while not done:
            action = 0
            if turn == 1:
                s_ = str(list(s.reshape(3*3)))
                if s_ not in q_Table.keys():
                    q_Table[s_] = []
                    avaliable_Pos = env.board.availablePositions()
                    for pos in avaliable_Pos:
                        q_Table[s_].append((pos, 0))
                    action = avaliable_Pos[np.random.choice(len(avaliable_Pos))]

                else:
                    if np.random.rand() > epsilon:
                        q_Values = [a[0] for a in q_Table[s_]]
                        q_Actions = [a[1] for a in q_Table[s_]]
                        action_Index = q_Values.index(max(q_Values))
                        action = q_Actions[action_Index]
                    else:
                        avaliable_Pos = env.board.availablePositions()
                        action = avaliable_Pos[np.random.choice(len(avaliable_Pos))]
                        #action = (int(action/3), int(action%3))
                
                s, r, done, _ = env.step(action = action, player = turn)

                st = str(list(s.reshape(3*3)))
                s_list.append((s_, action, r))
                

                # if done:
                #     st = str(list(s.reshape(3*3)))
                #     s_list.append((s_, action, r))
                #     break

            #s = s_obv
                
            # else:
            #     avaliable_Pos = env.board.availablePositions()
            #     rand_action = avaliable_Pos[np.random.choice(len(avaliable_Pos))]
            #     s_obv, r, done, _ = env.step(action = rand_action, player = turn)
            #     print(s_obv)
            #     st = str(list(s.reshape(3*3)))
            #     s_list.append((st, action, r))
            #     s = s_obv

            #turn *= -1

        s_list.reverse()
        G = 0
        print(s_list)
        print()
        for st in s_list:
            # print(st)
            # print()
            # print(q_Table[st[0]])
            # print('here')
            # print(q_Table[st[0]][0])
            print()
            for q_Values in q_Table[st[0]]:
                print(st[1])
                print(q_Values[0])
                if st[1] == q_Values[0]:
                    #q_Table
                    G = q_Values[1] + 0.001*(G*0.95 + st[2])
                    q_Table[st[0]] = (st[1], G)
        
        if epsilon >= 0.02:
            epsilon -= 0.00005
    
    return q_Table

#-------------------- Monta Carlo Test ----------------------

# all_rewards = []
# for i in range(1000):
#     state = env.reset()
#     done = False
#     turn = 1

#     while not done:
#         if turn == 1:
#             possible_Boards = env.possible_move_boards(player = 1)
#             possible_Values = [f_resultes[str(s[0])] for s in possible_Boards]

#             best_State_Index = possible_Values.index(max(possible_Values))

#             a = possible_Boards[best_State_Index][1]

#             s, r, done, _ = env.step(action = a, player = 1)
#         else:
#             s, r, done, _ = env.step(player = -1)
        
#         if done:
#             print(r)
#             all_rewards.append(r)
#             break

#         turn *= -1

# print(sum(all_rewards)/len(all_rewards))


q_Table = q_learning(10000)

#-------------------- Q-Learning Test --------------------------

all_rewards = []
for i in range(5000):
    s = env.reset()
    done = False
    turn = 1

    while not done:
        s_ = str(list(s.reshape(3*3)))
        if turn == 1:
            q_Values = [a[0] for a in q_Table[s_]]
            q_Actions = [a[1] for a in q_Table[s_]]
            action_Index = q_Values.index(max(q_Values))
            action = q_Actions[action_Index]

            s, r, done, _ = env.step(action = action, player = 1)

            env.render()
            if done:
                #print(r)
                all_rewards.append(r)
                break

        # else:
        #     avaliable_Pos = env.board.availablePositions()
        #     rand_action = avaliable_Pos[np.random.choice(len(avaliable_Pos))]
        #     s, r, done, _ = env.step(action = rand_action, player = -1)
        
        # if done:
        #     #print(r)
        #     all_rewards.append(r)
        #     break

        # turn *= -1

print(sum(all_rewards)/len(all_rewards))
print(q_Table)



# while not done:
#     while True:
#         row = input("Input your action row:")
#         col = input("Input your action col:")
#         try:
#             row = int(row)
#             col = int(col)
#             a = row*3 + col

#         except:
#             print("Wrong move format")

#         s, r, done, _ = env.step(action = a, player = 1)
#         env.render()
#         break

#     possible_Boards = env.possible_move_boards(player = -1)
#     possible_Values = [f_resultes[str(s[0])] for s in possible_Boards]
#     print(possible_Values)
#     best_State_Index = possible_Values.index(min(possible_Values))
#     print(best_State_Index)
#     #move = np.array(possible_Boards[best_State]) - env.board.state.reshape(3*3)
#     a = possible_Boards[best_State_Index][1]
#     # = list(move).index(1)
#     # for m in move:
#     #     if m == 1:
#     #         a = move.index(m)
#     print('here')
#     print(a)
#     s, r, done, _ = env.step(action = a, player = -1)

#     env.render()


    
            
