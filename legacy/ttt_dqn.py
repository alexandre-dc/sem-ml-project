import gym
import gym_ttt
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

'import matplotlib.pyplot as plt'

env = gym.make('ttt-v0')

n_actions = env.action_space.n  # dim of output layer 
#input_dim = env.observation_space.shape[0]  # dim of input layer
input_dim = 18
model = Sequential()
model.add(Dense(128, input_dim = input_dim , activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer=Adam(), loss = 'mse')

#model.summary()

n_episodes = 20000
gamma = 0.99
epsilon = 1
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
save_f = 50
save_models = []

def replay(replay_memory, minibatch_size=32):
    # choose <s,a,r,s',done> experiences randomly from the memory
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    # create one list containing s, one list containing a, etc
    s_l =      np.array(list(map(lambda x: x['s'], minibatch)))
    a_l =      np.array(list(map(lambda x: x['a'], minibatch)))
    r_l =      np.array(list(map(lambda x: x['r'], minibatch)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
    done_l   = np.array(list(map(lambda x: x['done'], minibatch)))
    # Find q(s', a') for all possible actions a'. Store in list
    # We'll use the maximum of these values for q-update  
    #print(sprime_l.shape)
    qvals_sprime_l = model.predict(sprime_l.reshape(32,18))
    # Find q(s,a) for all possible actions a. Store in list
    target_f = model.predict(s_l.reshape(32,18))
    # q-update target
    # For the action we took, use the q-update value  
    # For other actions, use the current nnet predicted value
    for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
        if not done:  target = r + gamma * np.max(qvals_sprime)
        else:         target = r
        target_f[i][a] = target
    # Update weights of neural network with fit() 
    # Loss function is 0 for actions we didn't take
    model.fit(s_l.reshape(32,18), target_f, epochs=1, verbose=0)
    return model

#-----------------------------------

log_print = []
for n in range(n_episodes): 
    s = env.reset()
    done=False
    r_sum = 0
    #print(0.04*len(save_models))
    if np.random.rand() <= 0.01*len(save_models):
        #save_model = save_models[np.random.choice(len(save_models))]
        save_model = None
    else:
        save_model = None

    if np.random.rand() < 0:
        model_turn = -1
        if save_model == None:
            sprime, r, done, info = env.step('random', -1)
        else:
            qvals_s = save_model.predict(s.reshape(1,18))
            a = np.argmax(qvals_s)
            sprime, r, done, info = env.step(a, -1)
        s = sprime
    else:
        model_turn = 1

    while not done: 
        # Uncomment this to see the agent learning
        # env.render()
        
        # Feedforward pass for current state to get predicted q-values for all actions 
        qvals_s = model.predict(s.reshape(1,18))
        # Choose action to be epsilon-greedy
        if np.random.random() < epsilon:  
            a = env.action_space.sample()
        else:                             
            a = np.argmax(qvals_s) 
        # Take step, store results 
        sprime, r, done, info = env.step(a, 1)
        r_sum += r

        if done and model_turn == 1 and r_sum == 0:
            r_sum -= 0.2

        # add to memory, respecting memory buffer limit 
        # if len(replay_memory) > mem_max_size:
        #     replay_memory.pop(0)
        # replay_memory.append({"s":s.reshape(1,18),"a":a,"r":r,"sprime":sprime.reshape(1,18),"done":done})
        
        # Update state
        #s=sprime
        # Train the nnet that approximates q(s,a), using the replay memory
        #model=replay(replay_memory, minibatch_size = minibatch_size)
        # Decrease epsilon until we hit a target threshold 
        #if epsilon > 0.01:      epsilon -= 0.001

        if done:
            if len(replay_memory) > mem_max_size:
                replay_memory.pop(0)
            replay_memory.append({"s":s.reshape(1,18),"a":a,"r":r,"sprime":sprime.reshape(1,18),"done":done})
            break

        if save_model == None:
            sprime_1, r, done, info = env.step('random', -1)
        else:
            qvals_s = save_model.predict(sprime.reshape(1,18))
            a = np.argmax(qvals_s)
            sprime_1, r, done, info = env.step(a, -1)
        r_sum += r 

        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({"s":s.reshape(1,18),"a":a,"r":r,"sprime":sprime.reshape(1,18),"done":done})
       
        if done:
            # if len(replay_memory) > mem_max_size:
            #     replay_memory.pop(0)
            # replay_memory.append({"s":s.reshape(1,18),"a":a,"r":r,"sprime":sprime.reshape(1,18),"done":done})
            break
        s = sprime_1

        

    model=replay(replay_memory, minibatch_size = minibatch_size)
    if epsilon > 0.02:      
        epsilon -= 0.0005
    #print("Total reward:", r_sum)
    
    r_sums.append(r_sum)
    log_int = 500
    if n % log_int == 0:
        print(epsilon)
        #print(r_sums[-100:])
        print(str(n) + " : " + str(sum(r_sums[-log_int:])/log_int))
        log_print.append(sum(r_sums[-log_int:])/log_int)

    if n % save_f == 0:
        save_models.append(model)
        if len(save_models) >= 20:
            save_models.pop(0)

for i in range(10):
    s = env.reset()
    env.render()
    done=False
    save_model = save_models[-1]

    print('-----------------------------------')

    while 1:
        qvals_s = model.predict(s.reshape(1,18))
        a = np.argmax(qvals_s)

        sprime, r, done, info = env.step(a, 1)

        env.render()
        if done:
            print(r)
            break

        s=sprime

        sprime, r, done, info = env.step('random', -1)
        # qvals_s = save_model.predict(s.reshape(1,18))
        # a = np.argmax(qvals_s)
        # sprime, r, done, info = env.step(a, -1)

        env.render()
        if done:
            print(r)
            break

        s=sprime

plt.plot(log_print)
plt.show()
