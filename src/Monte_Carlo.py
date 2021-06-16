import gym
import gym_sem
import numpy as np
import time
import pickle

class Monte_Carlo:
    def __init__(self, _type):
        self.env = gym.make('sem-v0', _type='Monte Carlo')
        self._type = _type

    def monte_carlo_search(self, n_steps, resultes={}, dict_canonic_states={}):
        # test_resultes = {}
        # time_searching = {}
        # t0 = time.clock()
        resultes = resultes
        dict_canonic_states = dict_canonic_states

        for i in range(n_steps):
            print("Searching:", int(i/n_steps * 100), "%", end='\r' + '\t'*4)

            s = self.env.reset()
            done = False
            turn = 1
            s_list = []

            while not done:
                s, r, done, _ = self.env.step()
                if done:
                    s_list.append((s, r))
                else:
                    s_list.append((s, 0))

                #self.env.board.turn *= -1
            
            G = 0
            s_list.reverse()
            for st in s_list:
                G = G * 1 + st[1]
                #------------------------ SnC Search -----------------------------    Symmetry non Canonic
                # in_results = []
                # all_symmetry = env.get_symmetry(st[0])
                # for sym in all_symmetry:
                #     if str(sym) in resultes:
                #         in_results = str(sym)
                #         break
                # if in_results == []:
                #     resultes[str(st[0])] = [G]
                # else:
                #     resultes[in_results].append(G)
                #------------------------- SCM Search ----------------------------     Symmetry and Canonic Memory
                #print(type(st[0]))
                if str(st[0]) in dict_canonic_states:
                    key_state = dict_canonic_states[str(st[0])]
                    resultes[key_state].append(G)
                else:
                    canonic_state, all_symmetry = self.env.get_canonic_state(st[0])
                    key_state = str(canonic_state)
                    for sym in all_symmetry:
                        dict_canonic_states[str(sym)] = key_state

                    resultes[key_state] = [G]
                #------------------------- SCM-mask Search ----------------------------     Symmetry and Canonic Memory with Mask
                # if str(st[0]) in dict_canonic_states:
                #     key_state = dict_canonic_states[str(st[0])]
                #     resultes[key_state].append(G)
                # else:
                #     canonic_state, all_symmetry = env.get_canonic_state(st[0])
                #     key_state = str(canonic_state)
                #     for sym in all_symmetry:
                #         dict_canonic_states[str(sym)] = key_state

                #     resultes[key_state] = [G]
                #------------------------- SC-mask Search ----------------------------     Symmetry and Canonic with Mask
                # key_state = str(env.get_canonic_state_mask(st[0]))
                # if key_state in resultes:
                #     resultes[key_state].append(G)
                # else:
                #     resultes[key_state] = [G]
                #------------------------- SC Search ----------------------------     Symmetry and Canonic
                # canonic_state, all_symmetry = env.get_canonic_state(st[0])
                # key_state = str(canonic_state)
                # if key_state in resultes:
                #     resultes[key_state].append(G)
                # else:
                #     resultes[key_state] = [G]
                #------------------------- nS Search ----------------------------     Non Symmetry
                # key_state = str(st[0])
                # if key_state in resultes:
                #     resultes[key_state].append(G)
                # else:
                #     resultes[key_state] = [G]
                #----------------------------------------------------------

        #     if i in n_tests:
        #         t_searching = time.clock() - t0
        #         time_searching[i] = t_searching
        #         f_results = {}
        #         for key in resultes:
        #             f_results[key] = np.mean(resultes[key])
        #         test_resultes[i] = f_results
        #         t0 = time.clock()

        f_results = {}
        for key in resultes:
            f_results[key] = np.mean(resultes[key])
        
        print(" "*100, end='\r' + '\t'*4)
        return f_results, dict_canonic_states

    def run_test (self, f_results, dict_canonic_states, turn):
        #self.env.board.showBoard()
        #print("turn-  " + str(turn))
        #self.env.board.showBoard()
        possible_Boards = self.env.possible_move_boards()
        #------------------------ SnC Test -----------------------------    Symmetry non Canonic
        # possible_Values = []
        # for s in possible_Boards:
        #     all_symmetry = env.get_symmetry(s[0])
        #     for sym in all_symmetry:
        #         if str(sym) in f_results:
        #             possible_Values.append(f_results[str(sym)])
        #             break
        #     else: print(possible_Boards[len(possible_Boards)])
        #------------------------- SCM Test ----------------------------     Symmetry and Canonic Memory
        possible_canonic_boards = []
        for s in possible_Boards:
            if str(s) in dict_canonic_states:
                possible_canonic_boards.append(dict_canonic_states[str(s)])
            else:
                possible_canonic_boards.append(self.env.get_canonic_state(s[0])[0])
        possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        #------------------------- SC-mask Test ----------------------------     Symmetry and Canonic with mask
        # possible_canonic_boards = [env.get_canonic_state_mask(s[0]) for s in possible_Boards]
        # possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        #------------------------- SC Test ----------------------------     Symmetry and Canonic
        # possible_canonic_boards = [env.get_canonic_state(s[0])[0] for s in possible_Boards]
        # possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        #------------------------- nS Test ----------------------------     Non Symmetry
        # possible_Values = [f_results[str(s[0])] for s in possible_Boards]
        #--------------------------------------------------------------
        #print(possible_Values)
        if turn == 1:
            best_State_Index = possible_Values.index(max(possible_Values))
        else:
            best_State_Index = possible_Values.index(min(possible_Values))
        action = possible_Boards[best_State_Index][1]

        return action
                    
    #-------------------- Monte Carlo Test ----------------------
    def test_MC (self, n_test, f_results, dict_canonic_states, minimax_test=False):
        if minimax_test:
            self.env.minimax_test = True
            self.env.agent_turn = 1

        all_rewards = []
        error = 0
        for i in range(n_test):
            print("Testing:", int(i/n_test * 100), "%", end='\r' + '\t'*4)
            state = self.env.reset()
            done = False
            turn = -1

            while not done:
                if turn == -1:
                    try:
                        action = self.run_test(f_results, dict_canonic_states, turn)
                        s, r, done, _ = self.env.step(action = action)
                    except:
                        error += 1
                        s, r, done, _ = self.env.step()
                else:
                    s, r, done, _ = self.env.step()
                
                if done:
                    all_rewards.append(r)
                    break

                turn *= -1

        if minimax_test:
            self.env.minimax_test = False
            self.env.agent_turn = 1

        return error, sum(all_rewards)/len(all_rewards), len(f_results)

    def save_log_to_file (self, f_name, total_time_searching, total_errors, total_rewards, total_size_results, n_search):
        try:
            f = open(f_name, "a")
            #for key in total_time_searching:
            f.write("\n")
            f.write(str(n_search) + "\n")
            line_to_write = ""
            for s in total_time_searching[n_search]:
                line_to_write += str(s) + ", "
            line_to_write += str(sum(total_time_searching[n_search])/len(total_time_searching[n_search]))
            f.write(line_to_write + "\n")
            line_to_write = ""
            for s in total_errors[n_search]:
                line_to_write += str(s) + ", "
            line_to_write += str(sum(total_errors[n_search])/len(total_errors[n_search]))
            f.write(line_to_write + "\n")
            line_to_write = ""
            for s in total_rewards[n_search]:
                line_to_write += str(s) + ", "
            line_to_write += str(sum(total_rewards[n_search])/len(total_rewards[n_search]))
            f.write(line_to_write + "\n")
            line_to_write = ""
            for s in total_size_results[n_search]:
                line_to_write += str(s) + ", "
            line_to_write += str(sum(total_size_results[n_search])/len(total_size_results[n_search]))
            f.write(line_to_write + "\n")
            f.close()

            return 1
        except:
            return 0


