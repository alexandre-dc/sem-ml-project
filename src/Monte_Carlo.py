import gym
import gym_sem
import numpy as np
import time
import pickle

class Monte_Carlo:
    def __init__(self, _type):
        self.env = gym.make('sem-v0', _type='Monte Carlo')
        self.env_test = gym.make('sem-v0', _type='Monte Carlo Test')
        self._type = _type
        self.gamma = 1

    def monte_carlo_search(self, n_steps, results={}, dict_canonic_states={}, nodes_info={}):
        results = results
        dict_canonic_states = dict_canonic_states
        nodes_info = nodes_info

        for i in range(n_steps):
            print("Searching: " + str(int(i/n_steps * 100)) + "%", end='\r' + '\t'*4)

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
            
            G = 0
            s_list.reverse()
            for st in s_list:
                G = G * self.gamma + st[1]
                if self._type == "SCM-mask":                #------------------------- SCM-mask Search --------------     Symmetry and Canonic Memory with Mask
                    if str(st[0]) in dict_canonic_states:
                        key_state = dict_canonic_states[str(st[0])]
                        results[key_state].append(G)
                    else:
                        canonic_state, all_symmetry = env.get_canonic_state(st[0])
                        key_state = str(canonic_state)
                        for sym in all_symmetry:
                            dict_canonic_states[str(sym)] = key_state

                        results[key_state] = [G]
                elif self._type == "SC-mask":               #------------------------- SC-mask Search ----------------    Symmetry and Canonic with Mask
                    key_state = str(env.get_canonic_state_mask(st[0]))
                    if key_state in results:
                        results[key_state].append(G)
                    else:
                        results[key_state] = [G]
                elif self._type == "SCM":                   #------------------------- SCM Search --------------------    Symmetry and Canonic Memory
                    if str(st[0]) in dict_canonic_states:
                        key_state = dict_canonic_states[str(st[0])]
                        results[key_state].append(G)
                    else:
                        canonic_state, all_symmetry = self.env.get_canonic_state(st[0])
                        key_state = str(canonic_state)
                        for sym in all_symmetry:
                            dict_canonic_states[str(sym)] = key_state

                        results[key_state] = [G]
                elif self._type == "SC":                    #------------------------- SC Search ---------------------    Symmetry and Canonic
                    canonic_state, all_symmetry = self.env.get_canonic_state(st[0])
                    key_state = str(canonic_state)
                    if key_state in results:
                        results[key_state].append(G)
                    else:
                        results[key_state] = [G]
                elif self._type == "SnC":                   #------------------------ SnC Search ---------------------    Symmetry non Canonic
                    in_results = []
                    all_symmetry = env.get_symmetry(st[0])
                    for sym in all_symmetry:
                        if str(sym) in results:
                            in_results = str(sym)
                            break
                    if in_results == []:
                        results[str(st[0])] = [G]
                    else:
                        results[in_results].append(G)
                elif self._type == "nS":                    # ------------------------- nS Search --------------------    Non Symmetry
                    key_state = str(st[0])
                    if key_state in results:
                        results[key_state].append(G)
                    else:
                        results[key_state] = [G]

        f_results = {}
        for key in results:
            f_results[key] = np.mean(results[key])
        
        return f_results, results, dict_canonic_states

    def run_test (self, f_results, dict_canonic_states, turn):
        possible_Boards = self.env_test.possible_move_boards()
        if self._type == "SCM-mask":                #------------------------- SCM-mask Search --------------     Symmetry and Canonic Memory with Mask
            possible_canonic_boards = [env.get_canonic_state_mask(s[0]) for s in possible_Boards]
            possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        elif self._type == "SC-mask":
            possible_canonic_boards = [env.get_canonic_state_mask(s[0]) for s in possible_Boards]
            possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        elif self._type == "SCM":
            possible_canonic_boards = []
            for s in possible_Boards:
                if str(s) in dict_canonic_states:
                    possible_canonic_boards.append(dict_canonic_states[str(s)])
                else:
                    possible_canonic_boards.append(self.env_test.get_canonic_state(s[0])[0])
            possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        elif self._type == "SC":
            possible_canonic_boards = [self.env_test.get_canonic_state(s[0])[0] for s in possible_Boards]
            possible_Values = [f_results[str(s)] for s in possible_canonic_boards]
        elif self._type == "SnC":
            possible_Values = []
            for s in possible_Boards:
                all_symmetry = env.get_symmetry(s[0])
                for sym in all_symmetry:
                    if str(sym) in f_results:
                        possible_Values.append(f_results[str(sym)])
                        break
                else: print(possible_Boards[len(possible_Boards)])
        elif self._type == "nS":
            possible_Values = [f_results[str(s[0])] for s in possible_Boards]

        if turn == 1:
            best_State_Index = possible_Values.index(max(possible_Values))
        else:
            best_State_Index = possible_Values.index(min(possible_Values))
        action = possible_Boards[best_State_Index][1]

        return action
                    
    #-------------------- Monte Carlo Test ----------------------
    def test_MC (self, n_test, f_results, dict_canonic_states, minimax_test=False):
        self.env_test.agent_turn = -1

        all_rewards = []
        error = 0
        for i in range(n_test):
            print("Testing: " + str(int(i/n_test * 100)) + "%     ", end='\r' + '\t'*4)
            state = self.env_test.reset()
            done = False
            turn = 1

            while not done:
                try:
                    action = self.run_test(f_results, dict_canonic_states, self.env_test.agent_turn)
                    s, r, done, _ = self.env_test.step(action = action)
                except:
                    error += 1
                    s, r, done, _ = self.env_test.step()
                
                if done:
                    all_rewards.append(r)
                    break

                turn *= -1

        f_error = error
        f_rewards = sum(all_rewards)/len(all_rewards)

        if minimax_test:
            print("", end='\r' + '\t'*4)
            self.env_test._minimax_rate = 1
            all_rewards = []
            error = 0
            for i in range(n_test):
                print("Testing vs MM: " + str(int(i/n_test * 100)) + "%      ", end='\r' + '\t'*4)
                state = self.env_test.reset()
                done = False
                turn = 1
                while not done:
                    try:
                        action = self.run_test(f_results, dict_canonic_states, self.env_test.agent_turn)
                        s, r, done, _ = self.env_test.step(action = action)
                    except:
                        error += 1
                        s, r, done, _ = self.env_test.step()
                    
                    if done:
                        all_rewards.append(r)
                        break

                    turn *= -1

            self.env_test._minimax_rate = 0
            f_error = [f_error, error]
            f_rewards = [f_rewards, sum(all_rewards)/len(all_rewards)]

        return f_error, f_rewards, len(f_results)

    def save_log_to_file (self, f_name, total_time_searching, total_errors, total_rewards, total_size_results, n_search, minimax_data=False):
        # try:
        f = open("/home/alexandre/sem-project-logs/times/" + f_name, "a")
        f.write("\n")
        f.write(str(n_search) + "\n")
        line_to_write = ""
        for s in total_time_searching[n_search]:
            line_to_write += str(s) + ", "
        line_to_write += str(sum(total_time_searching[n_search])/len(total_time_searching[n_search]))
        f.write(line_to_write + "\n")
        line_to_write = ""

        if not minimax_data:
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
        else:
            for i in range(len(total_errors[n_search])):
                for s in total_errors[n_search]:
                    line_to_write += str(s[i]) + ", "
                line_to_write += str(sum(total_errors[n_search][i])/len(total_errors[n_search][i]))
                f.write(line_to_write + "\n")
                line_to_write = ""
                for s in total_rewards[n_search]:
                    line_to_write += str(s[i]) + ", "
                line_to_write += str(sum(total_rewards[n_search][i])/len(total_rewards[n_search][i]))
                f.write(line_to_write + "\n")
                line_to_write = ""
        for s in total_size_results[n_search]:
            line_to_write += str(s) + ", "
        line_to_write += str(sum(total_size_results[n_search])/len(total_size_results[n_search]))
        f.write(line_to_write + "\n")
        f.close()

        return 1
        # except:
        #     return 0


