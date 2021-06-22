from os import terminal_size
from Monte_Carlo import Monte_Carlo

import time
import pickle


total_time_searching = {}
total_errors = {}
total_rewards = {}
total_size_results = {}

_type='SCM'
start_n_search = 20000
step_n_search = 10000
n_tests = 5000
max_n_search = 1 * step_n_search + start_n_search
attemps = 1

monte_carlo = Monte_Carlo(_type=_type)

n_search_lst = [10000, 20000, 30000, 40000, 50000]

#n_search_lst1 = [5000]

f_name = "mc_tests_SCM_g95_1_3x4.txt"

for i in range(attemps):
    results = {}
    dict_canonic_states = {}
    n_search_prev = 0
    t1_prev = 0
    for n_search in n_search_lst:
        print("n_search =", n_search, "  attemp", i+1, end="\t")

        t0 = time.clock()       # Timer start
        f_results, results, dict_canonic_states  = monte_carlo.monte_carlo_search(n_search - n_search_prev, results, dict_canonic_states)     # Searching processe
        t_searching = t1_prev + time.clock() - t0     # Timer end

        error, total_reward, dict_size = monte_carlo.test_MC(int(n_tests), f_results, dict_canonic_states, minimax_test=True)      # Testing processe

        if n_search in total_time_searching:
            total_time_searching[n_search].append(t_searching)
            total_errors[n_search].append(error)
            total_rewards[n_search].append(total_reward)
            total_size_results[n_search].append(dict_size)
        else:
            total_time_searching[n_search] = [t_searching]
            total_errors[n_search] = [error]
            total_rewards[n_search] = [total_reward]
            total_size_results[n_search] = [dict_size]

        print(" "*100, end='\r' + '\t'*4)
        print()
        print("Time searching:" + "\t"*2, t_searching)
        print("Errors:" + "\t"*2, error)
        print("Total reward:" + "\t"*2, total_reward)
        print("Size of dict_results:" + "\t"*2, len(f_results))
        print()
        print()

        n_search_prev = n_search
        t1_prev = t_searching

for n_search in n_search_lst:
    monte_carlo.save_log_to_file (f_name, total_time_searching, total_errors, total_rewards, total_size_results, n_search, minimax_data=False)

fw = open('/home/alexandre/sem-project-logs/monte_carlo/policy_sem1_3x4_' + _type + '_f_results', 'wb')
pickle.dump(f_results, fw)
fw.close()
fw = open('/home/alexandre/sem-project-logs/monte_carlo/policy_sem1_3x4_' + _type + '_dcs', 'wb')
pickle.dump(dict_canonic_states, fw)
fw.close()

print(total_time_searching)
print(total_errors)
print(total_rewards)
print(total_size_results)