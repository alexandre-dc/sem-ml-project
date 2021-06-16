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
n_tests = 1000
max_n_search = 1 * step_n_search + start_n_search
attemps = 1

monte_carlo = Monte_Carlo(_type=_type)

n_search_lst = [100, 200, 300, 400, 500, 600, 700, 800, 900,
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
            10000, 20000]

n_search_lst1 = [200000]

f_name = "mc_tests_SCM_g100+_3x4_1.txt"
# f_name = "mc_time_tests_SnC.txt"

#for n_search in range(start_n_search, max_n_search, step_n_search):
for n_search in n_search_lst1:
    for i in range(attemps):
        print("n_search =", n_search, "  attemp", i+1, end="\t")

        t0 = time.clock()       # Timer start
        f_results, dict_canonic_states  = monte_carlo.monte_carlo_search(n_search)     # Searching processe
        t_searching = time.clock() - t0     # Timer end

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

    monte_carlo.save_log_to_file (f_name, total_time_searching, total_errors, total_rewards, total_size_results, n_search)

fw = open('/home/alexandre/sem-project-logs/monte_carlo/policy_sem1_3_4_' + _type + '_f_results', 'wb')
pickle.dump(f_results, fw)
fw.close()
fw = open('/home/alexandre/sem-project-logs/monte_carlo/policy_sem1_3_4_' + _type + '_dcs', 'wb')
pickle.dump(dict_canonic_states, fw)
fw.close()

print(total_time_searching)
print(total_errors)
print(total_rewards)
print(total_size_results)