########## Algorithm discrprtion
# ###States:
# For the agent, the definition of the state will coincide with the jobs that have finished and the order in which they were done, that is,
# the sequence that has been built. This means that at the beginning of an episode, the state of the environment is empty, since no work has been added to the sequence;
# therefore, at the end of the episode the state of the environment will be given by all the sequenced works.
#### Actions: In this method, we are moving from position k = 1,2,...,n and find the best job for each position k
# Action 1: pick job 1
# Action 2: pick job 2
# Action 3: pick job 3
# …
#### Rewards: As described, the state is represented by the sequence that has been built, an action constitutes inserting a new job in said sequence and taking into account that
# our final objective is to minimize the makespan; We define reward -𝑚𝑎𝑘𝑒𝑠𝑝𝑎𝑛.
# It should be noted then that the highest Q-value determines the best action in any state.

# import libraries
import numpy as np
import pandas as pd
import random
import re

#from datetime import datetime
#random.seed(datetime.now())
random.seed(123)
"""
first we are going to read the benchmark data instances
"""

with open('C:/Users/RIA010/OneDrive - CSIRO/Desktop/MLAI/Q-Learning for scheduling- My code/Carlier_instances/car7_7.txt', 'r') as f: lines = f.readlines()

# remove spaces
lines = [line.replace('\n', ' ') for line in lines]
lines = [line.strip() for line in lines]
lines = [line.rstrip("\n") for line in lines]
lines = [line.replace(" ", ",") for line in lines]
lines = [line.replace(",,", ",") for line in lines]
lines = [line.rstrip("") for line in lines]
# print(lines)

n_jobs = int(lines[0])
n_mach = int(lines[1])
print(n_jobs, n_mach)
process_t = np.zeros((n_jobs, n_mach))

process_line = lines[3:]
#print(process_line)
for item in range(0, n_jobs):
    #   print(item)
    #    print(process_line[item])
    line_current = process_line[item]
    line_current = re.findall('\d+', line_current)
    #  print(line_current)
    for mach_counter in range(0, n_mach):
        process_t[item, mach_counter] = line_current[mach_counter]
        # print(process_t[item,mach_counter])


def objective_value(cur_sol):
    how_many_job = len(cur_sol)
    compl_t = np.zeros((how_many_job, n_mach))

    compl_t[cur_sol[0], 0] = process_t[cur_sol[0], 0]
    #    print(compl_t)
    for item_m in range(1, n_mach):
        compl_t[cur_sol[0], item_m] = compl_t[cur_sol[0], item_m - 1] + process_t[cur_sol[0], item_m]

    for item_j in range(1, how_many_job):
        for item_m in range(0, n_mach):
            if item_m == 0:
                compl_t[cur_sol[item_j], item_m] = compl_t[cur_sol[item_j - 1], item_m] + process_t[
                    cur_sol[item_j], item_m]
            else:
                compl_t[cur_sol[item_j], item_m] = max(compl_t[cur_sol[item_j], item_m - 1],
                                                       compl_t[cur_sol[item_j - 1], item_m]) + process_t[
                                                       cur_sol[item_j], item_m]

    return compl_t[cur_sol[how_many_job - 1], n_mach - 1]


def objective_value_reward(cur_sol):
    how_many_job = len(cur_sol)
    #    print(cur_sol, how_many_job, process_t[cur_sol[0], 0])
    compl_t = np.zeros((n_jobs, n_mach))

    compl_t[cur_sol[0], 0] = process_t[cur_sol[0], 0]
    #    print(compl_t)
    for item_m in range(1, n_mach):
        compl_t[cur_sol[0], item_m] = compl_t[cur_sol[0], item_m - 1] + process_t[cur_sol[0], item_m]

    if how_many_job >= 2:
        for item_j in range(1, how_many_job):
            for item_m in range(0, n_mach):
                if item_m == 0:
                    compl_t[cur_sol[item_j], item_m] = compl_t[cur_sol[item_j - 1], item_m] + process_t[cur_sol[item_j], item_m]
                else:
                    compl_t[cur_sol[item_j], item_m] = max(compl_t[cur_sol[item_j], item_m - 1], compl_t[cur_sol[item_j - 1], item_m]) + process_t[cur_sol[item_j], item_m]

    #    print(compl_t[cur_sol[how_many_job - 1], n_mach - 1])
    if how_many_job >= 2:
        reward = compl_t[cur_sol[how_many_job - 1], n_mach - 1] - compl_t[cur_sol[how_many_job - 2], n_mach - 1]
    else:
        reward = compl_t[cur_sol[how_many_job - 1], n_mach - 1]

    return reward


def argmax(q_values_input, current_state):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values_input)):
        if q_values_input[i] > top_value and i not in current_state:
            top_value = q_values_input[i]
            ties = []
            ties.append(i)
        elif q_values_input[i] == top_value and i not in current_state:
            ties.append(i)
    return np.random.choice(ties)


# define a function that determines if the current state is a terminal state (i.e., a complete sequence)
def is_terminal_state(current_state, n_jobs):
    if len(current_state) != n_jobs:
        return False
    else:
        return True


# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_state_indx, q_values, epsilon, n_jobs, current_state):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    all_actions_list = list(range(0, n_jobs))
    all_actions_not_picked = [x for x in all_actions_list if x not in current_state]

    if np.random.random() < epsilon:
        return argmax(q_values[current_state_indx], current_state)
    else:  # choose a random action
        return random.choice(all_actions_not_picked)


def current_state_index(current_state, q_values_state_table):
    q_values_state_table_list = q_values_state_table['states'].to_list()
    if current_state in q_values_state_table.values:
        index_of_current_state = q_values_state_table[q_values_state_table['states'] == current_state].index.values.astype(int)[0]
        return index_of_current_state
    else:
        return len(q_values_state_table)


def get_best_Q_next_state(new_state, q_values_state_table, q_values):
    new_state_index = current_state_index(new_state, q_values_state_table)
    if new_state_index == len(q_values_state_table):
        best_Q_next_state = 0
    else:
        best_Q_next_state = max(q_values[new_state_index])
    return best_Q_next_state


def update_q_values_state_table(current_state, q_values_state_table, q_values, n_jobs):
    if str(current_state) not in q_values_state_table.values:
        new_s = [0] * n_jobs
        q_values.append(new_s)

        new_state_df = pd.DataFrame({'states': [str(current_state)]})
        q_values_state_table = pd.concat([q_values_state_table, new_state_df], ignore_index=True)
    return q_values, q_values_state_table


def update_q_value(q_values_table, reward, learning_rate, discount_factor, state_index, picked_action, best_Q_next_state):
    q_values_table[state_index][picked_action] = q_values_table[state_index][picked_action] + learning_rate * (reward + discount_factor * best_Q_next_state -
                                                                                                               q_values_table[state_index][picked_action])
    return q_values_table


# define training parameters
epsilon = 0.9  # the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.9  # the rate at which the AI agent should learn

best_sol_obj = float('inf')
best_sol_prm = []
# Create a 2D numpy array to hold the current Q-values for each state_index and action pair: Q(s, a)
q_values = []
first_s = [0] * n_jobs
q_values.append(first_s)
q_values_state_table = pd.DataFrame({'states': [str([])]})

for episode in range(5000):
    # get the starting sequence for this episode which is an empty sequence
    current_state = []
    while not is_terminal_state(current_state, n_jobs):
        q_values, q_values_state_table = update_q_values_state_table(current_state, q_values_state_table, q_values, n_jobs)
        state_index = current_state_index(str(current_state), q_values_state_table)
        # choose which action to take (i.e., which job should be processed next)
        picked_action = get_next_action(state_index, q_values, epsilon, n_jobs, current_state)
        #       print(picked_action)
        new_state = current_state
        new_state.append(picked_action)  # update the current state
        best_Q_next_state = get_best_Q_next_state(new_state, q_values_state_table, q_values)
        reward = objective_value_reward(current_state)
        q_values = update_q_value(q_values, -reward, learning_rate, discount_factor, state_index, picked_action, best_Q_next_state)
        current_state = new_state

    current_state_obj = objective_value(current_state)
    if episode % 500 == 0:
        print(best_sol_obj)
    if current_state_obj < best_sol_obj:
        best_sol_obj = current_state_obj
        best_sol_prm = current_state

print(best_sol_prm, best_sol_obj)
# print(q_values)
# print(q_values_state_table)
