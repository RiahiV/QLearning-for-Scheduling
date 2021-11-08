# import libraries
import numpy as np
import pandas as pd
import random

random.seed(123)

#define training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn

# define the shape of the actions (the number of jobs)
n_jobs = 11

# Create a 2D numpy array to hold the current Q-values for each state_index and action pair: Q(s, a)
q_values = []
first_s = [0] * n_jobs
q_values.append(first_s)
print(q_values)
# print(np.random.choice(n_jobs, n_jobs, replace=True))
q_values.append(list(np.random.choice(n_jobs, n_jobs, replace=True)))
print(q_values)

q_values_state_table = pd.DataFrame({'states': [str([])]})
new_state = str([2])
new_state_df = pd.DataFrame({'states': [new_state]})
q_values_state_table = pd.concat([q_values_state_table, new_state_df], ignore_index=True)
print(q_values_state_table)
print(q_values[1][3])

if new_state in q_values_state_table.values:
    index_of_new_state = q_values_state_table[q_values_state_table['states'] == new_state].index.values.astype(int)[0]
    print(q_values[index_of_new_state])


####### argmax function for Q-values
# First we are going to implement the argmax function, which takes in a list of action values and returns an
# action with the highest value. Why are we implementing our own instead of using the argmax function that numpy uses?
# Numpy's argmax function returns the first instance of the highest value. We do not want that to happen as it biases
# the agent to choose a specific action in the case of ties. Instead we want to break ties between the highest values
# randomly. So we are going to implement our own argmax function.
def argmax(q_values_input):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values_input)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        # if a value is equal to top value add the index to ties
        # return a random selection from ties.
        if q_values_input[i] > top_value:
            top_value = q_values_input[i]
            ties = []
            ties.append(i)
        elif q_values_input[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)


picked_action_index = argmax(q_values[index_of_new_state])
print(picked_action_index)


def update_q_value(q_values_table, reward, learning_rate, discount_factor, state_index, last_action, best_Q_next_state):
    q_values_table[state_index][last_action] = q_values_table[state_index][last_action] + learning_rate * (reward + discount_factor * best_Q_next_state -
                                                                                                           q_values_table[state_index][last_action])
    return q_values_table


print(update_q_value(q_values, 10, 1, 1, 1, picked_action_index, 0))


# define a function that determines if the current state is a terminal state (i.e., a complete sequence)
def is_terminal_state(current_state, n_jobs):
    if len(current_state) != n_jobs:
        return False
    else:
        return True
