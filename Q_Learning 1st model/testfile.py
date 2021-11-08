import random
import numpy as np

# current_state = [1, 7, 5]
# all_actions_list = list(range(0, 10))
# all_actions_not_picked = [x for x in all_actions_list if x not in current_state]
# print(all_actions_list)
# print(all_actions_not_picked)
#
# current_state.sort(reverse=True)
# for an in current_state:
#     del all_actions_list[an]
# print(all_actions_list)

arr = [2,3,5,4,8,9,6,5,2,]
print(arr)
np.mean(arr.reshape(-1, 3), axis=1)


