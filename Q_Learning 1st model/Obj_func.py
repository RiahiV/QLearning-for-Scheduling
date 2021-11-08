# import libraries
import numpy as np
import re

# first get all lines from file
with open(
        'C:/Users/RIA010/OneDrive - CSIRO/Desktop/MLAI/Q-Learning for scheduling- My code/Carlier_instances/car1_11.txt',
        'r') as f:
    lines = f.readlines()

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
# print(process_line)
for item in range(0, n_jobs):
    #   print(item)
    #    print(process_line[item])
    line_current = process_line[item]
    line_current = re.findall('\d+', line_current)
    #  print(line_current)
    for mach_counter in range(0, n_mach):
        process_t[item, mach_counter] = line_current[mach_counter]
        # print(process_t[item,mach_counter])

print(process_t)


######## Objective function ##############
# define a function that calculate the objective function
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

    #    print(compl_t)
    return compl_t[cur_sol[how_many_job - 1], n_mach - 1]


cur_sol = list(range(0, n_jobs))
print(objective_value(cur_sol))
