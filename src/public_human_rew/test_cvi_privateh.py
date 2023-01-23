import pdb

import numpy as np
import operator
import copy
import random
import matplotlib.pyplot as plt
import itertools

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

from hip_mdp_1player import HiMDP
from robot_tom_agent import Robot_Model
from human_agent import Human_Hypothesis

start_state = [2,2,2]
all_colors_list = [BLUE, GREEN, RED]
task_reward = [1,1,1]
human_rew = [0.5, 0.1, 0.5]
h_rho = 0
robot_rew = [0.5, 0.5, 0.1]
r_rho = 1
vi_type = 'cvi'

himdp = HiMDP(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, vi_type)
himdp.enumerate_states()
himdp.value_iteration()

# s = himdp.state_to_idx[(2,2,2)]
# action = himdp.policy[s]
# print("action", action)

current_state = copy.deepcopy(start_state)
total_rew = 0
while sum(current_state) > 0:
    s = himdp.state_to_idx[tuple(current_state)]
    action = himdp.policy[s]
    print(f"Robot action distribution: {action}")
    # r_action = np.argmax(action)
    indices = [idx for idx, val in enumerate(action) if val == max(action)]
    r_action = np.random.choice(indices)
    total_rew += robot_rew[r_action]
    print(f"Robot action: {r_action} in state {current_state}")
    print(f"rew = {total_rew}")



    current_state[r_action] -= 1
    if sum(current_state) == 0:
        break

    best_acts = []
    best_rew = -100
    for i in range(len(current_state)):
        if current_state[i] > 0:
            h_rew = human_rew[i]
            if h_rew == best_rew:
                best_rew = h_rew
                best_acts.append(i)
            elif h_rew > best_rew:
                best_rew = h_rew
                best_acts = [i]
    h_action = np.random.choice(best_acts)
    # h_action = best_acts[0]
    total_rew += human_rew[h_action]
    print(f"Human action: {h_action} in state {current_state}")
    print(f"rew = {total_rew}")
    print()

    current_state[h_action] -= 1

print(f"FINAL = {total_rew}")
