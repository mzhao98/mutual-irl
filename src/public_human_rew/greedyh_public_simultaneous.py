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

from compute_optimal_rew import OptimalMDP
from human_agent import Human_Hypothesis

def compute_optimal_rew(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho):
    himdp = OptimalMDP(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
    himdp.enumerate_states()
    himdp.value_iteration()
    rew = himdp.rollout_full_game_vi_policy()
    return rew

def run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, vi_type):
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
        # print(f"Robot action distribution: {action}")
        # r_action = np.argmax(action)
        indices = [idx for idx, val in enumerate(action) if val == max(action)]
        r_action = np.random.choice(indices)
        total_rew += robot_rew[r_action]
        # print(f"Robot action: {r_action} in state {current_state}")
        # print(f"rew = {total_rew}")



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
        # print(f"Human action: {h_action} in state {current_state}")
        # print(f"rew = {total_rew}")
        # print()

        current_state[h_action] -= 1

    # print(f"FINAL = {total_rew}")
    return total_rew

def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

if __name__ == "__main__":
    first_player = 'r'
    # start_state = [2, 2, 2, 2]
    all_colors_list = [BLUE, GREEN, RED, YELLOW]
    task_reward = [1, 1, 1, 1]
    # human_rew = [0.5, 0.1, 0.5, 0.1]
    # h_rho = 0
    # robot_rew = [0.5, 0.5, 0.1, 0.1]
    r_rho = 1
    # vi_type = 'stdvi'

    cvi_percents = {0: [], 1: []}
    stdvi_percents = {0: [], 1: []}
    num_exps = 100
    for exp in range(num_exps):
        print("exp = ", exp)
        start_state = [2 + np.random.randint(0, 5), 2 + np.random.randint(0, 5), 2 + np.random.randint(0, 5),
                       2 + np.random.randint(0, 5)]
        all_colors_list = [BLUE, GREEN, RED, YELLOW]
        human_rew = [np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0),
                     np.random.uniform(0.1, 10.0)]
        robot_rew = [np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0),
                     np.random.uniform(0.1, 10.0)]

        for h_rho in [0,1]:
            optimal_rew = compute_optimal_rew(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
            cvi_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, 'cvi')
            stdvi_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, 'stdvi')

            print()
            print("h_rho = ", h_rho)
            print("optimal_rew", optimal_rew)
            print("cvi_rew", cvi_rew)
            print("stdvi_rew", stdvi_rew)

            cvi_percent_of_opt = cvi_rew / optimal_rew
            stdvi_percent_of_opt = stdvi_rew / optimal_rew

            cvi_percents[h_rho].append(cvi_percent_of_opt)
            stdvi_percents[h_rho].append(stdvi_percent_of_opt)


    greedy_means = [np.round(np.mean(cvi_percents[0]), 2), np.round(np.mean(stdvi_percents[0]), 2)]
    greedy_stds = [np.round(np.std(cvi_percents[0]), 2), np.round(np.std(stdvi_percents[0]), 2)]

    collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    print("greedy_means", greedy_means)
    print("collab_means", collab_means)

    ind = np.arange(len(greedy_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, greedy_means, width, yerr=greedy_stds,
                    label='Greedy Human')
    rects2 = ax.bar(ind + width / 2, collab_means, width, yerr=collab_stds,
                    label='Collab Human')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Robot Type')
    ax.set_ylabel('Percentage of Optimal Reward')
    ax.set_ylim(-0.01, 1.1)
    ax.set_title('A1: Public Human Reward, Iterative\nPerformance by Robot and Human Type')
    ax.set_xticks(ind)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'))
    ax.legend()

    autolabel(ax, rects1, "left")
    autolabel(ax, rects2, "right")

    fig.tight_layout()

    plt.show()
