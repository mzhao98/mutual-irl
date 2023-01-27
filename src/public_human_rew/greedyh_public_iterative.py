import pdb

import numpy as np
import operator
import copy
import random
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy import stats
from multiprocessing import Pool, freeze_support
rng = np.random.default_rng()

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

from compute_optimal_rew import OptimalMDP
from hip_mdp_1player import HiMDP
from human_agent import Human_Hypothesis

def compute_optimal_rew(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho):
    himdp = OptimalMDP(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
    himdp.enumerate_states()
    himdp.value_iteration()
    rew = himdp.rollout_full_game_vi_policy()
    human_best_rew = himdp.compute_max_human_reward()
    robot_best_rew = himdp.compute_max_robot_reward()
    altruism_case = himdp.compare_opt_to_greedy()
    return rew, human_best_rew, robot_best_rew, altruism_case

def run_exp_config(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, vi_type):
    himdp = HiMDP(start_state, all_colors_list, task_reward, human_rew, h_rho, None, robot_rew, robot_rew, r_rho, vi_type)
    himdp.enumerate_states()
    himdp.value_iteration()

    # s = himdp.state_to_idx[(2,2,2)]
    # action = himdp.policy[s]
    # print("action", action)

    current_state = copy.deepcopy(start_state)
    total_rew = 0
    total_human_rew = 0
    total_robot_rew = 0
    while sum(current_state) > 0:

        if first_player == 'r':
            s = himdp.state_to_idx[tuple(current_state)]
            action = himdp.policy[s]
            # print(f"Robot action distribution: {action}")
            # r_action = np.argmax(action)
            indices = [idx for idx, val in enumerate(action) if val == max(action)]
            r_action = np.random.choice(indices)
            total_rew += robot_rew[r_action]
            total_robot_rew += robot_rew[r_action]
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
            total_human_rew += human_rew[h_action]
            # print(f"Human action: {h_action} in state {current_state}")
            # print(f"rew = {total_rew}")
            # print()

            current_state[h_action] -= 1
        else:


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
            total_human_rew += human_rew[h_action]
            # print(f"Human action: {h_action} in state {current_state}")
            # print(f"rew = {total_rew}")
            # print()

            current_state[h_action] -= 1
            if sum(current_state) == 0:
                break

            s = himdp.state_to_idx[tuple(current_state)]
            action = himdp.policy[s]
            # print(f"Robot action distribution: {action}")
            # r_action = np.argmax(action)
            indices = [idx for idx, val in enumerate(action) if val == max(action)]
            r_action = np.random.choice(indices)
            total_rew += robot_rew[r_action]
            total_robot_rew += robot_rew[r_action]
            # print(f"Robot action: {r_action} in state {current_state}")
            # print(f"rew = {total_rew}")

            current_state[r_action] -= 1



    # print(f"FINAL = {total_rew}")
    return total_rew, total_human_rew, total_robot_rew

def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': -4, 'left': 4}

    for rect in rects:
        height = rect.get_height()
        # print("height",height)
        # height = [np.round(x,2) for x in height]
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.08),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=14)

def run_k_rounds(exp_num, all_colors_list, task_reward, r_rho, h_rho_of_interest):
    print("exp_num = ", exp_num)

    first_player = np.random.choice(['r', 'h'])
    # first_player = 'r'
    start_state = [np.random.randint(1, 10), np.random.randint(1, 10), np.random.randint(1, 10),
                   np.random.randint(1, 10)]
    if sum(start_state) % 2 == 1:
        start_state[-1] -= 1
    all_colors_list = [BLUE, GREEN, RED, YELLOW]
    human_rew = [np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2)]
    permutes = list(itertools.permutations(human_rew))
    # print("permutes",permutes)
    robot_rew = list(permutes[np.random.choice(np.arange(len(permutes)))])

    for h_rho in [h_rho_of_interest]:
        optimal_rew, best_human_rew, best_robot_rew, altruism_case = compute_optimal_rew(first_player, start_state, all_colors_list,
                                                                          task_reward,
                                                                          human_rew, h_rho, robot_rew, r_rho)
        cvi_rew, cvi_human_rew, cvi_robot_rew = run_exp_config(first_player, start_state, all_colors_list, task_reward, human_rew,
                                                               h_rho,
                                                               robot_rew, r_rho, 'cvi')
        stdvi_rew, stdvi_human_rew, stdvi_robot_rew = run_exp_config(first_player, start_state, all_colors_list, task_reward,
                                                                     human_rew, h_rho,
                                                                     robot_rew, r_rho, 'stdvi')

        # print()
        # print("h_rho = ", h_rho)
        # print("optimal_rew", optimal_rew)
        # print("cvi_rew", cvi_rew)
        # print("stdvi_rew", stdvi_rew)

        cvi_percent_of_opt_team = cvi_rew / optimal_rew
        stdvi_percent_of_opt_team = stdvi_rew / optimal_rew

        cvi_percent_of_opt_human = cvi_human_rew / best_human_rew
        stdvi_percent_of_opt_human = stdvi_human_rew / best_human_rew

        cvi_percent_of_opt_robot = cvi_robot_rew / best_robot_rew
        stdvi_percent_of_opt_robot = stdvi_robot_rew / best_robot_rew

    print("done with exp_num = ", exp_num)
    return cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
           cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case

def check_greedy_opt(exp_num, all_colors_list, task_reward, r_rho, h_rho_of_interest):
    print("exp_num = ", exp_num)

    first_player = np.random.choice(['r', 'h'])
    # first_player = 'r'
    start_state = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(1, 5),
                   np.random.randint(1, 5)]
    if sum(start_state) % 2 == 1:
        start_state[-1] -= 1
    all_colors_list = [BLUE, GREEN, RED, YELLOW]
    human_rew = [np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2),
                 np.round(np.random.uniform(0.1, 20.0),2)]
    permutes = list(itertools.permutations(human_rew))
    # print("permutes",permutes)
    robot_rew = list(permutes[np.random.choice(np.arange(len(permutes)))])

    for h_rho in [h_rho_of_interest]:
        optimal_rew, best_human_rew, best_robot_rew, altruism_case = compute_optimal_rew(first_player, start_state, all_colors_list,
                                                                          task_reward,
                                                                          human_rew, h_rho, robot_rew, r_rho)

    return altruism_case




def check_altruism():

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

    cvi_humanrew_percents = {0: [], 1: []}
    stdvi_humanrew_percents = {0: [], 1: []}

    cvi_robotrew_percents = {0: [], 1: []}
    stdvi_robotrew_percents = {0: [], 1: []}

    num_exps = 1000

    h_rho_of_interest = 0

    n_altruism = 0
    n_total = 0
    n_greedy = 0
    with Pool(processes=100) as pool:
        k_round_results = pool.starmap(check_greedy_opt, [(exp_num, all_colors_list, task_reward, r_rho, h_rho_of_interest) for exp_num in range(num_exps)])
        for result in k_round_results:
            altruism_case = result

            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1
    print("n_altruism = ", n_altruism)
    print("n_greedy = ", n_greedy)
    print("n_total = ", n_total)


def run_experiment():

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

    cvi_humanrew_percents = {0: [], 1: []}
    stdvi_humanrew_percents = {0: [], 1: []}

    cvi_robotrew_percents = {0: [], 1: []}
    stdvi_robotrew_percents = {0: [], 1: []}

    num_exps = 100

    h_rho_of_interest = 0

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent,2)] = 0

    with Pool(processes=100) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, all_colors_list, task_reward, r_rho, h_rho_of_interest) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case = result

            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1

            if altruism_case == 'opt':
                continue

            cvi_percents[h_rho_of_interest].append(cvi_percent_of_opt_team)
            stdvi_percents[h_rho_of_interest].append(stdvi_percent_of_opt_team)

            cvi_humanrew_percents[h_rho_of_interest].append(cvi_percent_of_opt_human)
            stdvi_humanrew_percents[h_rho_of_interest].append(stdvi_percent_of_opt_human)

            cvi_robotrew_percents[h_rho_of_interest].append(cvi_percent_of_opt_robot)
            stdvi_robotrew_percents[h_rho_of_interest].append(stdvi_percent_of_opt_robot)

            diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
            diff = np.round(diff, 2)
            print("percent_change = ", percent_change)
            percent_change[diff] += 1


    teamrew_means = [np.round(np.mean(cvi_percents[0]), 2), np.round(np.mean(stdvi_percents[0]), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents[0]), 2), np.round(np.std(stdvi_percents[0]), 2)]

    humanrew_means = [np.round(np.mean(cvi_humanrew_percents[0]), 2), np.round(np.mean(stdvi_humanrew_percents[0]), 2)]
    humanrew_stds = [np.round(np.std(cvi_humanrew_percents[0]), 2), np.round(np.std(stdvi_humanrew_percents[0]), 2)]

    robotrew_means = [np.round(np.mean(cvi_robotrew_percents[0]), 2), np.round(np.mean(stdvi_robotrew_percents[0]), 2)]
    robotrew_stds = [np.round(np.std(cvi_robotrew_percents[0]), 2), np.round(np.std(stdvi_robotrew_percents[0]), 2)]

    rvs3 = stats.norm.rvs(loc=5, scale=20, size=500, random_state=rng)
    print("team rew stat results: ", stats.ttest_ind([elem*100 for elem in cvi_percents[0]], [elem*100 for elem in stdvi_percents[0]]))
    print("human rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents[0]], [elem * 100 for elem in stdvi_humanrew_percents[0]]))
    print("robot rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents[0]],
                          [elem * 100 for elem in cvi_robotrew_percents[0]]))

    print("n_altruism = ", n_altruism)
    print("n_greedy = ", n_greedy)
    print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d]/sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.xlabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig("greedy_public_iterative_100_multiprocess_cdf.png")
    plt.show()






    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    ind = np.arange(len(robotrew_means))  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 5))
    rects1 = ax.bar(ind - width, teamrew_means, width, yerr=teamrew_stds,
                    label='Team Reward', capsize=10)
    rects2 = ax.bar(ind, humanrew_means, width, yerr=humanrew_stds,
                    label='Human Reward', capsize=10)
    rects3 = ax.bar(ind + width, robotrew_means, width, yerr=robotrew_stds,
                    label='Robot Reward', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    ax.set_ylim(-0.00, 1.5)

    plt.yticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0], [0.0,0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # plt.xticks([])

    ax.set_title('Assistance w/ Hidden RC', fontsize=16)
    ax.set_xticks(ind, fontsize=14)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'), fontsize=13)
    # ax.legend(fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    autolabel(ax, rects1, "left")
    autolabel(ax, rects2, "right")
    autolabel(ax, rects3, "right")

    fig.tight_layout()
    plt.savefig("greedy_public_iterative_100_multiprocess_testingcases.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()

