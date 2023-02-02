import pdb

import numpy as np
import operator
import copy
import random
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from multiprocessing import Pool, freeze_support

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

from compute_optimal_rew import OptimalMDP
from hip_mdp_1player import HiMDP
from human_agent import Human_Hypothesis

from human_agent import Human_Hypothesis
from robot_tom_agent import Robot_Model
from multiprocessing import Pool, freeze_support
import pickle
import json
import sys
import os

COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow', None:'none'}

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def randomList(m, n):
    # Create an array of size m where
    # every element is initialized to 0
    arr = [2] * m

    # To make the sum of the final list as n
    for i in range(n-(2*m)):
        # Increment any random element
        # from the array by 1
        arr[random.randint(0, n-(2*m)) % m] += 1

    # Print the generated list
    return arr

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RingOfFire():
    def __init__(self, first_player, robot, human, start_state, log_filename='', to_plot=False):
        self.first_player = first_player
        self.robot = robot
        self.human = human
        self.log_filename = log_filename
        self.total_reward = 0
        self.rew_history = []
        self.start_state = copy.deepcopy(start_state)
        self.to_plot = to_plot
        self.reset()

    def reset(self):
        self.initialize_game()
        self.total_reward = 0
        self.rew_history = []
        self.robot_history = []
        self.human_history = []

    def reset_given_start(self, start):
        self.start_state = start
        self.initialize_game()
        self.total_reward = 0
        self.rew_history = []
        self.robot_history = []
        self.human_history = []

    def initialize_game(self):
        self.state = copy.deepcopy(self.start_state)

    def is_done(self):
        if sum(self.state) == 0:
            return True
        return False

    def step_old(self, iteration, no_update=False):
        # have the robot act
        # pdb.set_trace()
        # with open(self.log_filename, 'a') as f:
        #     f.write(f"\n\nCurrent state = {self.state}")

        robot_action = self.robot.act(self.state, iteration)

        # update state and human's model of robot
        robot_state = copy.deepcopy(self.state)
        rew = 0
        rew_pair = [0, 0]
        if self.state[robot_action] > 0:
            self.state[robot_action] -= 1
            rew += self.robot.ind_rew[robot_action]
            rew_pair[0] = self.robot.ind_rew[robot_action]

        # if no_update is False:
        # self.human.update_with_partner_action(robot_state, robot_action)
        # self.robot.update_human_models_with_robot_action(robot_state, robot_action)
        # self.robot_history.append(robot_action)

        if self.is_done() is False:
            # have the human act
            human_action = self.human.act(self.state)

            # update state and human's model of robot
            human_state = copy.deepcopy(self.state)
            if self.state[human_action] > 0:
                self.state[human_action] -= 1
                rew += self.human.ind_rew[human_action]
                rew_pair[1] = self.robot.ind_rew[robot_action]

            # if no_update is False:
            # update_flag = self.is_done()
            update_flag = False
            self.human.update_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
            self.robot.update_human_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
            self.robot_history.append(robot_action)

            self.robot.update_robots_human_models_with_human_action(human_state, human_action, update_flag)
            self.human_history.append(human_action)

        else:
            human_action = None

        return self.state, rew, rew_pair, self.is_done(), robot_action, human_action

    def step(self, iteration, no_update=False):
        # have the robot act
        # pdb.set_trace()
        # with open(self.log_filename, 'a') as f:
        #     f.write(f"\n\nCurrent state = {self.state}")

        if self.first_player == 'r':
            robot_action = self.robot.act(self.state, iteration)

            # update state and human's model of robot
            robot_state = copy.deepcopy(self.state)
            rew = 0
            rew_pair = [0, 0]
            if self.state[robot_action] > 0:
                self.state[robot_action] -= 1
                rew += self.robot.ind_rew[robot_action]
                rew_pair[0] = self.robot.ind_rew[robot_action]

            # if no_update is False:
            # self.human.update_with_partner_action(robot_state, robot_action)
            # self.robot.update_human_models_with_robot_action(robot_state, robot_action)
            # self.robot_history.append(robot_action)

            if self.is_done() is False:
                # have the human act
                human_action = self.human.act(self.state)

                # update state and human's model of robot
                human_state = copy.deepcopy(self.state)
                if self.state[human_action] > 0:
                    self.state[human_action] -= 1
                    rew += self.human.ind_rew[human_action]
                    rew_pair[1] = self.human.ind_rew[human_action]

                # if no_update is False:
                # update_flag = self.is_done()
                update_flag = False
                self.human.update_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
                self.robot.update_human_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
                self.robot_history.append(robot_action)

                self.robot.update_robots_human_models_with_human_action(human_state, human_action, update_flag)
                self.human_history.append(human_action)

            else:
                human_action = None
        else:
            rew = 0
            rew_pair = [0, 0]

            # have the human act
            human_action = self.human.act(self.state)

            # update state and human's model of robot
            human_state = copy.deepcopy(self.state)
            if self.state[human_action] > 0:
                self.state[human_action] -= 1
                rew += self.human.ind_rew[human_action]
                rew_pair[0] = self.human.ind_rew[human_action]

            # if no_update is False:
            # update_flag = self.is_done()
            update_flag = False


            self.robot.update_robots_human_models_with_human_action(human_state, human_action, update_flag)
            self.human_history.append(human_action)

            if self.is_done() is False:


                robot_action = self.robot.act(self.state, iteration)

                # update state and human's model of robot
                robot_state = copy.deepcopy(self.state)

                if self.state[robot_action] > 0:
                    self.state[robot_action] -= 1
                    rew += self.robot.ind_rew[robot_action]
                    rew_pair[1] = self.robot.ind_rew[robot_action]

                self.human.update_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
                self.robot.update_human_beliefs_of_robot_with_robot_action(robot_state, robot_action, update_flag)
                self.robot_history.append(robot_action)

            else:
                robot_action = None

        return self.state, rew, rew_pair, self.is_done(), robot_action, human_action

    def run_full_game(self, round, no_update=False):
        self.reset()
        iteration_count = 0
        robot_history = []
        human_history = []

        human_only_reward = 0
        robot_only_reward = 0
        while self.is_done() is False:
            _, rew, rew_pair, _, r_action, h_action = self.step(iteration=iteration_count, no_update=no_update)
            # print(f"\nTimestep: {iteration_count}")
            # print(f"Current state: {self.state}")
            # print(f"Robot took action: {COLOR_TO_TEXT[r_action]}")
            # print(f"Robot took action: {COLOR_TO_TEXT[h_action]}")
            # print(f"Achieved reward: {rew_pair}")
            # print(f"Total reward: {self.total_reward}")

            self.rew_history.append(rew_pair)
            self.total_reward += rew
            if h_action is not None:
                human_only_reward += self.human.ind_rew[h_action]

            if r_action is not None:
                robot_only_reward += self.robot.ind_rew[r_action]

            robot_history.append(r_action)
            human_history.append(h_action)

            iteration_count += 1
        return self.total_reward, robot_history, human_history, human_only_reward, robot_only_reward


def compute_optimal_rew(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho):
    himdp = OptimalMDP(first_player, copy.deepcopy(start_state), all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
    himdp.enumerate_states()
    himdp.value_iteration()
    rew = himdp.rollout_full_game_vi_policy()
    human_best_rew = himdp.compute_max_human_reward()
    robot_best_rew = himdp.compute_max_robot_reward()
    altruism_case = himdp.compare_opt_to_greedy()
    return rew, human_best_rew, robot_best_rew, altruism_case

def run_exp_config_old(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, vi_type):
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


def run_exp_config(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, vi_type):
    robot_agent = Robot_Model(robot_rew, all_colors_list, task_reward, [1], vi_type)

    true_human_agent = Human_Hypothesis(human_rew, robot_rew, all_colors_list, task_reward, h_rho)
    # true_human_agent = True_Human_Model(human_rewards, true_human_order, num_particles=num_particles)

    rof_game = RingOfFire(first_player, robot_agent, true_human_agent, start_state)
    # rof_game.run_full_game()

    num_rounds = 10

    collective_scores = {x: [] for x in range(num_rounds)}

    max_rew = -100
    final_rew = -100
    final_human_rew = -100
    final_robot_rew = -100
    for round in range(num_rounds):
        # print(f"\n\nRound = {round}")
        rof_game.reset()
        total_rew, robot_history, human_history, human_only_reward, robot_only_reward = rof_game.run_full_game(round)
        collective_scores[round].append(total_rew)
        if total_rew > max_rew:
            max_rew = total_rew

        final_rew = total_rew
        final_human_rew = human_only_reward
        final_robot_rew = robot_only_reward

    # print("collective_scores", collective_scores)
    return final_rew, final_human_rew, final_robot_rew


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
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.09),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=14)

def run_one():
    first_player = 'r'
    start_state = [2, 2, 2]
    all_colors_list = [BLUE, GREEN, RED]
    task_reward = [1, 1, 1]
    human_rew = [0.5, 0.1, 0.5]
    h_rho = 1
    robot_rew = [0.5, 0.5, 0.1]
    r_rho = 1
    vi_type = 'cvi'

    optimal_rew = compute_optimal_rew(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
    cvi_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, 'cvi')

    print("cvi_rew", cvi_rew)



    # stdvi_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho, 'stdvi')

def run_k_rounds(exp, task_reward, r_rho, h_rho_of_interest):
    print("exp = ", exp)
    first_player = np.random.choice(['r', 'h'])
    start_state = [np.random.randint(1, 10), np.random.randint(1, 10), np.random.randint(1, 10),
                   np.random.randint(1, 10)]
    if sum(start_state) % 2 == 1:
        start_state[-1] -= 1
    all_colors_list = [BLUE, GREEN, RED, YELLOW]
    human_rew = [np.random.uniform(0.1, 20.0), np.random.uniform(0.1, 20.0), np.random.uniform(0.1, 20.0),
                 np.random.uniform(0.1, 20.0)]
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

    print("done w exp = ", exp)
    print("done with exp = ", exp)
    diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
    diff = np.round(diff, 2)

    if diff < 0:
        print()
        print("CVI less than StdVI")
        print("first_player = ", first_player)
        print("start_state = ", start_state)
        print("human_rew = ", human_rew)
        print("robot_rew = ", robot_rew)
        print("cvi_rew = ", cvi_rew)
        print("stdvi_rew = ", stdvi_rew)
        print()

    return cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
           cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case


if __name__ == "__main__":

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

    n_altruism = 0
    n_total = 0
    n_greedy = 0
    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    h_rho_of_interest = 1
    with Pool(processes=100) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward, r_rho, h_rho_of_interest) for exp_num in range(num_exps)])
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

    teamrew_means = [np.round(np.mean(cvi_percents[h_rho_of_interest]), 2),
                     np.round(np.mean(stdvi_percents[h_rho_of_interest]), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents[h_rho_of_interest]), 2),
                    np.round(np.std(stdvi_percents[h_rho_of_interest]), 2)]

    humanrew_means = [np.round(np.mean(cvi_humanrew_percents[h_rho_of_interest]), 2),
                      np.round(np.mean(stdvi_humanrew_percents[h_rho_of_interest]), 2)]
    humanrew_stds = [np.round(np.std(cvi_humanrew_percents[h_rho_of_interest]), 2),
                     np.round(np.std(stdvi_humanrew_percents[h_rho_of_interest]), 2)]

    robotrew_means = [np.round(np.mean(cvi_robotrew_percents[h_rho_of_interest]), 2),
                      np.round(np.mean(stdvi_robotrew_percents[h_rho_of_interest]), 2)]
    robotrew_stds = [np.round(np.std(cvi_robotrew_percents[h_rho_of_interest]), 2),
                     np.round(np.std(stdvi_robotrew_percents[h_rho_of_interest]), 2)]

    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    print("team rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_percents[h_rho_of_interest]]))
    print("human rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_humanrew_percents[h_rho_of_interest]]))

    print("robot rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_robotrew_percents[h_rho_of_interest]]))

    print("n_altruism = ", n_altruism)
    print("n_greedy = ", n_greedy)
    print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d] / sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.xlabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig("collab_private_iterative_100_multiprocess_cdf.png")
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

    ax.set_title('CIRL w/ RC', fontsize=16)
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
    plt.savefig("collab_private_iterative_100_multiprocessed.png")
    plt.show()

def non_multiprocessed():
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

    cvi_humanrew_percents = {0: [], 1: []}
    stdvi_humanrew_percents = {0: [], 1: []}

    cvi_robotrew_percents = {0: [], 1: []}
    stdvi_robotrew_percents = {0: [], 1: []}

    num_exps = 5
    h_rho_of_interest = 1
    for exp in range(num_exps):
        print("exp = ", exp)
        start_state = [2 + np.random.randint(0, 5), 2 + np.random.randint(0, 5), 2 + np.random.randint(0, 5),
                       2 + np.random.randint(0, 5)]
        if sum(start_state) % 2 == 1:
            start_state[-1] -= 1
        all_colors_list = [BLUE, GREEN, RED, YELLOW]
        human_rew = [np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0),
                     np.random.uniform(0.1, 10.0)]
        # robot_rew = [np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0), np.random.uniform(0.1, 10.0),
        #              np.random.uniform(0.1, 10.0)]
        permutes = list(itertools.permutations(human_rew))
        # print("permutes",permutes)
        robot_rew = list(permutes[np.random.choice(np.arange(len(permutes)))])

        print("TRUE human_rew = ", human_rew)
        print("TRUE robot_rew = ", robot_rew)

        for h_rho in [h_rho_of_interest]:
            optimal_rew, best_human_rew, best_robot_rew = compute_optimal_rew(first_player, start_state, all_colors_list, task_reward,
                                                              human_rew, h_rho, robot_rew, r_rho)
            cvi_rew, cvi_human_rew, cvi_robot_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho,
                                                    robot_rew, r_rho, 'cvi')
            stdvi_rew, stdvi_human_rew, stdvi_robot_rew = run_exp_config(start_state, all_colors_list, task_reward, human_rew, h_rho,
                                                        robot_rew, r_rho, 'stdvi')

            # print()
            # print("h_rho = ", h_rho)
            # print("optimal_rew", optimal_rew)
            # print("cvi_rew", cvi_rew)
            # print("stdvi_rew", stdvi_rew)
            print("cvi_rew = ", cvi_rew)
            print("stdvi_rew = ", stdvi_rew)

            cvi_percent_of_opt = cvi_rew / optimal_rew
            stdvi_percent_of_opt = stdvi_rew / optimal_rew
            cvi_percents[h_rho].append(cvi_percent_of_opt)
            stdvi_percents[h_rho].append(stdvi_percent_of_opt)

            cvi_percent_of_opt = cvi_human_rew / best_human_rew
            stdvi_percent_of_opt = stdvi_human_rew / best_human_rew
            cvi_humanrew_percents[h_rho].append(cvi_percent_of_opt)
            stdvi_humanrew_percents[h_rho].append(stdvi_percent_of_opt)

            cvi_percent_of_opt = cvi_robot_rew / best_robot_rew
            stdvi_percent_of_opt = stdvi_robot_rew / best_robot_rew
            cvi_robotrew_percents[h_rho].append(cvi_percent_of_opt)
            stdvi_robotrew_percents[h_rho].append(stdvi_percent_of_opt)

    teamrew_means = [np.round(np.mean(cvi_percents[h_rho_of_interest]), 2),
                      np.round(np.mean(stdvi_percents[h_rho_of_interest]), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents[h_rho_of_interest]), 2),
                     np.round(np.std(stdvi_percents[h_rho_of_interest]), 2)]

    humanrew_means = [np.round(np.mean(cvi_humanrew_percents[h_rho_of_interest]), 2),
                      np.round(np.mean(stdvi_humanrew_percents[h_rho_of_interest]), 2)]
    humanrew_stds = [np.round(np.std(cvi_humanrew_percents[h_rho_of_interest]), 2),
                     np.round(np.std(stdvi_humanrew_percents[h_rho_of_interest]), 2)]

    robotrew_means = [np.round(np.mean(cvi_robotrew_percents[h_rho_of_interest]), 2),
                      np.round(np.mean(stdvi_robotrew_percents[h_rho_of_interest]), 2)]
    robotrew_stds = [np.round(np.std(cvi_robotrew_percents[h_rho_of_interest]), 2),
                     np.round(np.std(stdvi_robotrew_percents[h_rho_of_interest]), 2)]

    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    print("team rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_percents[h_rho_of_interest]]))
    print("human rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_humanrew_percents[h_rho_of_interest]]))

    print("robot rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents[h_rho_of_interest]],
                          [elem * 100 for elem in stdvi_robotrew_percents[h_rho_of_interest]]))

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

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # plt.xticks([])

    ax.set_title('CIRL w/ RC', fontsize=16)
    ax.set_xticks(ind, fontsize=14)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'), fontsize=13)
    # ax.legend(fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    autolabel(ax, rects1, "center")
    autolabel(ax, rects2, "center")
    autolabel(ax, rects3, "center")

    fig.tight_layout()
    plt.savefig("collab_private_iterative_100.png")
    plt.show()