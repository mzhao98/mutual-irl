import copy
import pdb

import numpy as np
import operator
import random
import matplotlib.pyplot as plt
import scipy
from scipy.stats import sem
BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

from true_human_agent import True_Human_Model
from human_hypothesis import Human_Hypothesis
from himdp import HiMDP
from robot_tom_agent import Robot_Model
from multiprocessing import Pool, freeze_support


class RingOfFire():
    def __init__(self, robot, human):
        self.robot = robot
        self.human = human
        self.total_reward = 0
        self.rew_history = []
        self.reset()

    def reset(self):
        self.initialize_game()
        self.total_reward = 0
        self.rew_history = []
        self.robot_history = []
        self.human_history = []

    def initialize_game(self):
        self.state = [2, 5, 1, 2]

    def is_done(self):
        if sum(self.state) == 0:
            return True
        return False

    def step(self, no_update=False):
        # have the robot act
        # pdb.set_trace()
        robot_action = self.robot.act(self.state, self.robot_history, self.human_history)

        # update state and human's model of robot
        robot_state = copy.deepcopy(self.state)
        rew = 0
        rew_pair = [0, 0]
        if self.state[robot_action] > 0:
            self.state[robot_action] -= 1
            rew += self.robot.ind_rew[robot_action]
            rew_pair[0] = self.robot.ind_rew[robot_action]

        # if no_update is False:
        self.human.update_with_partner_action(robot_state, robot_action)
        self.robot.update_human_models_with_robot_action(robot_state, robot_action)
        self.robot_history.append(robot_action)

        # have the human act
        human_action = self.human.act(self.state, self.robot_history, self.human_history)

        # update state and human's model of robot
        human_state = copy.deepcopy(self.state)
        if self.state[human_action] > 0:
            self.state[human_action] -= 1
            rew += self.human.ind_rew[human_action]
            rew_pair[1] = self.robot.ind_rew[robot_action]

        # if no_update is False:
        self.robot.update_with_partner_action(human_state, human_action)
        self.human_history.append(human_action)

        return self.state, rew, rew_pair, self.is_done()

    def run_full_game(self, no_update=False):
        self.reset()
        while self.is_done() is False:
            _, rew, rew_pair, _ = self.step(no_update)
            self.rew_history.append(rew_pair)
            self.total_reward += rew

        # self.robot.resample()
        # self.human.resample()

        # print("self.rew_history", self.rew_history)
        # print("self.total_reward", self.total_reward)
        return self.total_reward

def plot_results(experiment_results, num_rounds, true_human_order, savename="images/test.png"):

    means = []
    stds = []
    for i in range(num_rounds):
        means.append(np.mean(experiment_results[1][i]))
        stds.append(sem(experiment_results[1][i]))
    means = np.array(means)
    stds = np.array(stds)

    plt.plot(range(len(means)), means, label='First', c='g',linewidth=4, alpha=0.5)
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2, facecolor='g', edgecolor='g')

    means = []
    stds = []
    for i in range(num_rounds):
        means.append(np.mean(experiment_results[2][i]))
        stds.append(sem(experiment_results[2][i]))
    means = np.array(means)
    stds = np.array(stds)

    plt.plot(range(len(means)), means, label='Second', c='b',linewidth=6, alpha=0.4)
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2, facecolor='b', edgecolor='b')

    means = []
    stds = []
    for i in range(num_rounds):
        means.append(np.mean(experiment_results[3][i]))
        stds.append(sem(experiment_results[3][i]))
    means = np.array(means)
    stds = np.array(stds)

    plt.plot(range(len(means)), means, label='Both', c='r', linewidth=4, alpha=0.7)
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2, facecolor='r', edgecolor='r')

    plt.axhline(y=0.8, xmin=0, xmax=num_rounds - 1, linewidth=2, color='k', label="greedy")
    plt.axhline(y=3.0, xmin=0, xmax=num_rounds - 1, linewidth=2, color='m', label="maximum")

    plt.legend()
    plt.xlabel("Round Number")
    plt.ylabel("Collective Reward")
    plt.title(f"True Human Order d={true_human_order-1}")
    plt.savefig(savename)
    print("saved to: ", savename)
    plt.close()


def run_k_rounds(mm_order, seed, num_rounds, vi_type, true_human_order):
    print(f"running seed {seed} ")
    np.random.seed(seed)

    robot_agent = Robot_Model((0.9, -0.9, 0.1, 0.3), mm_order=mm_order, vi_type=vi_type)


    true_human_agent = True_Human_Model((0.9, 0.1, -0.9, 0.2), true_human_order)
    rof_game = RingOfFire(robot_agent, true_human_agent)
    rof_game.run_full_game()

    collective_scores = {x: [] for x in range(num_rounds)}
    for round in range(num_rounds):
        # print(f"round = {round}")
        rof_game.reset()
        total_rew = rof_game.run_full_game()
        collective_scores[round].append(total_rew)

    # robot_agent.plot_weight_updates(f"exp26_robot_weightupdates_mmorder{mm_order}_{seed}seed.png")
    return collective_scores


def run_experiment(vi_type, n_seeds, true_human_order):
    # np.random.seed(0)
    num_seeds = n_seeds
    list_of_random_seeds = np.random.randint(0, 100000, num_seeds)
    # list_of_random_seeds = [76317, 76219, 83657, 54528, 81906, 70048, 89183, 82939, 98333, 52622]
    # robot_agent = Human_Model((0.9, -0.9, 0.1, 0.3), 2)

    experiment_results = {}
    num_rounds = 10

    first_results = {x: [] for x in range(num_rounds)}
    second_results = {x: [] for x in range(num_rounds)}
    both_results = {x: [] for x in range(num_rounds)}

    with Pool(processes=8) as pool:
        first_order_scores = pool.starmap(run_k_rounds, [('first-only', seed_val, num_rounds, vi_type, true_human_order) for seed_val in list_of_random_seeds])
        second_order_scores = pool.starmap(run_k_rounds, [('second-only', seed_val, num_rounds, vi_type, true_human_order) for seed_val in list_of_random_seeds])

        both_order_scores = pool.starmap(run_k_rounds, [('both', seed_val, num_rounds, vi_type, true_human_order) for seed_val in list_of_random_seeds])

        for result in first_order_scores:
            for round_no in first_results:
                first_results[round_no].extend(result[round_no])

        for result in second_order_scores:
            for round_no in first_results:
                second_results[round_no].extend(result[round_no])

        for result in both_order_scores:
            for round_no in both_results:
                both_results[round_no].extend(result[round_no])

    experiment_results[1] = first_results
    experiment_results[2] = second_results
    experiment_results[3] = both_results

    plot_results(experiment_results, num_rounds, true_human_order, f"images/exp31_opt_true-order-{true_human_order}_{vi_type}-actual_100p_{num_seeds}-seeds.png")

    print(f"Model {vi_type}: Results:")

    end_rews = [elem for elem in experiment_results[1][num_rounds-1]]
    print("First only = ", (np.mean(end_rews), np.std(end_rews)))

    end_rews = [elem for elem in experiment_results[2][num_rounds - 1]]
    print("Second only = ", (np.mean(end_rews), np.std(end_rews)))

    end_rews = [elem for elem in experiment_results[3][num_rounds - 1]]
    print("Both = ", (np.mean(end_rews), np.std(end_rews)))


def run_ablation():
    number_of_seeds = 100
    true_human_order = 2
    run_experiment(vi_type='mmvi', n_seeds = number_of_seeds, true_human_order=true_human_order)
    run_experiment(vi_type='stdvi', n_seeds = number_of_seeds, true_human_order=true_human_order)
    run_experiment(vi_type='mmvi-nh', n_seeds= number_of_seeds, true_human_order=true_human_order)

if __name__ == "__main__":
    run_ablation()













