import copy
import pdb

import numpy as np
import operator
import random
import matplotlib.pyplot as plt
import scipy
from scipy.stats import sem
import matplotlib
BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow'}

from true_human_agent import True_Human_Model
from human_hypothesis import Human_Hypothesis
from himdp import HiMDP
from robot_tom_agent import Robot_Model
from multiprocessing import Pool, freeze_support
import os
import pickle
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RingOfFire():
    def __init__(self, robot, human, start_state, log_filename):
        self.robot = robot
        self.human = human
        self.log_filename = log_filename
        self.total_reward = 0
        self.rew_history = []
        self.start_state = copy.deepcopy(start_state)
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

    def step(self, iteration, no_update=False):
        # have the robot act
        # pdb.set_trace()
        with open(self.log_filename, 'a') as f:
            f.write(f"\n\nCurrent state = {self.state}")

        robot_action = self.robot.act(self.state, self.robot_history, self.human_history, iteration)

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

        # have the human act
        human_action = self.human.act(self.state, self.robot_history, self.human_history)

        # update state and human's model of robot
        human_state = copy.deepcopy(self.state)
        if self.state[human_action] > 0:
            self.state[human_action] -= 1
            rew += self.human.ind_rew[human_action]
            rew_pair[1] = self.robot.ind_rew[robot_action]

        # if no_update is False:
        # update_flag = self.is_done()
        update_flag = False
        self.human.update_with_partner_action(robot_state, robot_action, update_flag)
        self.robot.update_human_models_with_robot_action(robot_state, robot_action, update_flag)
        self.robot_history.append(robot_action)

        self.robot.update_with_partner_action(human_state, human_action, update_flag)
        self.human_history.append(human_action)

        idx_to_text = {BLUE: 'blue', GREEN: 'green', RED:'red', YELLOW:'yellow'}
        # if iteration == 0:
            # print()
        # print(f"robot {idx_to_text[robot_action]}, human {idx_to_text[human_action]} --> {self.state}")
        if self.log_filename is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nrobot {idx_to_text[robot_action]}, human {idx_to_text[human_action]} --> {self.state}")

        return self.state, rew, rew_pair, self.is_done(), robot_action, human_action

    def run_full_game(self, round, no_update=False):
        self.reset()
        iteration_count = 0
        robot_history = []
        human_history = []

        num_objects = int(sum(self.start_state)/2)
        fig, axs = plt.subplots(num_objects, 6, figsize=(30, 28))
        fig.tight_layout(pad=5.0)
        # self.total_reward = 0

        while self.is_done() is False:
            _, rew, rew_pair, _, r_action, h_action = self.step(iteration=iteration_count, no_update=no_update)
            self.rew_history.append(rew_pair)
            self.total_reward += rew

            robot_history.append(r_action)
            human_history.append(h_action)

            self.plot_game_actions(axs[iteration_count, 0], iteration_count, self.total_reward, robot_history, human_history)
            self.plot_robot_beliefs(axs[iteration_count, 1], axs[iteration_count, 2], axs[iteration_count, 3])
            self.plot_true_human_beliefs(axs[iteration_count, 4], axs[iteration_count, 5])

            iteration_count += 1

        with open(self.log_filename, 'a') as f:
            f.write(f"\nfinal_reward = {self.total_reward}")
            f.write('\n')

        savefolder = self.log_filename.split("log.txt")[0]
        plt.savefig(savefolder + f'single_round_result_{round}.png')
        plt.close()
        # self.robot.resample()
        # self.human.resample()

        # print("self.rew_history", self.rew_history)
        # print("self.total_reward", self.total_reward)
        return self.total_reward, robot_history, human_history

    def plot_game_actions(self, ax, round, total_rew, robot_history, human_history):
        ax.set_xlim([0, 600])
        ax.set_ylim([0, 400])

        ax.set_xticks([])
        ax.set_yticks([])

        robot_circle = plt.Circle((50, 300), radius=30, color='#ffcccc')
        ax.add_patch(robot_circle)
        label = ax.annotate("R", xy=(50, 280), fontsize=8, ha="center")

        human_circle = plt.Circle((50, 110), radius=30, color='#86b6e2')
        ax.add_patch(human_circle)
        label = ax.annotate("H", xy=(50, 90), fontsize=8, ha="center")

        robot_x = 120
        robot_y = 280
        for i in range(len(robot_history)):
            rect = matplotlib.patches.Rectangle(((robot_x + 70*i), robot_y), 50, 50, color=COLOR_TO_TEXT[robot_history[i]])
            ax.add_patch(rect)

        human_x = 170
        human_y = 90
        for i in range(len(human_history)):
            rect = matplotlib.patches.Rectangle(((human_x + 70 * i), human_y), 50, 50,
                                                color=COLOR_TO_TEXT[human_history[i]])
            ax.add_patch(rect)

        ax.title.set_text(f'Round {round}: Reward={np.round(total_rew, 2)}')
        return

    def plot_robot_beliefs(self, ax1, ax2, ax3):
        self.robot.plot_beliefs_to_axes(ax1, ax2, ax3)


    def plot_true_human_beliefs(self, ax1, ax2):
        self.human.plot_beliefs_to_axes(ax1, ax2)



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
    # print("saved to: ", savename)
    plt.close()


def run_k_rounds(list_of_start_states, mm_order, seed, num_rounds, vi_type, true_human_order, robot_rewards, human_rewards, num_particles, exp_path, h_scalar, r_scalar):
    # print(f"running seed {seed} for mm_order {mm_order} ")
    np.random.seed(seed)
    log_filename = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/' + 'log.txt'
    robot_agent = Robot_Model(robot_rewards, mm_order=mm_order, vi_type=vi_type, num_particles=num_particles,
                              h_scalar=h_scalar, r_scalar=r_scalar, log_filename=log_filename)

    true_human_agent = Human_Hypothesis(human_rewards, true_human_order, num_particles=num_particles, r_scalar=r_scalar,
                                        log_filename=log_filename)
    # true_human_agent = True_Human_Model(human_rewards, true_human_order, num_particles=num_particles)

    rof_game = RingOfFire(robot_agent, true_human_agent, list_of_start_states[0], log_filename)
    # rof_game.run_full_game()
    n_rounds_plot = num_rounds+2
    collective_scores = {x: [] for x in range(n_rounds_plot)}

    fig, axs = plt.subplots(n_rounds_plot, 6, figsize=(30, 28))
    # fig.tight_layout()
    fig.tight_layout(pad=5.0)
    # plt.subplots_adjust(wspace=None, hspace=None)

    for round in range(num_rounds):
        with open(log_filename, 'a') as f:
            f.write(f"\nROUND = {round}")
            f.write('\n')
        # print(f"round = {round}")
        rof_game.reset()
        total_rew, robot_history, human_history = rof_game.run_full_game(round)
        collective_scores[round].append(total_rew)

        rof_game.plot_game_actions(axs[round, 0], round, total_rew, robot_history, human_history)
        rof_game.plot_robot_beliefs(axs[round, 1], axs[round, 2], axs[round, 3])
        rof_game.plot_true_human_beliefs(axs[round, 4], axs[round, 5])

    for i in range(1, len(list_of_start_states)):
        round_n = num_rounds + (i-1)
        rof_game.reset_given_start(list_of_start_states[i])
        total_rew, robot_history, human_history = rof_game.run_full_game(round_n)
        collective_scores[num_rounds].append(total_rew)
        rof_game.plot_game_actions(axs[round_n, 0], round_n, total_rew, robot_history, human_history)
        rof_game.plot_robot_beliefs(axs[round_n, 1], axs[round_n, 2], axs[round_n, 3])
        rof_game.plot_true_human_beliefs(axs[round_n, 4], axs[round_n, 5])



    savefolder = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/'
    plt.savefig(savefolder + 'multi_round_results.png')
    plt.close()
    # for round in range(1):
    #     with open(log_filename, 'a') as f:
    #         f.write(f"\nROUND = {round}")
    #         f.write('\n')
    #     # print(f"round = {round}")
    #
    # rof_game.reset_given_start(list_of_start_states[1])
    # total_rew1 = rof_game.run_full_game(num_rounds)
    # print(f"total_rew for start {list_of_start_states[1]} = {total_rew1}")
    # #
    # rof_game.reset_given_start(list_of_start_states[2])
    # total_rew2 = rof_game.run_full_game(num_rounds+1)
    # print(f"total_rew for start {list_of_start_states[2]} = {total_rew2}")
    #     collective_scores[num_rounds].append(total_rew1 + total_rew2)

    # print("Plotting Weight Updates")

    robot_agent.plot_weight_updates(savefolder + f"Robot_{seed}-seed_depth-rewards.png",
                                    savefolder + f"Robot_{seed}-seed_accuracy.png")
    robot_agent.plot_beliefs_of_humans(savefolder + f"RsHumanBeliefs_{seed}-seed_depth-rewards.png",
                                    savefolder + f"RsHumanBeliefs_{seed}-seed_accuracy.png")
    if true_human_order == 2:
        true_human_agent.plot_weight_updates(savefolder + f"TrueHuman_{seed}-seed_depth-rewards.png",
                                             savefolder + f"TrueHuman_{seed}-seed_accuracy.png")
    # print("Done Weight Updates")
    return collective_scores


def randomList(m, n):
    # Create an array of size m where
    # every element is initialized to 0
    arr = [0] * m

    # To make the sum of the final list as n
    for i in range(n):
        # Increment any random element
        # from the array by 1
        arr[random.randint(0, n) % m] += 1

    # Print the generated list
    return arr

def run_experiment(list_of_start_states, vi_type, n_seeds, true_human_order, robot_rewards, human_rewards, num_particles,
                   exp_path, num_rounds, h_scalar, r_scalar):
    # np.random.seed(0)
    num_seeds = n_seeds
    list_of_random_seeds = np.random.randint(0, 100000, num_seeds)
    # list_of_random_seeds = [76317, 76219, 83657, 54528, 81906, 70048, 89183, 82939, 98333, 52622]
    # robot_agent = Human_Model((0.9, -0.9, 0.1, 0.3), 2)
    # start_state = randomList(4, 10)
    # print("start_state = ", start_state)

    experiment_results = {}

    first_results = {x: [] for x in range(num_rounds)}
    second_results = {x: [] for x in range(num_rounds)}
    both_results = {x: [] for x in range(num_rounds)}

    with Pool(processes=8) as pool:
        second_order_scores = pool.starmap(run_k_rounds, [(list_of_start_states, 'second', seed_val, num_rounds, vi_type,
                                                           true_human_order, robot_rewards, human_rewards, num_particles,
                                                           exp_path, h_scalar, r_scalar) for seed_val in list_of_random_seeds])

        first_order_scores = pool.starmap(run_k_rounds, [(list_of_start_states, 'first', seed_val, num_rounds, vi_type,
                                                          true_human_order, robot_rewards, human_rewards, num_particles,
                                                          exp_path, h_scalar, r_scalar) for seed_val in list_of_random_seeds])

        both_order_scores = pool.starmap(run_k_rounds, [(list_of_start_states, 'both', seed_val, num_rounds, vi_type,
                                                         true_human_order, robot_rewards, human_rewards, num_particles,
                                                         exp_path, h_scalar, r_scalar) for seed_val in list_of_random_seeds])

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

    experiment_params = {'start_state':list_of_start_states,
                        'vi_type':vi_type,
                         'n_seeds':n_seeds,
                         'h_scalar':h_scalar,
                         'r_scalar': r_scalar,
                         'true_human_order':true_human_order,
                        'robot_rewards':robot_rewards,
                        'human_rewards':human_rewards,
                        'num_particles': num_particles,
                        'exp_path':exp_path,
                        'num_rounds':num_rounds,
                        'list_of_random_seeds': list_of_random_seeds}

    savefile = "exp_results/" + exp_path + '/' + vi_type + '/experiment_params.json'
    with open(savefile, 'w') as fp:
        # json.dumps(experiment_params, cls=NumpyEncoder)
        json.dump(experiment_params, fp, cls=NumpyEncoder)
        print("Saved experiment params")

    savefile = "exp_results/" + exp_path + '/' + vi_type + '/pkl_results' + '/results.pkl'
    with open(savefile, 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved pickled experiment results")

    savefile = "exp_results/" + exp_path + '/' + vi_type + '/agg_imgs' + '/scores_over_time.png'
    plot_results(experiment_results, num_rounds, true_human_order, savefile)
    print("Plotted experiment results")

    print(f"Model {vi_type}: Results:")

    end_rews = [elem for elem in experiment_results[1][num_rounds-1]]
    print("First only = ", (np.mean(end_rews), np.std(end_rews)))

    end_rews = [elem for elem in experiment_results[2][num_rounds - 1]]
    print("Second only = ", (np.mean(end_rews), np.std(end_rews)))

    end_rews = [elem for elem in experiment_results[3][num_rounds - 1]]
    print("Both = ", (np.mean(end_rews), np.std(end_rews)))


def run_ablation():
    start_seed = 0
    number_of_seeds = 1
    true_human_order = 2
    h_scalar = 1
    r_scalar = 1
    # human_rewards = (0.25, 0.25, 0.25, 0.25)  # (0.9, -0.9, 0.1, 0.3)
    # robot_rewards = (0.9, 0.9, -0.9, -0.9)  # (0.9, 0.1, -0.9, 0.2)
    start_state = [2,2,2,2]
    # start_state = [2,6,2,2]
    # human_rewards = (0.25, 0.25, 0.25, 0.25)  # (0.9, -0.9, 0.1, 0.3)
    # robot_rewards = (0.9, 0.9, -0.9, -0.9)  # (0.9, 0.1, -0.9, 0.2)

    # robot_rewards = (0.6, -0.9, 0.1, 0.3)
    # human_rewards = (0.9, 0.1, -0.9, 0.2)
    # human_rewards = (-0.9, -0.5, 0.5, 1.0)
    # robot_rewards = (1.0, -0.9, 0.5, -0.5)

    # robot_rewards = (-0.9, -0.5, 0.5, 1.0)
    # human_rewards = (1.0, -0.9, 0.5, -0.5)
    robot_rewards = (1, 1.1, 3, 1)
    human_rewards = (1, 3, 1.1, 1)

    list_of_start_states = [start_state, [1,1,0,0], [0,0,1,1]]
    # human_rewards = (-0.5, -0.9, 1.0, 0.5)
    # robot_rewards = (0.2, 0.1, -0.2, 0.9)
    # human_rewards = (0.4, 0.3, -0.1, -0.1)

    num_particles = 500
    num_rounds = 3
    vi_types = ['mmvi', 'stdvi']
    # vi_types = ['mmvi']

    path = f"DEBUG8_start-{start_state}_hscalar-{h_scalar}_rscalar-{r_scalar}_Hrew-{human_rewards}_Rrew-{robot_rewards}_horder-{true_human_order}_nparticles-{num_particles}_nrounds-{num_rounds}_nseeds-{number_of_seeds}_startseed-{start_seed}"
    # Check whether the specified path exists or not
    isExist = os.path.exists("exp_results/" + path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("exp_results/" + path)
        for vi_type in vi_types:
            os.makedirs("exp_results/" + path + '/' + vi_type)
            os.makedirs("exp_results/" + path + '/' + vi_type + '/weights_imgs')
            os.makedirs("exp_results/" + path + '/' + vi_type + '/weights_imgs' + '/' + 'first')
            os.makedirs("exp_results/" + path + '/' + vi_type + '/weights_imgs'+ '/' + 'second')
            os.makedirs("exp_results/" + path + '/' + vi_type + '/weights_imgs'+ '/' + 'both')

            os.makedirs("exp_results/" + path + '/' + vi_type + '/agg_imgs')
            os.makedirs("exp_results/" + path + '/' + vi_type + '/pkl_results')

        print(f"Path created at {path}")

    if start_seed is not None:
        np.random.seed(start_seed)
    for vi_type in vi_types:
        run_experiment(list_of_start_states=list_of_start_states, vi_type=vi_type, n_seeds = number_of_seeds, true_human_order=true_human_order,
                       robot_rewards=robot_rewards, human_rewards=human_rewards, num_particles=num_particles, exp_path=path,
                       num_rounds=num_rounds, h_scalar=h_scalar, r_scalar=r_scalar)


if __name__ == "__main__":
    run_ablation()













