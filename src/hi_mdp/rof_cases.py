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
from robot_tom_agent_dualpf import Robot_Model
# from robot_tom_agent import Robot_Model
from multiprocessing import Pool, freeze_support
import os
import pickle
import json
from check_greedy_vs_altruistic import compare_optimal_to_greedy, Joint_MDP


import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RingOfFire():
    def __init__(self, robot, human, h_player_idx, r_player_idx, start_state, log_filename, to_plot):
        self.robot = robot
        self.human = human
        self.log_filename = log_filename
        self.total_reward = 0
        self.rew_history = []
        self.start_state = copy.deepcopy(start_state)
        self.to_plot = to_plot
        self.h_idx = h_player_idx
        self.r_idx = r_player_idx
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

        if self.r_idx == 0:
            robot_action = self.robot.act(self.state, self.robot_history, self.human_history, iteration)

            # update state and human's model of robot
            robot_state = copy.deepcopy(self.state)
            rew = 0
            rew_pair = [0, 0]
            if self.state[robot_action] > 0:
                self.state[robot_action] -= 1
                rew += self.robot.ind_rew[robot_action]
                rew_pair[0] = self.robot.ind_rew[robot_action]

            # have the human act
            human_action = self.human.act(self.state, self.robot_history, self.human_history)

            # update state and human's model of robot
            human_state = copy.deepcopy(self.state)
            if self.state[human_action] > 0:
                self.state[human_action] -= 1
                rew += self.human.ind_rew[human_action]
                rew_pair[1] = self.human.ind_rew[human_action]
        else:
            # have the human act
            human_action = self.human.act(self.state, self.robot_history, self.human_history)

            # update state and human's model of robot
            human_state = copy.deepcopy(self.state)
            rew = 0
            rew_pair = [0, 0]
            if self.state[human_action] > 0:
                self.state[human_action] -= 1
                rew += self.human.ind_rew[human_action]
                rew_pair[0] = self.human.ind_rew[human_action]

            robot_action = self.robot.act(self.state, self.robot_history, self.human_history, iteration)

            # update state and human's model of robot
            robot_state = copy.deepcopy(self.state)
            if self.state[robot_action] > 0:
                self.state[robot_action] -= 1
                rew += self.robot.ind_rew[robot_action]
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

        if self.to_plot:
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

            if self.to_plot:
                self.plot_game_actions(axs[iteration_count, 0], iteration_count, self.total_reward, robot_history, human_history)
                self.plot_robot_beliefs(axs[iteration_count, 1], axs[iteration_count, 2], axs[iteration_count, 3])
                self.plot_true_human_beliefs(axs[iteration_count, 4], axs[iteration_count, 5])

            iteration_count += 1

        with open(self.log_filename, 'a') as f:
            f.write(f"\nfinal_reward = {self.total_reward}")
            f.write('\n')

        if self.to_plot:
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


def run_k_rounds(list_of_start_states, mm_order, seed, num_rounds, vi_type, true_human_order, robot_rewards,
                 human_rewards, num_particles, exp_path, h_scalar, r_scalar, to_plot, h_player_idx, r_player_idx):
    # print(f"running seed {seed} for mm_order {mm_order} ")
    np.random.seed(seed)
    log_filename = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/' + 'log.txt'
    robot_agent = Robot_Model(robot_rewards, mm_order=mm_order, vi_type=vi_type, num_particles=num_particles,
                              h_scalar=h_scalar, r_scalar=r_scalar, log_filename=log_filename)

    true_human_agent = Human_Hypothesis(human_rewards, true_human_order, num_particles=num_particles, r_scalar=r_scalar,
                                        log_filename=log_filename)
    # true_human_agent = True_Human_Model(human_rewards, true_human_order, num_particles=num_particles)

    rof_game = RingOfFire(robot_agent, true_human_agent, h_player_idx, r_player_idx, list_of_start_states[0], log_filename, to_plot)
    # rof_game.run_full_game()

    collective_scores = {x: [] for x in range(num_rounds)}

    if to_plot:
        fig, axs = plt.subplots(num_rounds, 6, figsize=(30, 28))
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

        if to_plot:
            rof_game.plot_game_actions(axs[round, 0], round, total_rew, robot_history, human_history)
            rof_game.plot_robot_beliefs(axs[round, 1], axs[round, 2], axs[round, 3])
            rof_game.plot_true_human_beliefs(axs[round, 4], axs[round, 5])

    if to_plot:
        savefolder = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/'
        plt.savefig(savefolder + 'multi_round_results.png')
        plt.close()
    # for round in range(1):
    #     with open(log_filename, 'a') as f:
    #         f.write(f"\nROUND = {round}")
    #         f.write('\n')
    #     # print(f"round = {round}")
    #
    #     rof_game.reset_given_start(list_of_start_states[1])
    #     total_rew1 = rof_game.run_full_game()
    #     print(f"total_rew for start {list_of_start_states[1]} = {total_rew1}")
    #
    #     rof_game.reset_given_start(list_of_start_states[2])
    #     total_rew2 = rof_game.run_full_game()
    #     print(f"total_rew for start {list_of_start_states[2]} = {total_rew2}")
    #     collective_scores[num_rounds].append(total_rew1 + total_rew2)

    # print("Plotting Weight Updates")
    if to_plot:
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
                   exp_path, num_rounds, h_scalar, r_scalar, to_plot, h_player_idx, r_player_idx):
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

    seed_val = list_of_random_seeds[0]
    # first_order_scores = run_k_rounds(list_of_start_states, 'first', seed_val, num_rounds, vi_type,
    #                                                    true_human_order, robot_rewards, human_rewards, num_particles,
    #                                                    exp_path, h_scalar, r_scalar, to_plot)

    # second_order_scores = run_k_rounds(list_of_start_states, 'second', seed_val, num_rounds, vi_type,
    #                                    true_human_order, robot_rewards, human_rewards, num_particles,
    #                                    exp_path, h_scalar, r_scalar, to_plot)

    both_order_scores = run_k_rounds(list_of_start_states, 'both', seed_val, num_rounds, vi_type,
                                       true_human_order, robot_rewards, human_rewards, num_particles,
                                       exp_path, h_scalar, r_scalar, to_plot, h_player_idx, r_player_idx)

    # for result in first_order_scores:
    # for round_no in first_order_scores:
    #     first_results[round_no].extend(first_order_scores[round_no])

    # for result in second_order_scores:
    # for round_no in second_order_scores:
    #     second_results[round_no].extend(second_order_scores[round_no])

    # for result in both_order_scores:
    for round_no in both_order_scores:
        both_results[round_no].extend(both_order_scores[round_no])

    # experiment_results[1] = first_results
    # experiment_results[2] = second_results
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
        # print("Saved experiment params")

    savefile = "exp_results/" + exp_path + '/' + vi_type + '/pkl_results' + '/results.pkl'
    with open(savefile, 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print("Saved pickled experiment results")

    savefile = "exp_results/" + exp_path + '/' + vi_type + '/agg_imgs' + '/scores_over_time.png'
    if to_plot:
        plot_results(experiment_results, num_rounds, true_human_order, savefile)
    # print("Plotted experiment results")
    #
    # print(f"Model {vi_type}: Results:")

    final_result = {}
    # end_rews = [elem for elem in experiment_results[1][num_rounds-1]]
    # print("experiment_results", experiment_results)
    # end_rews = []
    # for r_no in range(num_rounds):
    #     for elem in experiment_results[1][r_no]:
    #         end_rews.append(elem)
    # # print("First only = ", (np.mean(end_rews), np.std(end_rews)))
    # # final_result['first'] = np.mean(end_rews)
    # final_result['first'] = max(end_rews)
    #
    # # end_rews = [elem for elem in experiment_results[2][num_rounds - 1]]
    # end_rews = []
    # for r_no in range(num_rounds):
    #     for elem in experiment_results[2][r_no]:
    #         end_rews.append(elem)
    # # print("Second only = ", (np.mean(end_rews), np.std(end_rews)))
    # # final_result['second'] = np.mean(end_rews)
    # final_result['second'] = max(end_rews)

    # end_rews = [elem for elem in experiment_results[3][num_rounds - 1]]
    end_rews = []
    for r_no in range(num_rounds):
        for elem in experiment_results[3][r_no]:
            end_rews.append(elem)
    # print("Both = ", (np.mean(end_rews), np.std(end_rews)))
    # final_result['both'] = np.mean(end_rews)
    final_result['both'] = max(end_rews)+10
    # final_result['both'] = end_rews[-1] + 10
    return final_result

def run_ablation(start_state, robot_rewards, human_rewards, to_plot):

    h_player_idx = np.random.choice([0, 1])
    r_player_idx = 1 - h_player_idx
    if h_player_idx == 0:
        players_to_reward = [human_rewards, robot_rewards]
    else:
        players_to_reward = [robot_rewards, human_rewards]

    optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances = compare_optimal_to_greedy(
        players_to_reward, start_state, h_player_idx, r_player_idx)

    game_type = 'greedy'
    if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
        game_type = 'mutual'
    elif human_altruism_n_instances > 0:
        game_type = 'human'
    elif robot_altruism_n_instances > 0:
        game_type = 'robot'
    else:
        game_type = 'greedy'

    start_seed = np.random.randint(0, 10000)
    number_of_seeds = 1
    true_human_order = 1
    h_scalar = 1
    r_scalar = 1



    list_of_start_states = [start_state]

    num_particles = 500
    num_rounds = 10
    vi_types = ['mmvi', 'stdvi']
    # vi_types = ['mmvi']

    path = f"random_trials/DEBUG7_start-{start_state}_hscalar-{h_scalar}_rscalar-{r_scalar}_Hrew-{human_rewards}_Rrew-{robot_rewards}_horder-{true_human_order}_nparticles-{num_particles}_nrounds-{num_rounds}_nseeds-{number_of_seeds}_startseed-{start_seed}"
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

        # print(f"Path created at {path}")

    if start_seed is not None:
        np.random.seed(start_seed)

    vi_type_to_result = {}
    for vi_type in vi_types:
        final_result = run_experiment(list_of_start_states=list_of_start_states, vi_type=vi_type, n_seeds = number_of_seeds, true_human_order=true_human_order,
                       robot_rewards=robot_rewards, human_rewards=human_rewards, num_particles=num_particles, exp_path=path,
                       num_rounds=num_rounds, h_scalar=h_scalar, r_scalar=r_scalar, to_plot=to_plot,
                                      h_player_idx=h_player_idx, r_player_idx=r_player_idx)
        vi_type_to_result[vi_type] = {}
        vi_type_to_result[vi_type]['result'] = final_result
        vi_type_to_result[vi_type]['game_type'] = game_type
        # optimal_rew, greedy_rew,
        vi_type_to_result[vi_type]['optimal_rew'] = optimal_rew
        vi_type_to_result[vi_type]['greedy_rew'] = greedy_rew

    # print("vi_type_to_result", vi_type_to_result)
    return vi_type_to_result


def run_random_start(random_iter, to_plot):
    start_state = randomList(4, 10)

    corpus = (np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1),
              np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1))
    robot_rewards = corpus
    human_rewards = list(corpus)
    random.shuffle(human_rewards)
    human_rewards = tuple(human_rewards)



    # print("\n\n")
    print("Running experiment ", random_iter)
    # print("start_state =", start_state)
    # print("robot_rewards = ", robot_rewards)
    # print("human_rewards = ", human_rewards)
    vi_type_to_result = run_ablation(start_state, robot_rewards, human_rewards, to_plot)
    print("Done running experiment ", random_iter)
    return vi_type_to_result


def plot_diffs_boxplot(data, num_iters, savename):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, vert=0)

    # x-axis labels
    ax.set_yticklabels(['both'])

    # Adding title
    plt.title(f"Diff MIRL minus VI: Nseeds={num_iters}")

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    plt.savefig(savename)
    plt.show()


def test_plot(better_bools, percents, savename):



    num_times_better = 0
    num_times_worse = 0
    num_times_same = 0
    for i in range(len(better_bools)):
        if better_bools[i] == 1.0:
            num_times_better += 1
        elif better_bools[i] == 0.0:
            num_times_same += 1
        else:
            num_times_worse += 1

    num_times_better /= len(better_bools)
    num_times_worse /= len(better_bools)
    num_times_same /= len(better_bools)

    plt.bar(['better', 'same', 'worse'], [num_times_better, num_times_same, num_times_worse], color='maroon',
            width=0.4)

    plt.xlabel("Outcomes")
    plt.ylabel(f"Percent of Instances in {len(better_bools)} seeds")
    plt.title("Game Outcomes")
    plt.savefig(savename + '_outcomes.png')
    plt.close()

    plt.hist(percents, bins=50)
    plt.xlabel("Percent Improvement")
    plt.ylabel(f"Counts in {len(better_bools)} seeds")
    plt.title("Game Percent Improvements")
    plt.savefig(savename + '_percent_improvements.png')


def main():
    vi_types = ['mmvi', 'stdvi']
    diffs_both = []
    diffs_first = []
    diffs_second = []

    percents_both = []

    num_iters = 500
    to_plot = False

    with Pool(processes=100) as pool:
        trial_results = pool.starmap(run_random_start, [(i, to_plot) for i in range(num_iters)])


    cvi_gametype_to_list_percent_of_possible_improvement_achieved = {'mutual': [], 'robot': [], 'human': [], 'greedy': []}
    stdvi_gametype_list_percent_of_possible_improvement_achieved = {'mutual': [], 'robot': [], 'human': [], 'greedy': []}

    for vi_type_to_dict in trial_results:
        # print("vi_type_to_dict", vi_type_to_dict)
        game_type = vi_type_to_dict['mmvi']['game_type']

        # if game_type == 'greedy'
        optimal_rew = vi_type_to_dict['mmvi']['optimal_rew']
        greedy_rew = vi_type_to_dict['mmvi']['greedy_rew']

        cvi_rew = vi_type_to_dict['mmvi']['result']['both']
        stdvi_rew = vi_type_to_dict['stdvi']['result']['both']

        # print("game_type = ", game_type)
        # print("optimal_rew = ", optimal_rew)
        # print("greedy_rew = ", greedy_rew)
        # print("cvi_rew = ", cvi_rew)
        # print("stdvi_rew = ", stdvi_rew)

        # cvi_rew = vi_type_to_result['mmvi']['both']
        # stdvi_rew = vi_type_to_result['stdvi']['both']

        # Do for CVI
        if game_type != 'greedy':
            if cvi_rew >= greedy_rew:
                improved_diff = cvi_rew - greedy_rew
                max_possible_diff = abs(optimal_rew - greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                    # max_possible_diff += 1
                    # improved_diff += 1
                # print(f"improved_diff = {improved_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = improved_diff/max_possible_diff
                cvi_gametype_to_list_percent_of_possible_improvement_achieved[game_type].append(percent_of_possible_improvement_achieved)

            elif cvi_rew < greedy_rew:
                worsened_diff = greedy_rew - cvi_rew
                max_possible_diff = abs(optimal_rew - greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                    # max_possible_diff += 1
                    # worsened_diff += 1
                # print(f"worsened_diff = {worsened_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = worsened_diff/max_possible_diff
                cvi_gametype_to_list_percent_of_possible_improvement_achieved[game_type].append(-percent_of_possible_improvement_achieved)

        else:
            if cvi_rew == greedy_rew:
                percent_of_possible_improvement_achieved = 0
                cvi_gametype_to_list_percent_of_possible_improvement_achieved[game_type].append(
                    percent_of_possible_improvement_achieved)

            elif cvi_rew < greedy_rew:
                worsened_diff = greedy_rew - cvi_rew
                max_possible_diff = abs(greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                    # max_possible_diff += 1
                    # worsened_diff += 1
                # print(f"worsened_diff = {worsened_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = worsened_diff / max_possible_diff
                cvi_gametype_to_list_percent_of_possible_improvement_achieved[game_type].append(
                    -percent_of_possible_improvement_achieved)

        # Do the same for Std VI
        if game_type != 'greedy':
            if stdvi_rew >= greedy_rew:
                improved_diff = stdvi_rew - greedy_rew
                max_possible_diff = abs(optimal_rew - greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                # print(f"improved_diff = {improved_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = improved_diff/max_possible_diff
                stdvi_gametype_list_percent_of_possible_improvement_achieved[game_type].append(percent_of_possible_improvement_achieved)

            elif stdvi_rew < greedy_rew:
                worsened_diff = greedy_rew - stdvi_rew
                max_possible_diff = abs(optimal_rew - greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                    # max_possible_diff += 1
                    # worsened_diff += 1
                # print(f"worsened_diff = {worsened_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = worsened_diff/max_possible_diff
                stdvi_gametype_list_percent_of_possible_improvement_achieved[game_type].append(-percent_of_possible_improvement_achieved)

        else:
            if stdvi_rew == greedy_rew:
                percent_of_possible_improvement_achieved = 0
                stdvi_gametype_list_percent_of_possible_improvement_achieved[game_type].append(
                    percent_of_possible_improvement_achieved)

            elif stdvi_rew < greedy_rew:
                worsened_diff = greedy_rew - stdvi_rew
                max_possible_diff = abs(greedy_rew)
                if max_possible_diff < 0.01:
                    max_possible_diff = 1
                    # max_possible_diff += 1
                    # worsened_diff += 1
                # print(f"worsened_diff = {worsened_diff}, max_possible_diff = {max_possible_diff}")
                percent_of_possible_improvement_achieved = worsened_diff / max_possible_diff
                stdvi_gametype_list_percent_of_possible_improvement_achieved[game_type].append(
                    -percent_of_possible_improvement_achieved)


    game_types = ['mutual', 'robot', 'human', 'greedy']
    # cvi_means = [np.mean(cvi_gametype_to_list_percent_of_possible_improvement_achieved[gtype]) for gtype in game_types]
    # cvi_stds = [np.std(cvi_gametype_to_list_percent_of_possible_improvement_achieved[gtype]) for gtype in game_types]
    #
    # stdvi_means = [np.mean(stdvi_gametype_list_percent_of_possible_improvement_achieved[gtype]) for gtype in game_types]
    # stdvi_stds = [np.std(stdvi_gametype_list_percent_of_possible_improvement_achieved[gtype]) for gtype in game_types]

    data_a = [cvi_gametype_to_list_percent_of_possible_improvement_achieved[gtype] for gtype in game_types]
    data_b = [stdvi_gametype_list_percent_of_possible_improvement_achieved[gtype] for gtype in game_types]

    ticks = ['mutual', 'robot', 'human', 'greedy']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    bpl = plt.boxplot(data_a, positions=np.array(np.arange(len(data_a))) * 2.0 - 0.4, widths=0.6, showmeans=True,patch_artist=True,
                      medianprops=dict(linestyle='-', linewidth=4, color='purple'),
                      meanprops=dict(linestyle='--', linewidth=8, color='black'),
                      boxprops=dict(facecolor='pink', color='black'),)
    bpr = plt.boxplot(data_b, positions=np.array(np.arange(len(data_b))) * 2.0 + 0.4, widths=0.6, showmeans=True,patch_artist=True,
                      medianprops=dict(linestyle='-', linewidth=4, color='purple'),
                      meanprops=dict(linestyle='--', linewidth=8, color='black'),
                      boxprops=dict(facecolor='lightblue', color='black'),)

    # set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
    # set_box_color(bpr, '#2C7BB6')

    # for median in bpl['medians']:
    #     median.set_color('black')
    #
    # for median in bpl['means']:
    #     median.set_color('green')
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for patch, color in zip(bpl['boxes'], colors):
    #     patch.set_facecolor('red')
    # for patch, color in zip(bpr['boxes'], colors):
    #     patch.set_facecolor('blue')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='CVI')
    plt.plot([], c='#2C7BB6', label='StdVI')
    plt.legend()

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    # plt.xlim(-2, len(ticks) * 2)
    # plt.ylim(-1.3, 1.3)
    plt.xlabel("Percent of Possible Improvement Achieved")
    plt.ylabel("Game Type")
    plt.title("Percent of Boost by Game Type")
    plt.tight_layout()
    plt.savefig(f'randorder_h0_maxgame_percent_improve_niters-{num_iters}.png')
    plt.show()




def sample_configurations():
    blockPrint()
    total = 1000
    n_mutual = 0
    n_human = 0
    n_robot = 0
    n_greedy = 0

    for i in range(total):
        # print("i = ", i)
        initial_state = randomList(4, 10)

        corpus = (np.round(np.random.uniform(-1,1), 1), np.round(np.random.uniform(-1,1), 1),
                  np.round(np.random.uniform(-1,1), 1), np.round(np.random.uniform(-1,1), 1))

        robot_rewards = corpus
        human_rewards = list(corpus)
        random.shuffle(human_rewards)
        human_rewards = tuple(human_rewards)
        players_to_reward = [robot_rewards, human_rewards]

        print(f"initial_state = {initial_state}, players_to_reward={players_to_reward}")

        optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances = compare_optimal_to_greedy(
            players_to_reward, initial_state)

        if i % 100==0:
            # enablePrint()
            print()
            print("i = ", i)
            print(f"optimal_rew = {optimal_rew}, greedy_rew = {greedy_rew}")
            if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
                print("mutual")
            elif human_altruism_n_instances > 0:
                print("human altruism")
            elif robot_altruism_n_instances > 0:
                print("robot altruism")
            else:
                print("greedy optimal")
            # blockPrint()

        if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
            n_mutual += 1
        elif human_altruism_n_instances > 0:
            n_human += 1
        elif robot_altruism_n_instances > 0:
            n_robot += 1
        else:
            print(f"optimal_rew = {optimal_rew}, greedy_rew = {greedy_rew}")
            eps = 0.01
            assert abs(optimal_rew - greedy_rew) < eps
            n_greedy += 1

    enablePrint()
    n_mutual /= total
    n_human /= total
    n_robot /= total
    n_greedy /= total

    plt.bar(['mutual', 'human', 'robot', 'greedy'], [n_mutual, n_human, n_robot, n_greedy], color='maroon',
            width=0.4)

    plt.xlabel("Game Type")
    plt.ylabel(f"Percent of Instances in {total} rounds")
    plt.title("Frequency of Game Types")
    plt.savefig(f"freq_game_types_nrounds-{total}.png")
    plt.close()

if __name__ == "__main__":
    # sample_configurations()
    main()









