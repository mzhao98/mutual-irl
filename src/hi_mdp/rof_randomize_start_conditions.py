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



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RingOfFire():
    def __init__(self, robot, human, start_state, log_filename, to_plot):
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



def plot_results(experiment_results, num_rounds, true_human_order, savename="old_images/test.png"):

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
                 human_rewards, num_particles, exp_path, h_scalar, r_scalar, to_plot):
    # print(f"running seed {seed} for mm_order {mm_order} ")
    np.random.seed(seed)
    log_filename = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/' + 'log.txt'
    robot_agent = Robot_Model(robot_rewards, mm_order=mm_order, vi_type=vi_type, num_particles=num_particles,
                              h_scalar=h_scalar, r_scalar=r_scalar, log_filename=log_filename)

    true_human_agent = Human_Hypothesis(human_rewards, true_human_order, num_particles=num_particles, r_scalar=r_scalar,
                                        log_filename=log_filename)
    # true_human_agent = True_Human_Model(human_rewards, true_human_order, num_particles=num_particles)

    rof_game = RingOfFire(robot_agent, true_human_agent, list_of_start_states[0], log_filename, to_plot)
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
                   exp_path, num_rounds, h_scalar, r_scalar, to_plot):
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
    first_order_scores = run_k_rounds(list_of_start_states, 'first', seed_val, num_rounds, vi_type,
                                                       true_human_order, robot_rewards, human_rewards, num_particles,
                                                       exp_path, h_scalar, r_scalar, to_plot)

    second_order_scores = run_k_rounds(list_of_start_states, 'second', seed_val, num_rounds, vi_type,
                                       true_human_order, robot_rewards, human_rewards, num_particles,
                                       exp_path, h_scalar, r_scalar, to_plot)

    both_order_scores = run_k_rounds(list_of_start_states, 'both', seed_val, num_rounds, vi_type,
                                       true_human_order, robot_rewards, human_rewards, num_particles,
                                       exp_path, h_scalar, r_scalar, to_plot)

    # for result in first_order_scores:
    for round_no in first_order_scores:
        first_results[round_no].extend(first_order_scores[round_no])

    # for result in second_order_scores:
    for round_no in second_order_scores:
        second_results[round_no].extend(second_order_scores[round_no])

    # for result in both_order_scores:
    for round_no in both_order_scores:
        both_results[round_no].extend(both_order_scores[round_no])

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
    end_rews = [elem for elem in experiment_results[1][num_rounds-1]]
    # print("First only = ", (np.mean(end_rews), np.std(end_rews)))
    final_result['first'] = np.mean(end_rews)

    end_rews = [elem for elem in experiment_results[2][num_rounds - 1]]
    # print("Second only = ", (np.mean(end_rews), np.std(end_rews)))
    final_result['second'] = np.mean(end_rews)

    end_rews = [elem for elem in experiment_results[3][num_rounds - 1]]
    # print("Both = ", (np.mean(end_rews), np.std(end_rews)))
    final_result['both'] = np.mean(end_rews)
    return final_result

def run_ablation(start_state, robot_rewards, human_rewards, to_plot):
    start_seed = np.random.randint(0, 10000)
    number_of_seeds = 1
    true_human_order = 2
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
                       num_rounds=num_rounds, h_scalar=h_scalar, r_scalar=r_scalar, to_plot=to_plot)
        vi_type_to_result[vi_type] = final_result

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
    print("Experiment ", random_iter)
    # print("start_state =", start_state)
    # print("robot_rewards = ", robot_rewards)
    # print("human_rewards = ", human_rewards)
    vi_type_to_result = run_ablation(start_state, robot_rewards, human_rewards, to_plot)
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

    num_iters = 5
    to_plot = True

    with Pool(processes=10) as pool:
        trial_results = pool.starmap(run_random_start, [(i, to_plot) for i in range(num_iters)])

    for vi_type_to_result in trial_results:
        diff_mmvi_minus_std_both = vi_type_to_result['mmvi']['both'] - vi_type_to_result['stdvi']['both']
        if diff_mmvi_minus_std_both > 0:
            diff_mmvi_minus_std_both = 1.0
        elif diff_mmvi_minus_std_both == 0:
            diff_mmvi_minus_std_both = 0.0
        else:
            diff_mmvi_minus_std_both = -1.0
        diffs_both.append(diff_mmvi_minus_std_both)

        mmvi = vi_type_to_result['mmvi']['both']
        std = vi_type_to_result['stdvi']['both']

        minval = min(mmvi, std)
        if std < 0 or mmvi < 0:
            mmvi = mmvi - minval + 1
            std = std - minval + 1
        # assert std > 0 and mmvi > 0
        if std == 0:
            std = 0.01
        change_percent = ((float(mmvi)-std)/abs(std))*100
        percents_both.append(change_percent)

        # diff_mmvi_minus_std_first = vi_type_to_result['mmvi']['first'] - vi_type_to_result['stdvi']['first']
        # if diff_mmvi_minus_std_first > 0:
        #     diff_mmvi_minus_std_first = 1.0
        # elif diff_mmvi_minus_std_first == 0:
        #     diff_mmvi_minus_std_first = 0.0
        # else:
        #     diff_mmvi_minus_std_first = -1.0
        # diffs_first.append(diff_mmvi_minus_std_first)
        #
        # diff_mmvi_minus_std_second = vi_type_to_result['mmvi']['second'] - vi_type_to_result['stdvi']['second']
        # if diff_mmvi_minus_std_second > 0:
        #     diff_mmvi_minus_std_second = 1.0
        # elif diff_mmvi_minus_std_second == 0:
        #     diff_mmvi_minus_std_second = 0.0
        # else:
        #     diff_mmvi_minus_std_second = -1.0
        # diffs_second.append(diff_mmvi_minus_std_second)




    # for random_iter in range(2):
    #     start_state = randomList(4, 10)
    #
    #     corpus = (np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1),
    #               np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1))
    #     robot_rewards = corpus
    #     human_rewards = list(corpus)
    #     random.shuffle(human_rewards)
    #     human_rewards = tuple(human_rewards)
    #
    #     print("\n\n")
    #     print("Experiment ", random_iter)
    #     print("start_state =", start_state)
    #     print("robot_rewards = ", robot_rewards)
    #     print("human_rewards = ", human_rewards)
    #     vi_type_to_result = run_ablation(start_state, robot_rewards, human_rewards)
    #
    #     diff_mmvi_minus_std_both = vi_type_to_result['mmvi']['both'] - vi_type_to_result['stdvi']['both']
    #     diffs_both.append(diff_mmvi_minus_std_both)
    #
    #     diff_mmvi_minus_std_first = vi_type_to_result['mmvi']['first'] - vi_type_to_result['stdvi']['first']
    #     diffs_first.append(diff_mmvi_minus_std_first)
    #
    #     diff_mmvi_minus_std_second = vi_type_to_result['mmvi']['second'] - vi_type_to_result['stdvi']['second']
    #     diffs_second.append(diff_mmvi_minus_std_second)
    print("diffs_both", diffs_both)
    print(f"diffs_both: mean = {np.mean(diffs_both)}, std = {np.std(diffs_both)}")
    # print(f"diffs_first: mean = {np.mean(diffs_first)}, std = {np.std(diffs_first)}")
    # print(f"diffs_second: mean = {np.mean(diffs_second)}, std = {np.std(diffs_second)}")

    data = [diffs_both]
    # data = [diffs_both, diffs_first, diffs_second]
    plot_diffs_boxplot(data, num_iters,  f'randomized_aggregate/h1_bool_improvements_nseeds-{num_iters}.png')

    print("percents_both", percents_both)
    print(f"percents_both: mean = {np.mean(percents_both)}, std = {np.std(percents_both)}")
    data = [percents_both]
    # data = [diffs_both, diffs_first, diffs_second]
    plot_diffs_boxplot(data, num_iters, f'randomized_aggregate/h1_percent_change_nseeds-{num_iters}.png')

    test_plot(diffs_both, percents_both, f'randomized_aggregate/h1_result_nseeds-{num_iters}')

    # open a file, where you ant to store the data
    with open(f'randomized_aggregate/h1_booleans_nseeds-{num_iters}.pkl', 'wb') as file:
        pickle.dump(diffs_both, file)

    with open(f'randomized_aggregate/h1_percents_nseeds-{num_iters}.pkl', 'wb') as file:
        pickle.dump(percents_both, file)

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

        corpus = (np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1),
                  np.round(np.random.uniform(-1, 1), 1), np.round(np.random.uniform(-1, 1), 1))

        robot_rewards = corpus
        human_rewards = list(corpus)
        random.shuffle(human_rewards)
        human_rewards = tuple(human_rewards)
        players_to_reward = [robot_rewards, human_rewards]

        # print(f"initial_state = {initial_state}, players_to_reward={players_to_reward}")

        optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances = compare_optimal_to_greedy(
            players_to_reward, initial_state)

        if i % 100==0:
            enablePrint()
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
            blockPrint()

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
    plt.show()

if __name__ == "__main__":
    sample_configurations()
    # main()









