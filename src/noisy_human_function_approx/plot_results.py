import pdb

import numpy as np
import operator
import copy
import networkx as nx
import random
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from multiprocessing import Pool, freeze_support
# from robot_model_birl_rew import Robot
from robot_model_fixed_lstm import Robot
# from robot_model_fcn import Robot
# from robot_model_birl_prob_plan_out import Robot
# from robot_model_lstm_rew import Robot
from human_model import Greedy_Human, Collaborative_Human, Suboptimal_Collaborative_Human
# import seaborn as sns
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.stats import sem
import pickle
import json
import sys
import os
import subprocess
import glob
from scipy.stats import sem







def get_data(foldername, global_seed, experiment_number, task_type, exploration_type, replan_type, random_human, num_exps):
    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    r_h_str = 'random_human'
    if random_human is False:
        r_h_str = 'deter_human'


    replan_online = True
    if replan_type == 'wo_replan':
        replan_online = False

    use_exploration = True
    if exploration_type == 'wo_expl':
        use_exploration = False

    if foldername in ['exp1_noiseless_human', 'exp2_boltz_birl']:
        experiment_name = f'{foldername}/exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}'
    else:
        experiment_name = f'{foldername}/exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}_thresh-0.9'
    path = f"results/{experiment_name}"
    exp_path = f"results/{experiment_name}/exps/"
    # Check whether the specified path exists or not

    with open(path + '/exps/' + 'experiment_num_to_results.pkl', 'rb') as fp:
        experiment_num_to_results = pickle.load(fp)


    return experiment_num_to_results



def get_data_not_in_folder(global_seed, experiment_number, task_type, exploration_type, replan_type, random_human, num_exps):
    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    r_h_str = 'random_human'
    if random_human is False:
        r_h_str = 'deter_human'


    replan_online = True
    if replan_type == 'wo_replan':
        replan_online = False

    use_exploration = True
    if exploration_type == 'wo_expl':
        use_exploration = False

    experiment_name = f'exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}'
    path = f"results/{experiment_name}"
    exp_path = f"results/{experiment_name}/exps/"
    # Check whether the specified path exists or not

    with open(path + '/exps/' + 'experiment_num_to_results.pkl', 'rb') as fp:
        experiment_num_to_results = pickle.load(fp)


    return experiment_num_to_results


def plot_round_results():
    global_seed = 0
    experiment_number = '1'
    task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=2, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False, True]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(final_reward / optimal_rew)

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')
            ax.set_title(f"{task_type}, human: {r_h_str}")

    plt.savefig("exp1_results_full.png")
    plt.close()

def plot_round_results_birl_opt():
    global_seed = 0
    foldername = 'exp1_noiseless_human'
    experiment_number = '1'
    task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Noiseless Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")
            ax.set_ylabel("Percent of Optimal Reward")
            ax.set_ylim([0, 1.01])

    plt.savefig("exp1_results_full_birl_opt.png")

    plt.close()

def plot_round_results_birl_boltz():
    global_seed = 0
    foldername = 'exp2_boltz_birl'
    experiment_number = '2_birl_boltz_h-only'
    task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')
            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Boltzmann Rational Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")

            ax.set_ylabel("Percent of Optimal Reward")
            ax.set_ylim([0, 1.01])



    plt.savefig("exp1_results_full_birl_boltz.png")
    # plt.xlabel("Round Number")
    # plt.ylabel("Percent of Optimal Reward")
    plt.close()


def plot_round_results_new_expl_birl_opt():
    global_seed = 0
    foldername = 'exp3_new_explore_noiseless_human'
    experiment_number = '3_new_explore_no_noise_human'
    # task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    # replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['w_expl', 'wo_expl']:
                    if exploration_type == 'wo_expl':
                        foldername = 'exp1_noiseless_human'
                        experiment_number = '1'
                    else:
                        foldername = 'exp3_new_explore_noiseless_human'
                        experiment_number = '3_new_explore_no_noise_human'

                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                       round_no in range(num_rounds)}
                    std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                      round_no in range(num_rounds)}

                    print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                    print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                    print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                    print()

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Noiseless Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")
            ax.set_ylabel("Percent of Optimal Reward")
            # ax.set_ylim([0, 1.01])

    plt.savefig("exp3_new_explore_results_full_birl_opt_2.png")

    plt.close()


def plot_round_results_new_expl_birl_boltz():
    global_seed = 0
    foldername = 'exp3_new_explore_boltz_human'
    experiment_number = '3_new_explore_boltz_human'
    task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['w_expl', 'wo_expl']:
                    if exploration_type == 'wo_expl':
                        foldername = 'exp2_boltz_birl'
                        experiment_number = '2_birl_boltz_h-only'
                    else:
                        foldername = 'exp3_new_explore_boltz_human'
                        experiment_number = '3_new_explore_boltz_human'

                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                       round_no in range(num_rounds)}
                    std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                      round_no in range(num_rounds)}

                    print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                    print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                    print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                    print()

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')
            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Boltzmann Rational Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")

            ax.set_ylabel("Percent of Optimal Reward")
            # ax.set_ylim([0, 1.01])



    plt.savefig("exp3_new_explore_results_full_birl_boltz_2.png")
    # plt.xlabel("Round Number")
    # plt.ylabel("Percent of Optimal Reward")
    plt.close()


def plot_round_results_baseline_cirl_birl_boltz():
    global_seed = 0
    foldername = 'exp4_baseline_cirl_boltz_human'
    experiment_number = '4_baseline-cirl_boltz_human'
    task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                       round_no in range(num_rounds)}
                    std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                      round_no in range(num_rounds)}

                    print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                    print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                    print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                    print()

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')
            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Boltzmann Rational Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")

            ax.set_ylabel("Percent of Optimal Reward")
            ax.set_ylim([0, 1.01])



    plt.savefig("exp4_baseline_cirl_full_birl_boltz.png")
    # plt.xlabel("Round Number")
    # plt.ylabel("Percent of Optimal Reward")
    plt.close()

def plot_round_results_baseline_cirl_birl_opt():
    global_seed = 0
    foldername = 'exp4_baseline_cirl_noiseless_human'
    experiment_number = '4_baseline-cirl_no_noise_human'
    # task_type = 'cirl_w_hard_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    # replan_type = 'w_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    fig, ((ax1, ax2, ax3)) = plt.subplots(figsize=(7, 10), nrows=3, ncols=1, sharex=False,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]

    replan_explore_to_color = {('wo_replan', 'wo_expl'): '#228B22',
                               ('wo_replan', 'w_expl'): '#50C878',
                               ('w_replan', 'wo_expl'): '#000080',
                               ('w_replan', 'w_expl'): '#0096FF'}

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    color_to_plot_with = replan_explore_to_color[(replan_type, exploration_type)]
                    data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                    replan_type, random_human, num_exps)
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        if exp_config['random_h_alpha'] > 0.5:
                            continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))

                    mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                       round_no in range(num_rounds)}
                    std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                      round_no in range(num_rounds)}

                    print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                    print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                    print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                    print()

                    percents_means = np.array(
                        [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    percents_stds = np.array(
                        [sem(aggregate_data_over_rounds[round_no]) for round_no in range(num_rounds)])
                    ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                            label=f'{replan_type}, {exploration_type}')
                    ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                                    percents_means + percents_stds,
                                    alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
            if ax_counter == 1:
                ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No Robot Collaborative Utility',
                                  'cirl_w_easy_rc': 'Simple Robot Collaborative Utility',
                                  'cirl_w_hard_rc': 'Permuted Robot Collaborative Utility'}

            ax.set_title(f"{task_type_to_title[task_type]}, Noiseless Human")
            if ax == ax3:
                ax.set_xlabel("Round Number")
            ax.set_ylabel("Percent of Optimal Reward")
            ax.set_ylim([0, 1.01])

    plt.savefig("exp4_new_baseline_cirl_full_birl_opt.png")

    plt.close()

def check_values():
    global_seed = 0
    experiment_number = '1_lstm'
    task_type = 'cirl_w_easy_rc'  # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    replan_type = 'wo_replan'  # ['wo_replan', 'w_replan']
    # random_human = False
    num_exps = 100

    ax_counter = 0
    num_rounds = 6
    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            for replan_type in ['wo_replan', 'w_replan']:
                for exploration_type in ['wo_expl', 'w_expl']:
                    try:
                        data = get_data_not_in_folder(global_seed, experiment_number, task_type, exploration_type,
                                                      replan_type, random_human, num_exps)
                    except:
                        continue
                    aggregate_data_over_rounds = {round_no: [] for round_no in range(num_rounds)}
                    for exp_num in range(num_exps):
                        exp_results = data[exp_num]['results']
                        exp_config = data[exp_num]['config']
                        # if exp_config['random_h_alpha']> 0.5:
                        #     continue

                        # print("exp_results", exp_results.keys())
                        optimal_rew = exp_results['optimal_rew']
                        game_results = exp_results['cvi_game_results']
                        # final_reward_per_round = []
                        for round_no in range(num_rounds):
                            final_reward = game_results[round_no]['final_reward']
                            # final_reward_per_round.append(final_reward/optimal_rew)
                            aggregate_data_over_rounds[round_no].append(max(0, final_reward / optimal_rew))
                    mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                       round_no in range(num_rounds)}
                    std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                      round_no in range(num_rounds)}

                    print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                    print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                    print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                    print()

if __name__ == "__main__":
    print("BOLTZ HUMAN")
    plot_round_results_new_expl_birl_boltz()

    print("\n\nOPT HUMAN")
    # plot_round_results_new_expl_birl_opt()
    # np.random.seed(0)









