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
# from robot_model_fixed_lstm import Robot
# from robot_model_fcn import Robot
# from robot_model_birl_prob_plan_out import Robot
# from robot_model_lstm_rew import Robot
# from human_model import Greedy_Human, Collaborative_Human, Suboptimal_Collaborative_Human
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
import random


# from robot_1_birl_bsp_ig import Robot
# from robot_2_birl_bsp import Robot
# from robot_3_birl_maxplan import Robot
# from robot_4_birl_maxplan_ig import Robot
# from robot_5_pedbirl_pragplan import Robot
# from robot_6_pedbirl_taskbsp import Robot
# from robot_7_taskbirl_pragplan import Robot
# from robot_8_birlq_bsp_ig import Robot
# from robot_9_birlq_bsp import Robot
# from robot_10_maxent_maxplan import Robot




get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]



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
    print("experiment_number", experiment_number)
    if foldername in ['exp1_noiseless_human', 'exp2_boltz_birl']:
        experiment_name = f'{foldername}/exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}'
    else:
        experiment_name = f'{foldername}/exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}_thresh-0.9'
    path = f"{experiment_name}"
    exp_path = f"{experiment_name}/exps/"
    # Check whether the specified path exists or not

    with open(path + '/exps/' + 'experiment_num_to_results.pkl', 'rb') as fp:
        experiment_num_to_results = pickle.load(fp)


    return experiment_num_to_results

def get_data_with_beta(foldername, global_seed, experiment_number, task_type, exploration_type, replan_type, random_human, num_exps, update_thresh, beta):
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
        experiment_name = f'{foldername}/exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}_thresh-{update_thresh}_beta-{beta}'
    path = f"{experiment_name}"
    exp_path = f"{experiment_name}/exps/"
    # Check whether the specified path exists or not

    with open(path + '/exps/' + 'experiment_num_to_results.pkl', 'rb') as fp:
        experiment_num_to_results = pickle.load(fp)


    return experiment_num_to_results



def plot_round_results_same_plan_diff_learning_boltz_circle_2(human_type = 'noiseless'):
    global_seed = 0
    foldername = 'results'



    num_exps = 100

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=3, sharex=True,
                                                             sharey=True)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'
    replan_explore_to_color = {('coirl', replan_type, 'wo_expl'): '#228B22',
                               ('coirl', replan_type, 'w_expl'): '#50C878',
                               ('maxent', replan_type, 'wo_expl'): '#FF0000',
                               ('cirl', replan_type, 'wo_expl'): '#000080',
                               # ('cirl', replan_type, 'w_expl'): '#0096FF'
                               }

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6

    human_type_idx = 1
    if human_type == 'noiseless':
        human_type_idx = 2
    context = f'1_{human_type_idx}_3objs1'
    zero_reward = None
    zero_std = None

    # get a list of robot types from the 10 above
    # robot_types = [robot_1_birl_bsp_ig, robot_2_birl_bsp, robot_3_birl_maxplan,
    #                robot_4_birl_maxplan_ig, robot_5_pedbirl_pragplan, robot_6_pedbirl_taskbsp,
    #                robot_7_taskbirl_pragplan, robot_8_birlq_bsp_ig, robot_9_birlq_bsp, robot_10_maxent_maxplan]

    # get a list of robot types from the 10 above in string form
    robot_types = ['robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   # 'robot_3_birl_maxplan',
                   #    'robot_4_birl_maxplan_ig',
                   'robot_5_pedbirl_pragplan',
                   # 'robot_6_pedbirl_taskbsp',
                   #   'robot_7_taskbirl_pragplan',
                   # 'robot_8_birlq_bsp_ig',
                   # 'robot_9_birlq_bsp',
                   'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#00008B',
                   'robot_2_birl_bsp': '#228B22',
                   # 'robot_3_birl_maxplan',
                   #    'robot_4_birl_maxplan_ig',
                   'robot_5_pedbirl_pragplan': '#FF0000',
                   # 'robot_6_pedbirl_taskbsp': '#000080',
                   'robot_7_taskbirl_pragplan': '#0096FF',
                   # 'robot_8_birlq_bsp_ig',
                   # 'robot_9_birlq_bsp',
                   'robot_10_maxent_maxplan': '#FFA500',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           # 'robot_3_birl_maxplan',
                           #    'robot_4_birl_maxplan_ig',
                           'robot_5_pedbirl_pragplan': "s",
                           # 'robot_6_pedbirl_taskbsp': '#000080',
                           'robot_7_taskbirl_pragplan': '+',
                           # 'robot_8_birlq_bsp_ig',
                           # 'robot_9_birlq_bsp',
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                            'robot_2_birl_bsp': '--',
                            # 'robot_3_birl_maxplan',
                            #    'robot_4_birl_maxplan_ig',
                            'robot_5_pedbirl_pragplan': ':',
                            # 'robot_6_pedbirl_taskbsp': '#000080',
                            'robot_7_taskbirl_pragplan': ':',
                            # 'robot_8_birlq_bsp_ig',
                            # 'robot_9_birlq_bsp',
                            'robot_10_maxent_maxplan': '-.',
                            }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'Ours',
                          'robot_2_birl_bsp': 'Ours wo IG',
                          # 'robot_3_birl_maxplan',
                          #    'robot_4_birl_maxplan_ig',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          # 'robot_6_pedbirl_taskbsp': '#000080',
                          # 'robot_7_taskbirl_pragplan': ':',
                          # 'robot_8_birlq_bsp_ig',
                          # 'robot_9_birlq_bsp',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }


    for task_type in ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']:
    # for task_type in ['cirl_w_hard_rc']:
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for robot_type in robot_types:
                exploration_type = 'wo_expl'

                experiment_number = f'{context}_{robot_type}_{human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))


                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                # if zero_reward is None:
                #     zero_reward = percents_means[0]
                #     zero_std = percents_stds[0]
                #
                # elif zero_reward is not None:
                #     percents_means[0] = zero_reward
                #     percents_stds[0] = zero_std

                print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                print()


                # ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                #         label=f'{algo}, {exploration_type}')
                # ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                #                 percents_means + percents_stds,
                #                 alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
                ax.errorbar(range(1, len(percents_means)+1), percents_means, yerr=percents_stds, linewidth=2,
                            color=color_to_plot_with,
                            fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=7, markeredgecolor=color_to_plot_with, alpha=1.0,
                            markeredgewidth=2,linestyle=robot_type_to_line[robot_type], elinewidth=2,
                            markerfacecolor='white', label=f'{robot_type_to_name[robot_type]}')
            # if ax_counter == 1:
            #     ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No RC',
                                  'cirl_w_easy_rc': 'Intuitive RC',
                                  'cirl_w_hard_rc': 'Permuted RC'}

            ax.set_title(f"{task_type_to_title[task_type]}")
            ax.set_xticks(range(1, 4))
            ax.set_xticklabels(['1', '2', '3'])
            if ax == ax3:
                ax.legend(loc='lower right')

            ax.set_xlabel("Round Number")
            if ax == ax1:
                ax.set_ylabel("Percent of Optimal Reward")
            # ax.set_ylim([0, 1.6])

    plt.savefig(f"exp3_{human_type}.png")

    plt.close()

def plot_cirl_w_hard_rc(human_types):
    global_seed = 0
    foldername = 'results'

    human_type_to_name = {'boltz_prag_h_b1_actual': 'Pedagogic Human, \nBeta=1',
                          'boltz_b1_pmf': 'Expected Reward Human, \nBeta=1',
                          'boltz_b1': 'Optimistic Reward Human, \nBeta=1',
                          'boltz_binf_pmf': 'Expected Reward Human, \nRational',
                          'boltz_prag_h_b1': 'Pedagogic Human, \nRational',
                          'noiseless': 'Optimistic Reward Human, \nRational',
                          }

    num_exps = 100

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=3, sharex=True,
                                                             sharey=True)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6




    # get a list of robot types from the 10 above in string form
    robot_types = [
        # 'robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   'robot_5_pedbirl_pragplan',
                   'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#7b0f23',
                   'robot_2_birl_bsp': '#2e66a1',
                   'robot_5_pedbirl_pragplan': '#6e7e40',
                   'robot_10_maxent_maxplan': '#e2725b',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           'robot_5_pedbirl_pragplan': "s",
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                            'robot_2_birl_bsp': '-',
                            'robot_5_pedbirl_pragplan': ':',
                            'robot_10_maxent_maxplan': '-.',
                            }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'Ours',
                          'robot_2_birl_bsp': 'Ours',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }

    task_type = 'cirl_w_hard_rc'
    for human_type in human_types:
        human_type_idx = 1
        if human_type == 'noiseless':
            human_type_idx = 2
        context = f'1_{human_type_idx}_3objs1'

        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for robot_type in robot_types:
                # print("robot_type", robot_type)
                exploration_type = 'wo_expl'

                experiment_number = f'{context}_{robot_type}_{human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                seed_to_percent_optimal_rew = {}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    seed = exp_config['seed']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    seed_to_percent_optimal_rew[seed] = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))
                        seed_to_percent_optimal_rew[seed].append(max(0,final_reward / optimal_rew))



                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                # if zero_reward is None:
                #     zero_reward = percents_means[0]
                #     zero_std = percents_stds[0]
                #
                # elif zero_reward is not None:
                #     percents_means[0] = zero_reward
                #     percents_stds[0] = zero_std

                print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                # print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                # print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                # print()
                if robot_type == 'robot_5_pedbirl_pragplan' and human_type == 'boltz_prag_h_b1_actual':
                    print("seed_to_optimal_rew", seed_to_percent_optimal_rew)


                # ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                #         label=f'{algo}, {exploration_type}')
                # ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                #                 percents_means + percents_stds,
                #                 alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
                ax.errorbar(range(1, len(percents_means)+1), percents_means, yerr=percents_stds, linewidth=1.5,
                            color=color_to_plot_with,
                            fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=3, markeredgecolor=color_to_plot_with, alpha=1.0,
                            markeredgewidth=1,linestyle=robot_type_to_line[robot_type], elinewidth=1,
                            markerfacecolor=color_to_plot_with, label=f'{robot_type_to_name[robot_type]}')
            # if ax_counter == 1:
            #     ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No RC',
                                  'cirl_w_easy_rc': 'Intuitive RC',
                                  'cirl_w_hard_rc': 'Permuted RC'}

            ax.set_title(f"{human_type_to_name[human_type]}")
            ax.set_xticks(range(1, 4))
            ax.set_xticklabels(['1', '2', '3'])
            if ax == ax1:
                ax.legend(loc='lower right')

            ax.set_xlabel("Round Number")
            if ax == ax1:
                ax.set_ylabel("Percent of Optimal Reward")
            # ax.set_ylim([0, 1.6])

    plt.savefig(f"exp6_{human_types}.png")

    plt.close()

def plot_cirl_w_hard_rc_2(human_types):
    global_seed = 0
    foldername = 'results'

    human_type_to_name = {'boltz_prag_h_b1_actual': 'Pedagogic Human, \nBeta=1',
                          'boltz_b1_pmf': 'Expected Reward Human, \nBeta=1',
                          'boltz_b1': 'Optimistic Reward Human, \nBeta=1',
                          'boltz_binf_pmf': 'Expected Reward Human, \nRational',
                          'boltz_prag_h_b1': 'Pedagogic Human, \nRational',
                          'noiseless': 'Optimistic Reward Human, \nRational',
                          }

    num_exps = 5

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=3, sharex=True,
                                                             sharey=True)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6




    # get a list of robot types from the 10 above in string form
    robot_types = [
        'robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   'robot_5_pedbirl_pragplan',
                   'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#7b0f23',
                   'robot_2_birl_bsp': '#2e66a1',
                   'robot_5_pedbirl_pragplan': '#6e7e40',
                   'robot_10_maxent_maxplan': '#e2725b',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           'robot_5_pedbirl_pragplan': "s",
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                            'robot_2_birl_bsp': '-',
                            'robot_5_pedbirl_pragplan': ':',
                            'robot_10_maxent_maxplan': '-.',
                            }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'Ours',
                          'robot_2_birl_bsp': 'Ours wo IG',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }

    task_type = 'cirl_w_hard_rc'
    for human_type in human_types:
        human_type_idx = 1
        if human_type == 'noiseless':
            human_type_idx = 2
        context = f'1_{human_type_idx}_3objs1'


        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for robot_type in robot_types:
                print("robot_type", robot_type)
                print("human_type", human_type)
                exploration_type = 'wo_expl'
                if human_type == 'boltz_prag_h_b1':
                    if robot_type == 'robot_5_pedbirl_pragplan':
                        context = f'2_1_3objs1'
                        temp_human_type = 'boltz_prag_h_b1_actual_hv_2'
                    else:
                        context = f'2_1_3objs1'
                        temp_human_type = 'boltz_prag_h_b1_actual_hv'
                else:
                    temp_human_type = human_type
                experiment_number = f'{context}_{robot_type}_{temp_human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                seed_to_percent_optimal_rew = {}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    seed = exp_config['seed']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    seed_to_percent_optimal_rew[seed] = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))
                        seed_to_percent_optimal_rew[seed].append(max(0,final_reward / optimal_rew))



                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                # if zero_reward is None:
                #     zero_reward = percents_means[0]
                #     zero_std = percents_stds[0]
                #
                # elif zero_reward is not None:
                #     percents_means[0] = zero_reward
                #     percents_stds[0] = zero_std

                # print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                # print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                # print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                # print()
                # if robot_type == 'robot_5_pedbirl_pragplan' and human_type == 'boltz_prag_h_b1_actual':
                #     print("seed_to_optimal_rew", seed_to_percent_optimal_rew)


                # ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                #         label=f'{algo}, {exploration_type}')
                # ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                #                 percents_means + percents_stds,
                #                 alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
                ax.errorbar(range(1, len(percents_means)+1), percents_means, yerr=percents_stds, linewidth=1.5,
                            color=color_to_plot_with,
                            fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=3, markeredgecolor=color_to_plot_with, alpha=1.0,
                            markeredgewidth=1,linestyle=robot_type_to_line[robot_type], elinewidth=1,
                            markerfacecolor=color_to_plot_with, label=f'{robot_type_to_name[robot_type]}')
            # if ax_counter == 1:
            #     ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No RC',
                                  'cirl_w_easy_rc': 'Intuitive RC',
                                  'cirl_w_hard_rc': 'Permuted RC'}

            ax.set_title(f"{human_type_to_name[human_type]}")
            ax.set_xticks(range(1, 4))
            ax.set_xticklabels(['1', '2', '3'])
            if ax == ax1:
                ax.legend(loc='lower right')

            ax.set_xlabel("Round Number")
            if ax == ax1:
                ax.set_ylabel("Percent of Optimal Reward")
            # ax.set_ylim([0, 1.6])

    plt.savefig(f"exp6_{human_types}.png")

    plt.close()

def plot_cirl_w_hard_rc_3(human_types):
    global_seed = 0
    foldername = 'results'

    human_type_to_name = {'boltz_prag_h_b1_actual_hv_1': 'Pedagogic Human, \nBeta=1',
                          'boltz_b1_pmf': 'Expected Reward Human, \nBeta=1',
                          'boltz_b1': 'Optimistic Reward Human, \nBeta=1',
                          'boltz_binf_pmf': 'Expected Reward Human, \nRational',
                          'boltz_prag_h_binf_actual_hv_1': 'Pedagogic Human, \nRational',
                          'noiseless': 'Optimistic Reward Human, \nRational',
                          }

    num_exps = 50

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(15,4), nrows=1, ncols=3, sharex=True,
                                                             sharey=False)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    plt.rcParams.update({'font.size': 13})
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6




    # get a list of robot types from the 10 above in string form
    robot_types = [
                    'robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   'robot_5_pedbirl_pragplan',
                   'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#7b0f23',
                   'robot_2_birl_bsp': '#2e66a1',
                   'robot_5_pedbirl_pragplan': '#6e7e40',
                   'robot_10_maxent_maxplan': '#e2725b',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           'robot_5_pedbirl_pragplan': "s",
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                            'robot_2_birl_bsp': '--',
                            'robot_5_pedbirl_pragplan': ':',
                            'robot_10_maxent_maxplan': '-.',
                            }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'BaISL',
                          'robot_2_birl_bsp': 'BaL',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }

    task_type = 'cirl_w_hard_rc'
    exploration_type = 'wo_expl'
    for human_type in human_types:


        print("\n\nSTARTING human type", human_type)

        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = axes_list[ax_counter]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for robot_type in robot_types:
                print("\n\nSTARTING robot type", robot_type)

                # experiment_number = f'domain2_4objs0_{robot_type}_{human_type}_human'
                experiment_number = f'domain2_approp_diff_specified_3objs5_{robot_type}_{human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                seed_to_percent_optimal_rew = {}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    seed = exp_config['seed']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    seed_to_percent_optimal_rew[seed] = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))
                        seed_to_percent_optimal_rew[seed].append(max(0,final_reward / optimal_rew))



                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                print("percents_means", percents_means)
                print("percents_stds", percents_stds)

                # if zero_reward is None:
                #     zero_reward = percents_means[0]
                #     zero_std = percents_stds[0]
                #
                # elif zero_reward is not None:
                #     percents_means[0] = zero_reward
                #     percents_stds[0] = zero_std

                # print(f"{replan_type}, {exploration_type}, {task_type}, {random_human}")
                # print("mean_aggregate_data_over_rounds", mean_aggregate_data_over_rounds)
                # print("std_aggregate_data_over_rounds", std_aggregate_data_over_rounds)
                # print()
                # if robot_type == 'robot_5_pedbirl_pragplan' and human_type == 'boltz_prag_h_b1_actual':
                #     print("seed_to_optimal_rew", seed_to_percent_optimal_rew)


                # ax.plot(range(len(percents_means)), percents_means, color=color_to_plot_with,
                #         label=f'{algo}, {exploration_type}')
                # ax.fill_between(range(len(percents_means)), percents_means - percents_stds,
                #                 percents_means + percents_stds,
                #                 alpha=0.5, edgecolor=color_to_plot_with, facecolor=color_to_plot_with)
                ax.errorbar(range(1, len(percents_means)+1), percents_means, yerr=percents_stds, linewidth=1.5,
                            color=color_to_plot_with,
                            fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=3, markeredgecolor=color_to_plot_with, alpha=1.0,
                            markeredgewidth=1,linestyle=robot_type_to_line[robot_type], elinewidth=1,
                            markerfacecolor=color_to_plot_with, label=f'{robot_type_to_name[robot_type]}')
            # if ax_counter == 1:
            #     ax.legend(loc='lower right')

            task_type_to_title = {'cirl': 'No RC',
                                  'cirl_w_easy_rc': 'Intuitive RC',
                                  'cirl_w_hard_rc': 'Permuted RC'}

            ax.set_title(f"{human_type_to_name[human_type]}")
            ax.set_xticks(range(1, 4))
            ax.set_xticklabels(['1', '2', '3'], fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            if ax == ax1:
                ax.legend(loc='lower right')

            ax.set_xlabel("Round Number", fontsize=12, labelpad=0)
            if ax == ax1:
                ax.set_ylabel("Percent of Optimal Reward", fontsize=12)
            # ax.set_ylim([0, 1.6])

    plt.savefig(f"exp8_{human_types}.png")

    plt.close()

def plot_ablate_beta():
    global_seed = 0
    foldername = 'results'

    human_type_to_name = {'boltz_prag_h_b1_actual_hv_1': 'Pedagogic Human, \nBeta=1',
                          'boltz_b1_pmf': 'Expected Reward Human, \nBeta=1',
                          'boltz_b1': 'Optimistic Reward Human, \nBeta=1',
                          'boltz_binf_pmf': 'Expected Reward Human, \nRational',
                          'boltz_prag_h_binf_actual_hv_1': 'Pedagogic Human, \nRational',
                          'noiseless': 'Optimistic Reward Human, \nRational',
                          }

    num_exps = 50

    # fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=3, sharex=True,
    #                                                          sharey=True)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    fig, ((ax1)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=1, sharex=True,sharey=True)
    # axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6




    # get a list of robot types from the 10 above in string form
    robot_types = [
                    'robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   'robot_5_pedbirl_pragplan',
                   # 'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#7b0f23',
                   'robot_2_birl_bsp': '#2e66a1',
                   'robot_5_pedbirl_pragplan': '#6e7e40',
                   # 'robot_10_maxent_maxplan': '#e2725b',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           'robot_5_pedbirl_pragplan': "s",
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                          'robot_2_birl_bsp': '--',
                          'robot_5_pedbirl_pragplan': ':',
                          'robot_10_maxent_maxplan': '-.',
                          }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'BaISL',
                          'robot_2_birl_bsp': 'BaL',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }

    task_type = 'cirl_w_hard_rc'
    human_type = 'noiseless'
    update_thresh =0.9
    print("\n\nSTARTING human type", human_type)
    for robot_type in robot_types:

        beta_to_rewards = {}
        beta_to_rewards_std = {}
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = ax1
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for beta in [1, 2.5, 5, 15]:
                print("\n\nSTARTING robot type", robot_type)
                temp_robot_type = robot_type

                exploration_type = 'wo_expl'
                # if human_type == 'boltz_prag_h_b1':
                #     if robot_type == 'robot_5_pedbirl_pragplan':
                #         context = f'2_1_3objs1'
                #         temp_human_type = 'boltz_prag_h_b1_actual_hv_2'
                #     else:
                #         context = f'2_1_3objs1'
                #         temp_human_type = 'boltz_prag_h_b1_actual_hv'
                # else:
                #     temp_human_type = human_type
                print()
                print("robot_type", temp_robot_type)
                print("human_type", human_type)
                context = 'ablateb_3objs1'

                experiment_number = f'{context}_{temp_robot_type}_{human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data_with_beta(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps, update_thresh, beta)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                seed_to_percent_optimal_rew = {}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    seed = exp_config['seed']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    seed_to_percent_optimal_rew[seed] = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))
                        seed_to_percent_optimal_rew[seed].append(max(0,final_reward / optimal_rew))



                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                print("percents_means", percents_means)
                print("percents_stds", percents_stds)
                beta_to_rewards[beta] = percents_means[-1]
                beta_to_rewards_std[beta] = percents_stds[-1]

            beta_vals = [1, 2.5, 5, 15]
            performance_means = np.array([beta_to_rewards[beta] for beta in beta_vals])
            performance_stds = np.array([beta_to_rewards_std[beta] for beta in beta_vals])
            ax.errorbar(beta_vals, performance_means, yerr=performance_stds, linewidth=1.5,
                        color=color_to_plot_with,
                        fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=3, markeredgecolor=color_to_plot_with, alpha=1.0,
                        markeredgewidth=1,linestyle=robot_type_to_line[robot_type], elinewidth=1,
                        markerfacecolor=color_to_plot_with, label=f'{robot_type_to_name[robot_type]}')


        ax.set_title(f"Performance vs. Human Rationality Coefficient")
        ax.set_xticks(beta_vals)
        ax.set_xticklabels(beta_vals)
        if ax == ax1:
            ax.legend(loc='lower right')

        ax.set_xlabel("Simulated Human Rationality Coefficient")
        if ax == ax1:
            ax.set_ylabel("Percent of Optimal Reward")
        ax.set_ylim([0.75, 1.0])

    plt.savefig(f"exp7_ablateb.png")

    plt.close()

def plot_ablate_lambda():
    global_seed = 0
    foldername = 'results'

    human_type_to_name = {'boltz_prag_h_b1_actual_hv_1': 'Pedagogic Human, \nBeta=1',
                          'boltz_b1_pmf': 'Expected Reward Human, \nBeta=1',
                          'boltz_b1': 'Optimistic Reward Human, \nBeta=1',
                          'boltz_binf_pmf': 'Expected Reward Human, \nRational',
                          'boltz_prag_h_binf_actual_hv_1': 'Pedagogic Human, \nRational',
                          'noiseless': 'Optimistic Reward Human, \nRational',
                          }

    num_exps = 50

    # fig, ((ax1), (ax2), (ax3)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=3, sharex=True,
    #                                                          sharey=True)
    # fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False)
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    fig, ((ax1)) = plt.subplots(figsize=(10, 4), nrows=1, ncols=1, sharex=True,sharey=True)
    # axes_list = [ax1, ax2, ax3]
    replan_type = 'w_replan'

    task_human_to_ax = {}
    ax_counter = 0
    num_rounds = 6




    # get a list of robot types from the 10 above in string form
    robot_types = [
                    'robot_1_birl_bsp_ig',
                   'robot_2_birl_bsp',
                   'robot_5_pedbirl_pragplan',
                   # 'robot_10_maxent_maxplan'
                   ]
    robot_type_to_color = {}
    for robot_type in robot_types:
        robot_type_to_color[robot_type] = get_colors(1)[0]

    robot_type_to_color = {'robot_1_birl_bsp_ig': '#7b0f23',
                   'robot_2_birl_bsp': '#2e66a1',
                   'robot_5_pedbirl_pragplan': '#6e7e40',
                   # 'robot_10_maxent_maxplan': '#e2725b',
                           }
    robot_type_to_marker = {'robot_1_birl_bsp_ig': 'o',
                           'robot_2_birl_bsp': "^",
                           'robot_5_pedbirl_pragplan': "s",
                           'robot_10_maxent_maxplan': "*",
                           }

    robot_type_to_line = {'robot_1_birl_bsp_ig': '-',
                          'robot_2_birl_bsp': '--',
                          'robot_5_pedbirl_pragplan': ':',
                          'robot_10_maxent_maxplan': '-.',
                          }
    robot_type_to_name = {'robot_1_birl_bsp_ig': 'BaISL',
                          'robot_2_birl_bsp': 'BaL',
                          'robot_5_pedbirl_pragplan': 'Prag-Ped',
                          'robot_10_maxent_maxplan': 'MaxEntIRL',
                          }

    task_type = 'cirl_w_hard_rc'
    human_type = 'noiseless'
    update_thresh =0.9
    beta = 1
    print("\n\nSTARTING human type", human_type)
    for robot_type in robot_types:

        beta_to_rewards = {}
        beta_to_rewards_std = {}
        for random_human in [False]:
            r_h_str = 'sto'
            if random_human is False:
                r_h_str = 'opt'

            ax = ax1
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            task_human_to_ax[(task_type, random_human)] = ax
            ax_counter += 1
            for update_thresh in [0.6, 0.7, 0.8, 0.9, 1.0]:
                print("\n\nSTARTING robot type", robot_type)
                temp_robot_type = robot_type

                exploration_type = 'wo_expl'
                # if human_type == 'boltz_prag_h_b1':
                #     if robot_type == 'robot_5_pedbirl_pragplan':
                #         context = f'2_1_3objs1'
                #         temp_human_type = 'boltz_prag_h_b1_actual_hv_2'
                #     else:
                #         context = f'2_1_3objs1'
                #         temp_human_type = 'boltz_prag_h_b1_actual_hv'
                # else:
                #     temp_human_type = human_type
                print()
                print("robot_type", temp_robot_type)
                print("human_type", human_type)
                context = 'ablatelambda_3objs1'

                experiment_number = f'{context}_{temp_robot_type}_{human_type}_human'
                color_to_plot_with = robot_type_to_color[robot_type]

                data = get_data_with_beta(foldername, global_seed, experiment_number, task_type, exploration_type,
                                replan_type, random_human, num_exps, update_thresh, beta)
                aggregate_data_over_rounds = {round_no: [] for round_no in range(-1, 3)}
                seed_to_percent_optimal_rew = {}
                for exp_num in range(num_exps):
                    exp_results = data[exp_num]['results']
                    exp_config = data[exp_num]['config']
                    seed = exp_config['seed']
                    optimal_rew = exp_results['optimal_rew']
                    game_results = exp_results['cvi_game_results']
                    # final_reward_per_round = []
                    seed_to_percent_optimal_rew[seed] = []
                    for round_no in range(-1, 3):
                        final_reward = game_results[round_no]['final_reward']
                        aggregate_data_over_rounds[round_no].append(max(0,final_reward / optimal_rew))
                        seed_to_percent_optimal_rew[seed].append(max(0,final_reward / optimal_rew))



                mean_aggregate_data_over_rounds = {round_no: np.mean(aggregate_data_over_rounds[round_no]) for
                                                   round_no in range(-1, 3)}
                std_aggregate_data_over_rounds = {round_no: np.std(aggregate_data_over_rounds[round_no]) for
                                                  round_no in range(-1, 3)}

                percents_means = np.array(
                    [np.mean(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])
                percents_stds = np.array(
                    [sem(aggregate_data_over_rounds[round_no]) for round_no in range(0, 3)])

                print("percents_means", percents_means)
                print("percents_stds", percents_stds)
                beta_to_rewards[update_thresh] = percents_means[-1]
                beta_to_rewards_std[update_thresh] = percents_stds[-1]

            update_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
            performance_means = np.array([beta_to_rewards[beta] for beta in update_vals])
            performance_stds = np.array([beta_to_rewards_std[beta] for beta in update_vals])
            ax.errorbar(update_vals, performance_means, yerr=performance_stds, linewidth=1.5,
                        color=color_to_plot_with,
                        fillstyle='full', marker=robot_type_to_marker[robot_type], markersize=3, markeredgecolor=color_to_plot_with, alpha=1.0,
                        markeredgewidth=1,linestyle=robot_type_to_line[robot_type], elinewidth=1,
                        markerfacecolor=color_to_plot_with, label=f'{robot_type_to_name[robot_type]}')


        ax.set_title(f"Performance vs. Update Threshold Lambda")
        ax.set_xticks(update_vals)
        ax.set_xticklabels(update_vals)
        if ax == ax1:
            ax.legend(loc='lower right')

        ax.set_xlabel("Update Threshold Lambda")
        if ax == ax1:
            ax.set_ylabel("Percent of Optimal Reward")
        ax.set_ylim([0.75, 1.0])

    plt.savefig(f"exp7_ablatelambda.png")

    plt.close()

if __name__ == "__main__":
    # print("OPT HUMAN")
    # plot_round_results_same_plan_diff_learning_opt_circle()
    # human_type = 'boltz_b1_pmf'
    # print("\n\nBOLTZ HUMAN")
    # plot_round_results_same_plan_diff_learning_boltz_circle_2('boltz_b1_pmf')
    # plot_round_results_same_plan_diff_learning_boltz_circle_2('boltz_prag_h_b1')
    # plot_cirl_w_hard_rc_2(['boltz_b1', 'boltz_b1_pmf', 'boltz_prag_h_b1_actual'])
    # plot_cirl_w_hard_rc_2(['noiseless', 'boltz_binf_pmf', 'boltz_prag_h_b1'])
    # plot_round_results_same_plan_diff_learning_boltz_circle_2('noiseless')
    # np.random.seed(0)
    plot_cirl_w_hard_rc_3(['boltz_b1', 'boltz_b1_pmf', 'boltz_prag_h_b1_actual_hv_1'])
    plot_cirl_w_hard_rc_3(['noiseless', 'boltz_binf_pmf', 'boltz_prag_h_binf_actual_hv_1'])
    # plot_ablate_beta()
    # plot_ablate_lambda()









