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

from human_agent import Human_Hypothesis
from robot_tom_agent import Robot_Model
from multiprocessing import Pool, freeze_support
import pickle
import json


import sys, os

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
    def __init__(self, robot, human, start_state, log_filename='', to_plot=False):
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

        return self.state, rew, rew_pair, self.is_done(), robot_action, human_action

    def run_full_game(self, round, no_update=False):
        self.reset()
        iteration_count = 0
        robot_history = []
        human_history = []


        while self.is_done() is False:
            _, rew, rew_pair, _, r_action, h_action = self.step(iteration=iteration_count, no_update=no_update)
            print(f"\nTimestep: {iteration_count}")
            print(f"Current state: {self.state}")
            print(f"Robot took action: {COLOR_TO_TEXT[r_action]}")
            print(f"Robot took action: {COLOR_TO_TEXT[h_action]}")
            print(f"Achieved reward: {rew_pair}")
            print(f"Total reward: {self.total_reward}")

            self.rew_history.append(rew_pair)
            self.total_reward += rew

            robot_history.append(r_action)
            human_history.append(h_action)

            iteration_count += 1
        return self.total_reward, robot_history, human_history


def run_k_rounds(list_of_start_states, all_colors_list, seed, num_rounds, vi_type, true_human_rho,
                 human_rewards, task_reward, robot_rewards, rho_candidates, exp_path, to_plot):
    # print(f"running seed {seed} for mm_order {mm_order} ")
    np.random.seed(seed)
    # log_filename = "exp_results/" + exp_path + '/' + vi_type + '/weights_imgs/' + mm_order + '/' + 'log.txt'
    robot_agent = Robot_Model(robot_rewards, all_colors_list, task_reward, rho_candidates, vi_type)

    true_human_agent = Human_Hypothesis(human_rewards, all_colors_list, task_reward, true_human_rho)
    # true_human_agent = True_Human_Model(human_rewards, true_human_order, num_particles=num_particles)

    rof_game = RingOfFire(robot_agent, true_human_agent, list_of_start_states[0])
    # rof_game.run_full_game()

    collective_scores = {x: [] for x in range(num_rounds)}

    for round in range(num_rounds):
        print(f"\n\nRound = {round}")
        rof_game.reset()
        total_rew, robot_history, human_history = rof_game.run_full_game(round)
        collective_scores[round].append(total_rew)



    # print("Done Weight Updates")
    return collective_scores




def run_experiment(list_of_start_states, all_colors_list, n_seeds, num_rounds, vi_type, true_human_rho,
                 human_rewards, task_reward, robot_rewards, rho_candidates, exp_path, to_plot):
    # np.random.seed(0)
    num_seeds = n_seeds
    list_of_random_seeds = np.random.randint(0, 100000, num_seeds)

    both_results = {x: [] for x in range(num_rounds)}

    seed_val = list_of_random_seeds[0]


    scores = run_k_rounds(list_of_start_states, all_colors_list, seed_val, num_rounds, vi_type, true_human_rho,
                 human_rewards, task_reward, robot_rewards, rho_candidates, exp_path, to_plot)


    # for result in both_order_scores:
    for round_no in scores:
        both_results[round_no].extend(scores[round_no])




    if to_plot:
        experiment_params = {'start_state': list_of_start_states,
                             'vi_type': vi_type,
                             'n_seeds': n_seeds,

                             'robot_rewards': robot_rewards,
                             'human_rewards': human_rewards,
                             'exp_path': exp_path,
                             'num_rounds': num_rounds,
                             'list_of_random_seeds': list_of_random_seeds}

        savefile = "exp_results/" + exp_path + '/' + vi_type + '/experiment_params.json'
        with open(savefile, 'w') as fp:
            # json.dumps(experiment_params, cls=NumpyEncoder)
            json.dump(experiment_params, fp, cls=NumpyEncoder)
            # print("Saved experiment params")

        savefile = "exp_results/" + exp_path + '/' + vi_type + '/pkl_results' + '/results.pkl'
        with open(savefile, 'wb') as handle:
            pickle.dump(both_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print("Saved pickled experiment results")

        savefile = "exp_results/" + exp_path + '/' + vi_type + '/agg_imgs' + '/scores_over_time.png'
        print("Saved experiment results")

    final_result = {}
    end_rews = [elem for elem in both_results[num_rounds - 1]]
    # print("Both = ", (np.mean(end_rews), np.std(end_rews)))
    final_result['mean'] = np.mean(end_rews)
    return final_result

def run_ablation(all_colors_list, start_state, robot_rewards, human_rewards, to_plot=False):
    start_seed = np.random.randint(0, 10000)
    number_of_seeds = 1
    true_human_rho = 1
    task_reward = [1]*len(all_colors_list)

    rho_candidates = [0]
    list_of_start_states = [start_state]


    num_rounds = 3
    vi_types = ['cvi', 'stdvi']
    # vi_types = ['mmvi']

    if to_plot:
        path = f"random_trials/DEBUG7_start-{start_state}_Hrew-{human_rewards}_Rrew-{robot_rewards}_horder-{true_human_rho}_nparticles-{num_particles}_nrounds-{num_rounds}_nseeds-{number_of_seeds}_startseed-{start_seed}"
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
        final_result = run_experiment(list_of_start_states, all_colors_list, number_of_seeds, num_rounds, vi_type, true_human_rho,
                 human_rewards, task_reward, robot_rewards, rho_candidates, (path if to_plot else None), to_plot=to_plot)
        vi_type_to_result[vi_type] = final_result

    # print("vi_type_to_result", vi_type_to_result)
    return vi_type_to_result


def run_random_start(random_iter, to_plot):
    all_colors_list = [BLUE, GREEN, RED]
    n_colors = len(all_colors_list)
    total_num_objects = 6
    start_state = randomList(n_colors, total_num_objects)

    corpus = tuple([np.round(np.random.uniform(0.01, 2), 1) for _ in range(n_colors)])
    robot_rewards = corpus
    human_rewards = list(corpus)
    random.shuffle(human_rewards)
    human_rewards = tuple(human_rewards)

    # print("\n\n")
    print("Experiment ", random_iter)
    # print("start_state =", start_state)
    # print("robot_rewards = ", robot_rewards)
    # print("human_rewards = ", human_rewards)
    vi_type_to_result = run_ablation(all_colors_list, start_state, robot_rewards, human_rewards, to_plot=to_plot)
    return vi_type_to_result


def main():
    vi_types = ['cvi', 'stdvi']
    diffs_both = []

    num_iters = 1
    to_plot = False
    to_save = False

    with Pool(processes=10) as pool:
        trial_results = pool.starmap(run_random_start, [(i, to_plot) for i in range(num_iters)])

    for vi_type_to_result in trial_results:
        diff_mmvi_minus_std_both = vi_type_to_result['cvi']['mean'] - vi_type_to_result['stdvi']['mean']
        diffs_both.append(diff_mmvi_minus_std_both)

    plt.hist(diffs_both, bins=20)
    plt.show()




if __name__ == "__main__":
    main()









