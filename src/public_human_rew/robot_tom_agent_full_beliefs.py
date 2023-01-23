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

from hip_mdp_1player import HiMDP
from human_agent import Human_Hypothesis


def get_key_with_max_value_from_dict(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0]

def get_key_value_pair_with_max_value_from_dict(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0], dictionary[max(dictionary.items(), key=operator.itemgetter(1))[0]]


class Robot_Model():
    def __init__(self, individual_reward, all_colors_list, task_reward, rho_candidates, vi_type, log_filename='', true_human_reward=None):
        # Set individual robot objectives
        self.ind_rew = individual_reward
        self.task_reward = task_reward
        self.vi_type = vi_type
        self.log_filename = log_filename
        self.all_colors_list = all_colors_list
        self.true_human_reward = true_human_reward
        self.beliefs_to_himdp = {}

        # Set corpus of possible reward vectors
        # self.corpus =[-0.9, -0.5, 0.5, 1.0] # [1, 1, 1.1, 3]  #
        self.corpus = list(individual_reward)

        # Create robot beliefs of human models
        self.rho_candidates = rho_candidates
        self.beliefs_over_human_models = None
        self.human_beliefs_of_robot = None
        self.create_beliefs_over_human_models()
        self.create_human_beliefs_of_robot()

        # Set belief update history
        self.human_models_history = []
        self.human_beliefs_of_robot_history = []

    def create_beliefs_over_human_models(self):
        possible_hidden_params = {}

        for h_rho in self.rho_candidates:
            # possible_hidden_params[h_rho] = {}
            for vec in list(itertools.permutations(self.corpus)):
                # possible_hidden_params[h_rho][(vec, h_rho)] = 1 / (len(list(itertools.permutations(self.corpus))))
                possible_hidden_params[(vec, h_rho)] = 1 / (len(list(itertools.permutations(self.corpus))) * len(self.rho_candidates))

        self.beliefs_over_human_models = possible_hidden_params

    def create_human_beliefs_of_robot(self):
        possible_hidden_params = {}

        r_rho_human_belief = 0
        # possible_hidden_params[h_rho] = {}
        for vec in list(itertools.permutations(self.corpus)):
            # possible_hidden_params[h_rho][(vec, h_rho)] = 1 / (len(list(itertools.permutations(self.corpus))))
            possible_hidden_params[(vec, r_rho_human_belief)] = 1 / (len(list(itertools.permutations(self.corpus))))

        self.human_beliefs_of_robot = possible_hidden_params

    def get_human_proba_vector_given_state(self, human_state, human_reward, human_rho, robot_reward):

        collaborative_reward_vector = []

        for color in self.all_colors_list:
            rew = 0
            # hypothetical state update
            next_state = copy.deepcopy(human_state)

            if human_state[color] > 0:
                next_state[color] -= 1
                rew += human_reward[color]

                if human_rho > 0:
                    max_robot_rew = -10000
                    robot_color = None
                    for r_color in self.all_colors_list:
                        if next_state[r_color] > 0:
                            r_rew = robot_reward[r_color]
                            if r_rew > max_robot_rew:
                                max_robot_rew = r_rew
                                robot_color = r_color

                    if robot_color is not None:
                        rew += (human_rho * robot_reward[robot_color])
            # else:
            #     rew = -100

            collaborative_reward_vector.append(rew)

        return collaborative_reward_vector

    def update_beliefs_over_human_models(self, human_state, human_action):
        # state = [# blue, # green, # red, # yellow]
        # action = color
        epsilon = 0.01

        # We want P(theta|s,a) = P(s,a,theta)/P(s,a) = P(s,a|theta)P(theta)/P(s,a)
        # = P(a|s,theta)P(s|theta)P(theta)/P(s,a)
        # = P(a|s,theta)P(s)P(theta)/P(s,a)
        # = P(a|s,theta)P(s)P(theta)/P(a|s)P(s)
        # = P(a|s,theta)P(theta)/P(a|s)

        # compute P(a|s)
        prob_action_given_state_numer = human_state[human_action]
        prob_action_given_state_denom = sum(human_state)
        prob_action_given_state = prob_action_given_state_numer / prob_action_given_state_denom

        total_weight = 0
        (robot_reward, robot_rho) = get_key_with_max_value_from_dict(self.human_beliefs_of_robot)
        for human_model_params in self.beliefs_over_human_models:
            (candidate_reward, candidate_rho) = human_model_params

            weight_vector = self.get_human_proba_vector_given_state(human_state, candidate_reward,
                                                                    candidate_rho, robot_reward)

            prob_theta = self.beliefs_over_human_models[human_model_params]

            # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
            # Normalize weight vector P(a|s,theta)
            weight_vector_normed_positive = [(e - min(weight_vector) + epsilon if e - min(weight_vector) == 0
                                              else e - min(weight_vector)) for e in weight_vector]
            weight_vector_normed_positive = [np.round(e / (sum(weight_vector_normed_positive)), 3) for e in
                                             weight_vector_normed_positive]

            restructured_weight_vector = []

            for color in self.all_colors_list:
                if human_state[color] > 0:
                    if weight_vector_normed_positive[color] == 0:
                        restructured_weight_vector.append(0.01)
                    else:
                        restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in self.all_colors_list:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[human_action]

            posterior = (prob_theta * prob_action_given_theta_state) / (prob_action_given_state + epsilon)

            self.beliefs_over_human_models[human_model_params] = posterior
            total_weight += posterior

        # Normalize all beliefs
        for human_model_params in self.beliefs_over_human_models:
            self.beliefs_over_human_models[human_model_params] = self.beliefs_over_human_models[human_model_params] / total_weight

    def update_robots_human_models_with_human_action(self, human_state, human_action, is_done):
        self.update_beliefs_over_human_models(human_state, human_action)

    def update_human_beliefs_of_robot_with_robot_action(self, robot_state, robot_action, is_done):
        # state = [# blue, # green, # red, # yellow]
        # action = color
        epsilon = 0.01

        # We want P(theta|s,a) = P(s,a,theta)/P(s,a) = P(s,a|theta)P(theta)/P(s,a)
        # = P(a|s,theta)P(s|theta)P(theta)/P(s,a)
        # = P(a|s,theta)P(s)P(theta)/P(s,a)
        # = P(a|s,theta)P(s)P(theta)/P(a|s)P(s)
        # = P(a|s,theta)P(theta)/P(a|s)

        # compute P(a|s)
        prob_action_given_state_numer = robot_state[robot_action]
        prob_action_given_state_denom = sum(robot_state)
        prob_action_given_state = prob_action_given_state_numer / prob_action_given_state_denom

        total_weight = 0
        for robot_model_params in self.human_beliefs_of_robot:
            (weight_vector, _) = robot_model_params

            prob_theta = self.human_beliefs_of_robot[robot_model_params]

            # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
            # Normalize weight vector P(a|s,theta)
            weight_vector_normed_positive = [(e - min(weight_vector) + epsilon if e - min(weight_vector) == 0
                                              else e - min(weight_vector)) for e in weight_vector]
            weight_vector_normed_positive = [np.round(e / (sum(weight_vector_normed_positive)), 3) for e in
                                             weight_vector_normed_positive]

            restructured_weight_vector = []

            for color in self.all_colors_list:
                if robot_state[color] > 0:
                    if weight_vector_normed_positive[color] == 0:
                        restructured_weight_vector.append(0.01)
                    else:
                        restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in self.all_colors_list:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[robot_action]

            posterior = (prob_theta * prob_action_given_theta_state) / (prob_action_given_state + epsilon)

            self.human_beliefs_of_robot[robot_model_params] = posterior
            total_weight += posterior

        # Normalize all beliefs
        for robot_model_params in self.human_beliefs_of_robot:
            self.human_beliefs_of_robot[robot_model_params] = self.human_beliefs_of_robot[robot_model_params] / total_weight

    def act(self, input_state, iteration):

        if iteration == 0:
            (human_rew, h_rho) = get_key_with_max_value_from_dict(self.beliefs_over_human_models)
            (robot_rew, r_rho) = get_key_with_max_value_from_dict(self.human_beliefs_of_robot)

            if self.true_human_reward is not None:
                human_rew = self.true_human_reward
                h_rho = 1

            # print("MAX BELIEF (of human by robot)", human_rew)
            # print("MAX BELIEF (of human's beliefs of robot by robot)", robot_rew)
            for (cand_h_vec, cand_h_rho) in self.beliefs_over_human_models:
                # if self.true_human_reward is not None:
                #     cand_h_vec = self.true_human_reward
                #     cand_h_rho = 1
                himdp = HiMDP(input_state, self.all_colors_list, self.task_reward, cand_h_vec, cand_h_rho, robot_rew, r_rho, self.vi_type)
                himdp.enumerate_states()
                himdp.value_iteration()
                self.beliefs_to_himdp[(cand_h_vec, cand_h_rho)] = himdp

        final_action_distribution = [0] * (len(self.all_colors_list) + 1)
        final_action_distribution = np.array(final_action_distribution)

        # print("self.beliefs_to_himdp", self.beliefs_to_himdp)
        for (cand_h_vec, cand_h_rho) in self.beliefs_to_himdp:
            himdp = self.beliefs_to_himdp[(cand_h_vec, cand_h_rho)]
            flat_state = himdp.state_to_tuple(input_state)
            state_idx = himdp.state_to_idx[flat_state]

            action_distribution = np.array(himdp.policy[state_idx])
            final_action_distribution = final_action_distribution + (self.beliefs_over_human_models[(cand_h_vec, cand_h_rho)] * action_distribution)

        # pdb.set_trace()
        best_color = None
        highest_prob = -1000
        for color in self.all_colors_list:
            # color_idx = self.himdp.action_to_idx[color]
            if input_state[color] > 0:
                if final_action_distribution[color] > highest_prob:
                    best_color = color
                    highest_prob = action_distribution[color]

        # color_selected = self.himdp.idx_to_action[best_color]

        return best_color


















