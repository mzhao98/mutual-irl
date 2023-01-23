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


def get_key_with_max_value_from_dict(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0]

def get_key_value_pair_with_max_value_from_dict(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0], dictionary[max(dictionary.items(), key=operator.itemgetter(1))[0]]


class Human_Hypothesis():
    def __init__(self, individual_reward, all_colors_list, task_reward, individual_rho, log_filename=''):
        # Set individual robot objectives
        self.ind_rew = individual_reward
        self.task_reward = task_reward
        self.ind_rho = individual_rho
        self.log_filename = log_filename
        self.all_colors_list = all_colors_list

        # Set corpus of possible reward vectors
        # self.corpus =[-0.9, -0.5, 0.5, 1.0] # [1, 1, 1.1, 3]  #
        self.corpus = list(individual_reward)

        # Create human beliefs of robot models
        self.beliefs_of_robot = None
        self.create_beliefs_of_robot()

        # Set belief update history
        self.human_models_history = []
        self.human_beliefs_of_robot_history = []

    def create_beliefs_of_robot(self):
        possible_hidden_params = {}

        r_rho_human_belief = 0
        # possible_hidden_params[h_rho] = {}
        for vec in list(itertools.permutations(self.corpus)):
            # possible_hidden_params[h_rho][(vec, h_rho)] = 1 / (len(list(itertools.permutations(self.corpus))))
            possible_hidden_params[(vec, r_rho_human_belief)] = 1 / (len(list(itertools.permutations(self.corpus))))

        self.beliefs_of_robot = possible_hidden_params

    def update_beliefs_of_robot_with_robot_action(self, robot_state, robot_action, is_done):
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
        for robot_model_params in self.beliefs_of_robot:
            (weight_vector, _) = robot_model_params

            prob_theta = self.beliefs_of_robot[robot_model_params]

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

            self.beliefs_of_robot[robot_model_params] = posterior
            total_weight += posterior

        # Normalize all beliefs
        for robot_model_params in self.beliefs_of_robot:
            self.beliefs_of_robot[robot_model_params] = self.beliefs_of_robot[robot_model_params] / total_weight

    def act(self, input_state):
        human_action = None
        max_reward = -10000
        (robot_rew, r_rho) = get_key_with_max_value_from_dict(self.beliefs_of_robot)
        # print("MAX BELIEF (of robot by human)", robot_rew)

        for color in self.all_colors_list:
            rew = 0
            # hypothetical state update
            next_state = copy.deepcopy(input_state)

            if input_state[color] > 0:
                next_state[color] -= 1
                rew += self.ind_rew[color]

                if self.ind_rho > 0:
                    max_robot_rew = -10000
                    robot_color = None
                    for r_color in self.all_colors_list:
                        if next_state[r_color] > 0:
                            r_rew = robot_rew[r_color]
                            if r_rew > max_robot_rew:
                                max_robot_rew = r_rew
                                robot_color = r_color

                    if robot_color is not None:
                        rew += (self.ind_rho * robot_rew[robot_color])

                if rew > max_reward:
                    max_reward = rew
                    human_action = color

        return human_action



















