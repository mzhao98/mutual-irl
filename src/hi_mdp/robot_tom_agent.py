import pdb

import numpy as np
import operator
import random

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

from himdp_1player import HiMDP
from human_hypothesis import Human_Hypothesis

class Robot_Model():
    def __init__(self, individual_reward, first_order, num_particles=1):
        self.ind_rew = individual_reward
        self.num_particles = num_particles
        self.team_weights = [1, 1, 1, 1]
        self.first_order = first_order
        self.sample_hidden_parameters()



    def sample_hidden_parameters(self):
        self.num_particles = 100

        possible_hidden_params = {}
        self.param_to_model = {}
        for i in range(int(self.num_particles/2)):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            w = random.uniform(-1, 1)
            weight_vector = (np.round(x, 2), np.round(y, 2), np.round(z, 2), np.round(w, 2))

            possible_hidden_params[(weight_vector, 1)] = 1 / self.num_particles
            self.param_to_model[(weight_vector, 1)] = Human_Hypothesis(weight_vector, 2)


        for i in range(int(self.num_particles / 2)):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            w = random.uniform(-1, 1)
            weight_vector = (np.round(x, 2), np.round(y, 2), np.round(z, 2), np.round(w, 2))

            if self.first_order is False:
                depth = 2
            else:
                depth = 1
            possible_hidden_params[(weight_vector, depth)] = 1 / self.num_particles


            self.param_to_model[(weight_vector, depth)] = Human_Hypothesis(weight_vector, 2)

        self.beliefs = possible_hidden_params



    def update_particle_weights(self, human_state, human_action):
        # robot_state = [# blue, # green, # red, # yellow]
        # robot action = color
        epsilon = 0.001
        total_weight = 0

        prob_action_given_state_numer = human_state[human_action]
        # print("robot_state", robot_state)
        prob_action_given_state_denom = sum(human_state)

        prob_action_given_state = prob_action_given_state_numer / prob_action_given_state_denom
        # print("prob_action_given_state", prob_action_given_state)

        for human_model_params in self.beliefs:
            human_model = self.param_to_model[human_model_params]
            weight_vector = human_model.get_proba_vector_given_state(human_state)

            prob_theta = self.beliefs[human_model_params]
            # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
            # weight 0f particle self.beliefs[weight_vector]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / (sum(weight_vector_normed_positive)) for e in
                                             weight_vector_normed_positive]

            restructured_weight_vector = []

            for color in COLOR_LIST:
                if human_state[color] > 0:
                    restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in COLOR_LIST:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[human_action]
            posterior = (prob_theta * prob_action_given_theta_state) / (prob_action_given_state + epsilon)
            self.beliefs[human_model_params] = posterior
            total_weight += posterior

        for human_model_params in self.beliefs:
            self.beliefs[human_model_params] /= (total_weight + epsilon)

    def resample_particles(self):
        possible_weights = {}
        new_param_to_model = {}

        population = list(self.beliefs.keys())
        denom = sum(list(self.beliefs.values()))
        # population_weights = [np.round(elem, 2) for elem in list(self.beliefs.values())]
        # print("list(self.beliefs.values())", list(self.beliefs.values()))
        population_weights = [elem / denom for elem in list(self.beliefs.values())]
        # print("population_weights", population_weights)
        # print("population", len(population))
        # pdb.set_trace()
        sampled_particle_indices = np.random.choice(np.arange(len(population)), self.num_particles,
                                                    p=population_weights, replace=True)

        for i in sampled_particle_indices:
            x_noise = random.normalvariate(0, 0.1)
            y_noise = random.normalvariate(0, 0.1)
            z_noise = random.normalvariate(0, 0.1)
            w_noise = random.normalvariate(0, 0.1)
            x, y, z, w = population[i][0]
            depth = population[i][1]
            initial_human_model = self.param_to_model[population[i]]

            x, y, z, w = x + x_noise, y + y_noise, z + z_noise, w + w_noise

            weight_vector = (np.round(x, 2), np.round(y, 2), np.round(z, 2), np.round(w, 2))
            # weight_vector = (x, y, z, w)
            possible_weights[(weight_vector, depth)] = 1 / self.num_particles
            new_human_model = Human_Hypothesis(weight_vector, depth)

            # r = np.random.uniform(0, 1)
            # if r < 0.5:
            new_human_model.set_beliefs(initial_human_model.beliefs)

            new_param_to_model[(weight_vector, depth)] = new_human_model

        self.beliefs = possible_weights
        self.param_to_model = new_param_to_model

    def update_with_partner_action(self, human_state, human_action):
        self.update_particle_weights(human_state, human_action)
        self.resample_particles()

    def update_human_models_with_robot_action(self, robot_state, robot_action):
        for param in self.param_to_model:
            self.param_to_model[param].update_particle_weights(robot_state, robot_action)
            self.param_to_model[param].resample_particles()



    def get_max_likelihood_human_model(self):
        best_model = None
        best_params = None
        highest_prob = -1000
        for params in self.beliefs:
            if self.beliefs[params] > highest_prob:
                best_model = self.param_to_model[params]
                highest_prob = self.beliefs[params]
                best_params = params
        return best_params, best_model

    def act(self, input_state, robot_history, human_history):
        (human_pref, human_depth), human_model = self.get_max_likelihood_human_model()

        self.himdp = HiMDP(human_pref, human_depth)
        self.himdp.set_human_model(human_model)
        self.himdp.enumerate_states()
        self.himdp.value_iteration()


        state = [input_state, robot_history, human_history]
        flat_state = self.himdp.flatten_to_tuple(state)
        state_idx = self.himdp.state_to_idx[flat_state]

        action_distribution = self.himdp.policy[state_idx]
        # pdb.set_trace()
        best_color = None
        highest_prob = -1000
        for color in COLOR_LIST:
            color_idx = self.himdp.action_to_idx[color]
            if input_state[color] > 0:
                if action_distribution[color_idx] > highest_prob:
                    best_color = color
                    highest_prob = action_distribution[color_idx]


        # best_color = None
        # highest_prob = -1000
        # for joint_action_idx in self.himdp.idx_to_action:
        #     prob = action_distribution[joint_action_idx]
        #     color = self.himdp.idx_to_action[joint_action_idx][0]
        #     if input_state[color] > 0:
        #         if prob > highest_prob:
        #             highest_prob = prob
        #             best_color = color
        #
        # color_selected = best_color
        # color_selected = self.himdp.idx_to_action[int(self.himdp.pi[state_idx, 0])]
        color_selected = self.himdp.idx_to_action[best_color]

        return color_selected


















