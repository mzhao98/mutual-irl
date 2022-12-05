import copy

import numpy as np
import operator
import random
import itertools

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]


class Human_Hypothesis():
    def __init__(self, individual_reward, depth, num_particles=100):
        self.ind_rew = individual_reward
        self.depth = depth
        self.num_particles = num_particles
        self.team_weights = [1, 1, 1, 1]

        self.beliefs = None
        if self.depth == 2:
            self.construct_robot_pf()

    def set_beliefs(self, beliefs):
        self.beliefs = beliefs

    def set_mdp(self, mdp):
        self.mdp = mdp

    def update_with_partner_action(self, robot_state, robot_action):
        if self.depth == 2:
            self.update_particle_weights(robot_state, robot_action)
            # self.resample_particles()


    def construct_robot_pf(self):
        possible_weights = {}

        possible_rews = list(itertools.permutations([-1, -0.5, 0.5, 1.0]))
        for i in range(len(possible_rews)):
            weight_vector = possible_rews[i]

            possible_weights[weight_vector] = 1 / self.num_particles
        self.beliefs = possible_weights

    def update_particle_weights(self, robot_state, robot_action):
        # robot_state = [# blue, # green, # red, # yellow]
        # robot action = color

        if self.depth == 1:
            return

        epsilon = 0.001
        total_weight = 0

        prob_action_given_state_numer = robot_state[robot_action]
        # print("robot_state", robot_state)
        prob_action_given_state_denom = sum(robot_state)

        prob_action_given_state = prob_action_given_state_numer / prob_action_given_state_denom
        # print("prob_action_given_state", prob_action_given_state)

        for weight_vector in self.beliefs:
            prob_theta = self.beliefs[weight_vector]
            # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
            # weight 0f particle self.beliefs[weight_vector]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / (sum(weight_vector_normed_positive)) for e in
                                             weight_vector_normed_positive]
            # print("weight_vector_normed_positive", weight_vector_normed_positive)
            # print("robot_state", robot_state)
            # print("robot_action", robot_action)
            restructured_weight_vector = []

            for color in COLOR_LIST:
                if robot_state[color] > 0:
                    restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in COLOR_LIST:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[robot_action]
            # print("prob_action_given_theta_state", prob_action_given_theta_state)
            posterior = (prob_theta * prob_action_given_theta_state) / (prob_action_given_state + epsilon)
            self.beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in self.beliefs:
            self.beliefs[weight_vector] /= (total_weight + epsilon)

    def resample_particles(self):
        if self.depth == 1:
            return

        possible_weights = {}

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
            x, y, z, w = population[i]

            x, y, z, w = x + x_noise, y + y_noise, z + z_noise, w + w_noise

            weight_vector = (np.round(x, 2), np.round(y, 2), np.round(z, 2), np.round(w, 2))
            # weight_vector = (x, y, z, w)
            possible_weights[weight_vector] = 1 / self.num_particles

        self.beliefs = possible_weights

    def get_K_weighted_combination_belief(self, k=5):

        top_5 = dict(sorted(self.beliefs.items(), key=operator.itemgetter(1), reverse=True)[:k])
        weighted_combination = np.array([0, 0, 0, 0])
        for weight_vector in top_5:
            prob = self.beliefs[weight_vector]
            weighted_combination = weighted_combination + (np.array(list(weight_vector)) * prob)

        weighted_combination = tuple(weighted_combination)

        return weighted_combination

    def act(self, state, robot_history, human_history):
        human_action = None
        if self.depth == 1:
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                if state[color] == 0:
                    continue
                rew = self.ind_rew[color]
                if rew > max_rew:
                    max_rew = rew
                    best_color = color
            human_action = best_color

        # elif self.depth == 2:
        #     robot_rew = self.get_K_weighted_combination_belief()
        #     robot_rew_min = min(robot_rew)
        #     robot_rew = [elem - robot_rew_min for elem in robot_rew]
        #     robot_rew_sum = sum(robot_rew)
        #     robot_rew = [elem/robot_rew_sum for elem in robot_rew]
        #
        #     self_rew_min = min(self.ind_rew)
        #     normed_self_rew = [elem - self_rew_min for elem in self.ind_rew]
        #     normed_self_rew_sum = sum(normed_self_rew)
        #     normed_self_rew = [elem/normed_self_rew_sum for elem in normed_self_rew]
        #     # print(f"partner_rew: {robot_rew}, self: {self.ind_rew}")
        #     alpha = 0.6
        #     max_rew = -10000
        #     best_color = None
        #     for color in COLOR_LIST:
        #         if state[color] == 0:
        #             continue
        #         rew = - (alpha * robot_rew[color]) + ((1-alpha) * normed_self_rew[color])
        #         if rew > max_rew:
        #             max_rew = rew
        #             best_color = color
        #     human_action = best_color

        elif self.depth == 2:
            robot_rew = self.get_K_weighted_combination_belief()
            robot_rew_min = min(robot_rew)
            robot_rew = [elem - robot_rew_min for elem in robot_rew]
            robot_rew_sum = sum(robot_rew)
            robot_rew = [elem/robot_rew_sum for elem in robot_rew]

            self_rew_min = min(self.ind_rew)
            normed_self_rew = [elem - self_rew_min for elem in self.ind_rew]
            normed_self_rew_sum = sum(normed_self_rew)
            normed_self_rew = [elem/normed_self_rew_sum for elem in normed_self_rew]
            # print(f"partner_rew: {robot_rew}, self: {self.ind_rew}")
            alpha = 0.6
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                if state[color] == 0:
                    continue

                # hypothetical state update
                next_state = copy.deepcopy(state)
                next_state[color] -= 1
                max_robot_rew = -10000
                robot_color = None
                for r_color in COLOR_LIST:
                    if next_state[r_color] == 0:
                        continue
                    r_rew = robot_rew[r_color]
                    if r_rew > max_robot_rew:
                        max_robot_rew = r_rew
                        robot_color = r_color

                next_robot_rew = 0
                if robot_color:
                    next_robot_rew = robot_rew[robot_color]


                rew = normed_self_rew[color] + next_robot_rew
                if rew > max_rew:
                    max_rew = rew
                    best_color = color
            human_action = best_color

        return human_action

    def get_proba_vector_given_state(self, state):
        human_action = None
        weight_vector = []
        if self.depth == 1:
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                rew = self.ind_rew[color]
                weight_vector.append(rew)
                if state[color] == 0:
                    continue

                if rew > max_rew:
                    max_rew = rew
                    best_color = color
            human_action = best_color

        elif self.depth == 2:
            robot_rew = self.get_K_weighted_combination_belief()
            robot_rew = [elem/np.linalg.norm(robot_rew) for elem in robot_rew]

            normed_task_rew = [elem / np.linalg.norm(self.team_weights) for elem in self.team_weights]
            normed_self_rew = [elem / np.linalg.norm(self.ind_rew) for elem in self.ind_rew]
            # print(f"partner_rew: {robot_rew}, self: {self.ind_rew}")
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                rew = normed_task_rew[color] - robot_rew[color] + normed_self_rew[color]
                weight_vector.append(rew)

                if state[color] == 0:
                    continue

                if rew > max_rew:
                    max_rew = rew
                    best_color = color
            human_action = best_color
        return weight_vector





















