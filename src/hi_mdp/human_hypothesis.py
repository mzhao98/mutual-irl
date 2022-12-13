import copy

import numpy as np
import operator
import random
import matplotlib.pyplot as plt
import itertools

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

class Human_Hypothesis():
    def __init__(self, individual_reward, depth, num_particles, r_scalar=1, log_filename=None):
        self.ind_rew = individual_reward
        self.depth = depth
        self.num_particles = num_particles
        self.team_weights = [1, 1, 1, 1]
        self.complete_robot_history = []
        self.log_filename = log_filename
        self.r_scalar = r_scalar
        # self.corpus =  [-0.9, -0.5, 0.5, 1.0] #[1,1,1.1,3]
        self.corpus = list(individual_reward)

        self.beliefs = None
        self.collab_weight_vector = None
        self.pf_history = []
        if self.depth == 2:
            self.construct_robot_pf()

    def set_beliefs(self, beliefs):
        self.beliefs = beliefs

    def set_mdp(self, mdp):
        self.mdp = mdp


    def get_percent_particles_w_correct_prediction(self, true_state, true_action):
        total_weighted_accuracy = 0.0
        epsilon = 0.000000001

        top_1 = dict(sorted(self.beliefs.items(), key=operator.itemgetter(1), reverse=True)[:1])
        for weight_vector in top_1:
            prob_theta = self.beliefs[weight_vector]
            # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
            # weight 0f particle self.beliefs[weight_vector]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / (sum(weight_vector_normed_positive)) for e in
                                             weight_vector_normed_positive]

            restructured_weight_vector = []

            for color in COLOR_LIST:
                if true_state[color] > 0:
                    restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in COLOR_LIST:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[true_action]

            if abs(prob_action_given_theta_state - max(restructured_weight_vector)) < 0.02:
                accuracy = 1.0
            else:
                accuracy = 0.0

            weighted_accuracy = accuracy * prob_theta
            total_weighted_accuracy = weighted_accuracy

        if self.log_filename is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nTrue human's accuracy on robot = {total_weighted_accuracy}")
        return total_weighted_accuracy

    def update_with_partner_action(self, robot_state, robot_action, is_done):
        if self.depth == 2:
            weighted_accuracy = self.get_percent_particles_w_correct_prediction(robot_state, robot_action)
            self.update_particle_weights(robot_state, robot_action)
            if is_done:
                self.resample_particles()
                self.pf_history.append(
                    self.get_K_weighted_combination_belief(self.beliefs, weighted_accuracy, k=1, resampled=True))

            else:
                self.pf_history.append(
                    self.get_K_weighted_combination_belief(self.beliefs, weighted_accuracy, k=1, resampled=False))




    def construct_robot_pf(self):
        # possible_weights = {}
        # for i in range(self.num_particles):
        #     x = random.uniform(-1, 1)
        #     y = random.uniform(-1, 1)
        #     z = random.uniform(-1, 1)
        #     w = random.uniform(-1, 1)
        #     weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))
        #
        #     possible_weights[weight_vector] = 1 / self.num_particles
        # self.beliefs = possible_weights

        possible_weights = {}
        permutes = list(itertools.permutations(self.corpus))
        for vec in permutes:
            possible_weights[vec] = 1/len(permutes)
        self.beliefs = possible_weights

    def update_particle_weights(self, robot_state, robot_action):
        # robot_state = [# blue, # green, # red, # yellow]
        # robot action = color

        if self.depth == 1:
            return

        epsilon = 0.00001
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
                    if weight_vector_normed_positive[color] == 0:
                        restructured_weight_vector.append(0.01)
                    else:
                        restructured_weight_vector.append(weight_vector_normed_positive[color])
                else:
                    restructured_weight_vector.append(0)

            sum_weight_values = sum(restructured_weight_vector)
            for color in COLOR_LIST:
                restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

            prob_action_given_theta_state = restructured_weight_vector[robot_action]
            # if abs(prob_action_given_theta_state-max(restructured_weight_vector)) < 0.02:
            #     prob_action_given_theta_state = 0.8
            # else:
            #     prob_action_given_theta_state = 0.2
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
        sampled_particle_indices = np.random.choice(np.arange(len(population)), int(len(self.beliefs) * 0.75),
                                                    p=population_weights, replace=True)

        for i in sampled_particle_indices:
            x_noise = random.normalvariate(0, 0.05)
            y_noise = random.normalvariate(0, 0.05)
            z_noise = random.normalvariate(0, 0.05)
            w_noise = random.normalvariate(0, 0.05)
            x, y, z, w = population[i]

            x, y, z, w = x + x_noise, y + y_noise, z + z_noise, w + w_noise

            weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))
            # weight_vector = (x, y, z, w)
            possible_weights[weight_vector] = 1 / len(sampled_particle_indices)

        self.beliefs = possible_weights

    def get_K_weighted_combination_vector(self, k=1):

        top_5 = dict(sorted(self.beliefs.items(), key=operator.itemgetter(1), reverse=True)[:k])
        weighted_combination = np.array([0, 0, 0, 0])
        total_prob = sum([self.beliefs[weight_vector] for weight_vector in top_5])
        if total_prob == 0:
            total_prob = 0.00001

        for weight_vector in top_5:
            prob = self.beliefs[weight_vector]/total_prob
            weighted_combination = weighted_combination + (np.array(list(weight_vector)) * prob)

        weighted_combination = tuple(weighted_combination)

        return weighted_combination, total_prob

    def get_K_weighted_combination_belief(self, beliefs, weighted_accuracy, k=1, resampled=False):

        top_5 = dict(sorted(beliefs.items(), key=operator.itemgetter(1), reverse=True)[:k])
        weighted_combination = np.array([0, 0, 0, 0])
        total_prob = sum([beliefs[weight_vector] for weight_vector in top_5])
        if total_prob == 0:
            total_prob = 0.0001

        for weight_vector in top_5:
            prob = beliefs[weight_vector]/total_prob
            weighted_combination = weighted_combination + (np.array(list(weight_vector)) * prob)

        weighted_combination = tuple(weighted_combination)

        if self.log_filename is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nTrue human's belief of robot = {(weighted_combination, weighted_accuracy, resampled)}")
        return (weighted_combination, weighted_accuracy, resampled)




    def act(self, state, robot_history, human_history):
        weight_vector = []
        human_action = None
        if self.depth == 1:
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                if state[color] == 0:
                    weight_vector.append(-100)
                    continue
                rew = self.ind_rew[color]
                if rew >= max_rew:
                    max_rew = rew
                    best_color = color
                weight_vector.append(rew)
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
            robot_rew, confidence = self.get_K_weighted_combination_vector()
            confidence_scalar = 0.0
            if confidence > 0.5:
                confidence_scalar = 1.0
            # confidence_scalar = 1.0
            if self.log_filename is not None:
                with open(self.log_filename, 'a') as f:
                    f.write(f"\nTrue human's confidence = {confidence}, confidence scalar = {confidence_scalar}")

            # confidence_scalar = 1/(1 + np.exp(-confidence))
            # robot_rew_min = min(robot_rew)
            # robot_rew = [elem - robot_rew_min for elem in robot_rew]
            # robot_rew_sum = sum(robot_rew)
            # robot_rew = [elem/robot_rew_sum for elem in robot_rew]

            # self_rew_min = min(self.ind_rew)
            # normed_self_rew = [elem - self_rew_min for elem in self.ind_rew]
            # normed_self_rew_sum = sum(normed_self_rew)
            # normed_self_rew = [elem/normed_self_rew_sum for elem in normed_self_rew]
            normed_self_rew = self.ind_rew
            # print(f"partner_rew: {robot_rew}, self: {self.ind_rew}")
            alpha = 0.6
            max_rew = -10000
            best_color = None
            for color in COLOR_LIST:
                if state[color] == 0:
                    weight_vector.append(None)
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
                    if r_rew >= max_robot_rew:
                        max_robot_rew = r_rew
                        robot_color = r_color

                next_robot_rew = 0
                if robot_color:
                    next_robot_rew = robot_rew[robot_color]

                rew = normed_self_rew[color] + (self.r_scalar * confidence_scalar * next_robot_rew)
                weight_vector.append(rew)
                if rew >= max_rew:
                    max_rew = rew
                    best_color = color
            human_action = best_color


        if self.log_filename is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nTrue human's acting weight vector = {weight_vector}")
        self.collab_weight_vector = weight_vector
        return human_action




    def get_collab_proba_for_d2(self, state):
        robot_rew = self.get_K_weighted_combination_vector()
        # robot_rew_min = min(robot_rew)
        # robot_rew = [elem - robot_rew_min for elem in robot_rew]
        # robot_rew_sum = sum(robot_rew)
        # robot_rew = [elem/robot_rew_sum for elem in robot_rew]

        # self_rew_min = min(self.ind_rew)
        # normed_self_rew = [elem - self_rew_min for elem in self.ind_rew]
        # normed_self_rew_sum = sum(normed_self_rew)
        # normed_self_rew = [elem/normed_self_rew_sum for elem in normed_self_rew]
        normed_self_rew = self.ind_rew

        collaborative_reward_vector = []

        for color in COLOR_LIST:
            if state[color] == 0:
                collaborative_reward_vector.append(-10)
            else:
                # hypothetical state update
                next_state = copy.deepcopy(state)
                next_state[color] -= 1
                max_robot_rew = -10000
                robot_color = None
                for r_color in COLOR_LIST:
                    if next_state[r_color] == 0:
                        continue
                    r_rew = robot_rew[r_color]
                    if r_rew >= max_robot_rew:
                        max_robot_rew = r_rew
                        robot_color = r_color

                next_robot_rew = 0
                if robot_color:
                    next_robot_rew = robot_rew[robot_color]

                rew = normed_self_rew[color] + (self.r_scalar * next_robot_rew)
                collaborative_reward_vector.append(rew)

        return collaborative_reward_vector

    def get_proba_vector_given_state(self, state):
        robot_rew = None
        weight_vector = []
        if self.depth == 1:
            for color in COLOR_LIST:
                rew = self.ind_rew[color]

                # if state[color] == 0:
                #     weight_vector.append(-1000)
                # else:
                #
                weight_vector.append(rew)


        elif self.depth == 2:
            robot_rew, confidence = self.get_K_weighted_combination_vector()

            confidence_scalar = 0.0
            if confidence > 0.5:
                confidence_scalar = 1.0
            # confidence_scalar = 1.0
            # robot_rew_min = min(robot_rew)
            # robot_rew = [elem - robot_rew_min for elem in robot_rew]
            # robot_rew_sum = sum(robot_rew)
            # robot_rew = [elem / robot_rew_sum for elem in robot_rew]

            # self_rew_min = min(self.ind_rew)
            # normed_self_rew = [elem - self_rew_min for elem in self.ind_rew]
            # normed_self_rew_sum = sum(normed_self_rew)
            # normed_self_rew = [elem / normed_self_rew_sum for elem in normed_self_rew]
            normed_self_rew = self.ind_rew

            collaborative_reward_vector = []

            for color in COLOR_LIST:
                # if state[color] == 0:
                #     collaborative_reward_vector.append(-10000)
                # else:
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
                if robot_color is not None:
                    next_robot_rew = robot_rew[robot_color]
                # print("self.r_scalar", self.r_scalar)
                rew = normed_self_rew[color] + (self.r_scalar * confidence_scalar * next_robot_rew)
                collaborative_reward_vector.append(rew)
            weight_vector = collaborative_reward_vector
            # if self.depth == 2:
            #     print("collaborative_reward_vector", collaborative_reward_vector)

        if self.log_filename is not None:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nTrue human's weight vector = {weight_vector}")
        self.collab_weight_vector = weight_vector
        return weight_vector, robot_rew

    def plot_weight_updates(self, savename, savename_accuracy):
        # TODO

        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []

        particle_accuracy_over_time = []
        resampled_over_time = []

        for (weighted_combination, weighted_accuracy, resampled) in self.pf_history:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])
            particle_accuracy_over_time.append(weighted_accuracy)
            resampled_over_time.append(resampled)

        plt.figure()
        plt.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        plt.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        plt.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        plt.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')

        plt.xlabel("Timestep")
        plt.ylabel("Belief Weight and Depth")
        plt.legend()
        plt.title(f"Human's Beliefs of Robot: ")
        plt.savefig(f"{savename}")
        plt.close()
        plt.cla()
        plt.clf()

        plt.figure()
        plt.plot(range(len(particle_accuracy_over_time)), particle_accuracy_over_time, color='blue', label='blue')
        # print("particle_accuracy_over_time", particle_accuracy_over_time)
        for i in range(len(resampled_over_time)):
            if resampled_over_time[i] is True:
                plt.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='r')
            else:
                plt.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='g')
        plt.xlabel("Timestep")
        plt.ylabel("Particle Weighted Accuracy")
        plt.legend()
        plt.title(f"Human's PF Accuracy Whenever Robot Acts")
        plt.savefig(f"{savename_accuracy}")
        plt.close()
        plt.cla()
        plt.clf()


    def plot_beliefs_to_axes(self, ax1, ax2):
        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []

        particle_accuracy_over_time = []
        resampled_over_time = []

        for (weighted_combination, weighted_accuracy, resampled) in self.pf_history:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])
            particle_accuracy_over_time.append(weighted_accuracy)
            resampled_over_time.append(resampled)

        ax1.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        ax1.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        ax1.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        ax1.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')

        ax1.set(xlabel='Timestep', ylabel="Belief Weight and Depth")
        ax1.legend()
        ax1.title.set_text(f"Human's Beliefs of Robot\n Combined Rew: {[(np.round(elem, 2) if type(elem)==int else elem) for elem in self.collab_weight_vector]}")

        ax2.plot(range(len(particle_accuracy_over_time)), particle_accuracy_over_time, color='blue', label='blue')
        # print("particle_accuracy_over_time", particle_accuracy_over_time)
        for i in range(len(resampled_over_time)):
            if resampled_over_time[i] is True:
                ax2.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='r')
            else:
                ax2.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='g')

        ax2.set(xlabel='Timestep', ylabel="Particle Weighted Accuracy")
        # ax2.legend()
        ax2.title.set_text(f"Human's PF Accuracy Whenever Robot Acts")

















