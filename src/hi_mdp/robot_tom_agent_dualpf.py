import pdb

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

# from hip_mdp_1player_optimized import HiMDP
from hip_mdp_1player import HiMDP
from human_hypothesis import Human_Hypothesis


class Robot_Model():
    def __init__(self, individual_reward, mm_order, vi_type, num_particles, h_scalar, r_scalar, log_filename):
        self.ind_rew = individual_reward
        self.num_particles = num_particles
        self.team_weights = [1, 1, 1, 1]
        # self.corpus =[-0.9, -0.5, 0.5, 1.0] # [1, 1, 1.1, 3]  #
        self.corpus = list(individual_reward)
        self.mm_order = mm_order

        self.num_particles = num_particles
        self.vi_type = vi_type
        self.h_scalar = h_scalar
        self.r_scalar = r_scalar

        self.log_filename = log_filename
        self.param_to_model = {}
        self.beliefs = None
        self.sample_hidden_permutes()
        self.pf_history = []
        self.pf_history.append(self.get_K_weighted_combination_belief(self.beliefs, 0, k=1))

        self.human_ground_truth_history = []
        self.inferred_human_beliefs_of_robot = []

    def sample_hidden_permutes(self):
        possible_hidden_params = {}

        if self.mm_order == 'first':
            possible_hidden_params[1] = {}
            self.param_to_model[1] = {}
            permutes = list(itertools.permutations(self.corpus))
            for vec in permutes:
                possible_hidden_params[1][(vec, 1)] = 1 / len(permutes)
                self.param_to_model[1][(vec, 1)] = Human_Hypothesis(vec, 1, len(permutes), r_scalar=self.r_scalar)

        elif self.mm_order == 'second':
            possible_hidden_params[2] = {}
            self.param_to_model[2] = {}
            permutes = list(itertools.permutations(self.corpus))
            for vec in permutes:
                possible_hidden_params[2][(vec, 2)] = 1 / len(permutes)
                self.param_to_model[2][(vec, 2)] = Human_Hypothesis(vec, 2, len(permutes), r_scalar=self.r_scalar)

        else:
            possible_hidden_params[1] = {}
            possible_hidden_params[2] = {}
            self.param_to_model[1] = {}
            self.param_to_model[2] = {}

            permutes = list(itertools.permutations(self.corpus))
            for vec in permutes:
                possible_hidden_params[1][(vec, 1)] = 1 / (len(permutes))
                self.param_to_model[1][(vec, 1)] = Human_Hypothesis(vec, 1, len(permutes), r_scalar=self.r_scalar)

            permutes = list(itertools.permutations(self.corpus))
            for vec in permutes:
                possible_hidden_params[2][(vec, 2)] = 1 / (len(permutes))
                self.param_to_model[2][(vec, 2)] = Human_Hypothesis(vec, 2, len(permutes), r_scalar=self.r_scalar)

        self.beliefs = possible_hidden_params

    def sample_hidden_parameters(self):

        possible_hidden_params = {}

        if self.mm_order == 'first':
            # possible_hidden_params = {}
            self.param_to_model = {}
            for i in range(int(self.num_particles - 1)):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                w = random.uniform(-1, 1)
                weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))

                possible_hidden_params[(weight_vector, 1)] = 1 / self.num_particles - 1
                self.param_to_model[(weight_vector, 1)] = Human_Hypothesis(weight_vector, 1, self.num_particles)



        elif self.mm_order == 'second':
            # possible_hidden_params = {}
            self.param_to_model = {}
            for i in range(int(self.num_particles - 1)):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                w = random.uniform(-1, 1)
                weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))

                possible_hidden_params[(weight_vector, 2)] = 1 / self.num_particles - 1
                self.param_to_model[(weight_vector, 2)] = Human_Hypothesis(weight_vector, 2, self.num_particles)


        else:
            # possible_hidden_params = {}
            self.param_to_model = {}
            for i in range(int(self.num_particles)):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                w = random.uniform(-1, 1)
                weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))

                possible_hidden_params[(weight_vector, 1)] = 1 / (self.num_particles * 2 - 1)
                self.param_to_model[(weight_vector, 1)] = Human_Hypothesis(weight_vector, 1, self.num_particles)

            for i in range(int(self.num_particles)):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
                w = random.uniform(-1, 1)
                weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))

                possible_hidden_params[(weight_vector, 2)] = 1 / (self.num_particles * 2 - 1)
                self.param_to_model[(weight_vector, 2)] = Human_Hypothesis(weight_vector, 2, self.num_particles)

        true = (0.9, 0.1, -0.9, 0.2)
        possible_hidden_params[(true, 2)] = 1 / self.num_particles
        self.param_to_model[(true, 2)] = Human_Hypothesis(true, 2, self.num_particles)
        self.beliefs = possible_hidden_params

    def update_particle_weights(self, human_state, human_action):
        # robot_state = [# blue, # green, # red, # yellow]
        # robot action = color
        epsilon = 0.01


        prob_action_given_state_numer = human_state[human_action]
        # print("robot_state", robot_state)
        prob_action_given_state_denom = sum(human_state)

        prob_action_given_state = prob_action_given_state_numer / prob_action_given_state_denom

        if abs(prob_action_given_state - 1) < epsilon:
            with open(self.log_filename, 'a') as f:
                f.write(f"\nNo need to update robot beliefs")
            return
        # print("prob_action_given_state", prob_action_given_state)
        # print("human_state, human_action", (human_state, human_action))
        for mm_depth in self.beliefs:
            total_weight = 0
            for human_model_params in self.beliefs[mm_depth]:
                depth = human_model_params[1]
                human_model = self.param_to_model[mm_depth][human_model_params]
                # human_model_depth = human_model.depth
                weight_vector, _ = human_model.get_proba_vector_given_state(human_state)
                # print("weight_vector", weight_vector)

                prob_theta = self.beliefs[mm_depth][human_model_params]
                # weight_vector = (rew blue, rew green, rew red, rew yellow) tuple
                # weight 0f particle self.beliefs[weight_vector]
                weight_vector_normed_positive = [(e - min(weight_vector) + epsilon if e - min(weight_vector) == 0
                                                  else e - min(weight_vector)) for e in weight_vector]
                weight_vector_normed_positive = [np.round(e / (sum(weight_vector_normed_positive)), 2) for e in
                                                 weight_vector_normed_positive]

                restructured_weight_vector = []

                for color in COLOR_LIST:
                    if human_state[color] > 0:
                        if weight_vector_normed_positive[color] == 0:
                            restructured_weight_vector.append(0.01)
                        else:
                            restructured_weight_vector.append(weight_vector_normed_positive[color])
                    else:
                        restructured_weight_vector.append(0)

                sum_weight_values = sum(restructured_weight_vector)
                for color in COLOR_LIST:
                    restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

                prob_action_given_theta_state = restructured_weight_vector[human_action]
                # if prob_action_given_theta_state == 0:
                #     prob_action_given_theta_state = 0.1
                # if self.mm_order == 'both' and depth == 2:
                # # prob_action_given_theta_state = np.exp(prob_action_given_theta_state)
                # if abs(prob_action_given_theta_state - max(restructured_weight_vector)) < 0.02:
                #     prob_action_given_theta_state = 0.8
                # else:
                #     prob_action_given_theta_state = 0.2
                #
                # if prob_action_given_theta_state < 0.1:
                #     prob_action_given_theta_state = 0.1
                # else:
                #     prob_action_given_theta_state = 0.1
                # print("prob_action_given_theta_state", prob_action_given_theta_state)
                posterior = (prob_theta * prob_action_given_theta_state) / (prob_action_given_state + epsilon)

                self.beliefs[mm_depth][human_model_params] = posterior
                total_weight += posterior

            # print("total_weight", total_weight)
            for human_model_params in self.beliefs[mm_depth]:
                self.beliefs[mm_depth][human_model_params] = self.beliefs[mm_depth][human_model_params] / total_weight

    def get_percent_particles_w_correct_prediction(self, human_state, human_action):
        total_weighted_accuracy = 0.0
        epsilon = 0.01
        # print("self.beliefs MAX values", max(self.beliefs.values()))

        human_model = self.get_max_likelihood_human_model()
        weighted_combination, depth, prob_theta = self.get_top_K_likelihood_human_model()

        weight_vector, robot_rew = human_model.get_proba_vector_given_state(human_state)

        weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
        weight_vector_normed_positive = [np.round(e / (sum(weight_vector_normed_positive)), 2) for e in
                                         weight_vector_normed_positive]

        restructured_weight_vector = []

        for color in COLOR_LIST:
            if human_state[color] > 0:
                if weight_vector_normed_positive[color] == 0:
                    restructured_weight_vector.append(0.01)
                else:
                    restructured_weight_vector.append(weight_vector_normed_positive[color])
            else:
                restructured_weight_vector.append(0)

        sum_weight_values = sum(restructured_weight_vector)
        for color in COLOR_LIST:
            restructured_weight_vector[color] = restructured_weight_vector[color] / (sum_weight_values + epsilon)

        prob_action_given_theta_state = restructured_weight_vector[human_action]

        if abs(prob_action_given_theta_state - max(restructured_weight_vector)) < 0.02:
            accuracy = 1.0
        else:
            accuracy = 0.0

        weighted_accuracy = accuracy * prob_theta
        total_weighted_accuracy = weighted_accuracy

        with open(self.log_filename, 'a') as f:
            f.write(f"\nRobot's weighted accuracy = {total_weighted_accuracy}")

        return total_weighted_accuracy

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
        sampled_particle_indices = np.random.choice(np.arange(len(population)), int(len(self.beliefs) * 0.75),
                                                    p=population_weights, replace=True)

        for i in sampled_particle_indices:
            x_noise = random.normalvariate(0, 0.05)
            y_noise = random.normalvariate(0, 0.05)
            z_noise = random.normalvariate(0, 0.05)
            w_noise = random.normalvariate(0, 0.05)
            x, y, z, w = population[i][0]
            depth = population[i][1]
            initial_human_model = self.param_to_model[population[i]]

            x, y, z, w = x + x_noise, y + y_noise, z + z_noise, w + w_noise

            weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))
            # weight_vector = (x, y, z, w)
            possible_weights[(weight_vector, depth)] = 1 / len(sampled_particle_indices)
            new_human_model = Human_Hypothesis(weight_vector, depth, self.num_particles)

            # r = np.random.uniform(0, 1)
            # if r < 0.5:
            new_human_model.set_beliefs(initial_human_model.beliefs)

            new_param_to_model[(weight_vector, depth)] = new_human_model

        self.beliefs = possible_weights
        self.param_to_model = new_param_to_model

    def update_with_partner_action(self, human_state, human_action, is_done):
        weighted_accuracy = self.get_percent_particles_w_correct_prediction(human_state, human_action)

        self.update_particle_weights(human_state, human_action)
        if is_done:
            self.pf_history.append(
                self.get_K_weighted_combination_belief(self.beliefs, weighted_accuracy, k=1, resampled=True))
        else:
            self.pf_history.append(
                self.get_K_weighted_combination_belief(self.beliefs, weighted_accuracy, k=1, resampled=False))

    def plot_weight_updates(self, savename, savename_accuracy):
        # TODO

        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []
        depth_over_time = []

        particle_accuracy_over_time = []
        resampled_over_time = []

        for (weighted_combination, weighted_depth, weighted_accuracy, resampled) in self.pf_history:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])
            depth_over_time.append(weighted_depth)
            particle_accuracy_over_time.append(weighted_accuracy)
            resampled_over_time.append(resampled)

        plt.figure()
        plt.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        plt.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        plt.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        plt.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')
        plt.plot(range(len(depth_over_time)), depth_over_time, color='m', label='depth')

        plt.xlabel("Timestep")
        plt.ylabel("Belief Weight and Depth")
        plt.legend()
        plt.title(f"Robot's Beliefs of Human Models: ")
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
        plt.title(f"PF Accuracy Whenever Human Acts")
        # print("savename_accuracy", savename_accuracy)
        plt.savefig(f"{savename_accuracy}")
        plt.close()
        plt.cla()
        plt.clf()

    def get_K_weighted_combination_belief(self, beliefs, weighted_accuracy, k=1, resampled=False):

        k = 1
        if self.mm_order == 'first':
            top_1 = dict(sorted(beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            # print("beliefs", beliefs)
            # print("list(top_1.keys())", list(top_1.keys()))
            top_param = list(top_1.keys())[0]
            top_prob = beliefs[1][top_param]
        elif self.mm_order == 'second':
            top_1 = dict(sorted(beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param = list(top_1.keys())[0]
            top_prob = beliefs[2][top_param]
        else:
            top_1_fo = dict(sorted(beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_fo = list(top_1_fo.keys())[0]
            top_param_fo_prob = beliefs[1][top_param_fo]

            top_1_so = dict(sorted(beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_so = list(top_1_so.keys())[0]
            top_param_so_prob = beliefs[2][top_param_so]

            if top_param_so_prob >= top_param_fo_prob:
                top_param = top_param_so
                top_prob = top_param_so_prob
            else:
                top_param = top_param_fo
                top_prob = top_param_fo_prob

        total_prob = top_prob
        if total_prob == 0:
            total_prob = 0.0001

        (weight_vector, depth) = top_param
        prob = 1
        weighted_combination = (np.array(list(weight_vector)) * prob)
        weighted_depth = ((depth - 1) * prob)
        weighted_combination = tuple(weighted_combination)

        return weighted_combination, weighted_depth, weighted_accuracy, resampled


    def update_human_models_with_robot_action(self, robot_state, robot_action, is_done):
        for mm_depth in list(self.param_to_model.keys()):
            for param in self.param_to_model[mm_depth]:
                self.param_to_model[mm_depth][param].update_particle_weights(robot_state, robot_action)
                if is_done:
                    self.param_to_model[mm_depth][param].resample_particles()

    def resample(self):
        self.resample_particles()
        for mm_depth in list(self.param_to_model.keys()):
            for param in self.param_to_model[mm_depth]:
                self.param_to_model[mm_depth][param].resample_particles()

    def get_max_likelihood_human_model(self):
        k = 1
        if self.mm_order == 'first':
            top_1 = dict(sorted(self.beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param = list(top_1.keys())[0]
            best_model = self.param_to_model[1][top_param]
        elif self.mm_order == 'second':
            top_1 = dict(sorted(self.beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param = list(top_1.keys())[0]
            best_model = self.param_to_model[2][top_param]
        else:
            top_1_fo = dict(sorted(self.beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_fo = list(top_1_fo.keys())[0]
            top_param_fo_prob = self.beliefs[1][top_param_fo]

            top_1_so = dict(sorted(self.beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_so = list(top_1_so.keys())[0]
            top_param_so_prob = self.beliefs[2][top_param_so]

            if top_param_so_prob >= top_param_fo_prob:
                top_param = top_param_so
                best_model = self.param_to_model[2][top_param]
            else:
                top_param = top_param_fo
                best_model = self.param_to_model[1][top_param]

        return best_model

    def get_top_K_likelihood_human_model(self, k=1):
        k = 1

        if self.mm_order == 'first':
            top_1 = dict(sorted(self.beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param = list(top_1.keys())[0]
            top_prob = self.beliefs[1][top_param]
        elif self.mm_order == 'second':
            top_1 = dict(sorted(self.beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param = list(top_1.keys())[0]
            top_prob = self.beliefs[2][top_param]
        else:
            top_1_fo = dict(sorted(self.beliefs[1].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_fo = list(top_1_fo.keys())[0]
            top_param_fo_prob = self.beliefs[1][top_param_fo]

            top_1_so = dict(sorted(self.beliefs[2].items(), key=operator.itemgetter(1), reverse=True)[:k])
            top_param_so = list(top_1_so.keys())[0]
            top_param_so_prob = self.beliefs[2][top_param_so]

            if top_param_so_prob >= top_param_fo_prob:
                top_param = top_param_so
                top_prob = top_param_so_prob
            else:
                top_param = top_param_fo
                top_prob = top_param_fo_prob

        total_prob = top_prob
        (weight_vector, depth) = top_param
        prob = 1
        weighted_combination = (np.array(list(weight_vector)) * prob)
        weighted_depth = ((depth - 1) * prob)
        weighted_combination = tuple(weighted_combination)

        with open(self.log_filename, 'a') as f:
            f.write(f"\nRobot's top human model = {(weighted_combination, weighted_depth, total_prob)}")

        return weighted_combination, weighted_depth, total_prob

    def act_every_timestep(self, input_state, robot_history, human_history):
        (human_pref, human_depth), human_model = self.get_max_likelihood_human_model()
        # (weighted_combination, weighted_depth, weighted_accuracy) = self.get_K_weighted_combination_belief(self.beliefs, 0, k=5)

        # self.himdp = HiMDP(human_pref, human_depth, self.vi_type, start_state=input_state, robot_rew=self.ind_rew)
        self.himdp = HiMDP(human_pref, human_depth, self.vi_type, start_state=input_state, robot_rew=self.ind_rew,
                           h_scalar=self.h_scalar)
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

    def get_humans_beliefs_of_robot(self, k=1):
        weighted_combination = np.array([0,0,0,0])
        _, weighted_depth, total_prob = self.get_top_K_likelihood_human_model()
        human_model = self.get_max_likelihood_human_model()

        prob = 1

        if weighted_depth == 2:
            inferred_robot_rew, confidence = human_model.get_K_weighted_combination_vector()
            weighted_combination = weighted_combination + (np.array(list(inferred_robot_rew)) * prob)

        weighted_combination = tuple(weighted_combination)
        return weighted_combination

    def act(self, input_state, robot_history, human_history, iteration):
        if iteration == 0:
            self.inferred_human_beliefs_of_robot.append(self.get_humans_beliefs_of_robot())
            human_pref, human_depth, total_prob = self.get_top_K_likelihood_human_model()
            # print("Robot COnfidence = ", total_prob)
            human_model = self.get_max_likelihood_human_model()

            if self.log_filename is not None:
                with open(self.log_filename, 'a') as f:
                    f.write(f"\nRobot's own rewards + human pref = {np.array(self.ind_rew) + np.array(human_pref)}")

            if self.log_filename is not None:
                with open(self.log_filename, 'a') as f:
                    f.write(f"\nRobot's confidence = {total_prob}")

            self.himdp = HiMDP(human_pref, human_depth, self.vi_type, start_state=input_state, robot_rew=self.ind_rew,
                               h_scalar=self.h_scalar,
                               confidence=total_prob)
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

        color_selected = self.himdp.idx_to_action[best_color]

        return color_selected

    def plot_beliefs_of_humans(self, savename, savename_accuracy):

        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []

        for weighted_combination in self.inferred_human_beliefs_of_robot:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])

        plt.figure()
        plt.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        plt.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        plt.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        plt.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')

        plt.xlabel("Round")
        plt.ylabel("Belief Weight and Depth")
        plt.legend()
        plt.title(f"Robot's Beliefs of Human's Belief about robot: ")
        plt.savefig(f"{savename}")
        plt.close()
        plt.cla()
        plt.clf()

    def plot_beliefs_to_axes(self, ax1, ax2, ax3):
        self.plot_beliefs_of_human_to_axes(ax1, ax2)
        self.plot_beliefs_of_human_beliefs_to_axes(ax3)

    def plot_beliefs_of_human_to_axes(self, ax1, ax2):
        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []
        depth_over_time = []

        particle_accuracy_over_time = []
        resampled_over_time = []

        for (weighted_combination, weighted_depth, weighted_accuracy, resampled) in self.pf_history:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])
            depth_over_time.append(weighted_depth)
            particle_accuracy_over_time.append(weighted_accuracy)
            resampled_over_time.append(resampled)

        ax1.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        ax1.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        ax1.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        ax1.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')
        ax1.plot(range(len(depth_over_time)), depth_over_time, color='m', label='depth')

        ax1.set(xlabel='Timestep', ylabel="Belief Weight and Depth")
        ax1.legend()
        ax1.title.set_text(f"Robot's Beliefs of Human Models:")

        ax2.plot(range(len(particle_accuracy_over_time)), particle_accuracy_over_time, color='blue', label='blue')
        # print("particle_accuracy_over_time", particle_accuracy_over_time)
        for i in range(len(resampled_over_time)):
            if resampled_over_time[i] is True:
                ax2.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='r')
            else:
                ax2.scatter(np.arange(len(particle_accuracy_over_time))[i], particle_accuracy_over_time[i], c='g')

        ax2.set(xlabel='Timestep', ylabel="Particle Weighted Accuracy")
        # ax2.legend()
        ax2.title.set_text(f"PF Accuracy Whenever Human Acts")

    def plot_beliefs_of_human_beliefs_to_axes(self, ax):
        weights_of_blue_over_time = []
        weights_of_green_over_time = []
        weights_of_red_over_time = []
        weights_of_yellow_over_time = []

        for weighted_combination in self.inferred_human_beliefs_of_robot:
            weights_of_blue_over_time.append(weighted_combination[0])
            weights_of_green_over_time.append(weighted_combination[1])
            weights_of_red_over_time.append(weighted_combination[2])
            weights_of_yellow_over_time.append(weighted_combination[3])

        ax.plot(range(len(weights_of_blue_over_time)), weights_of_blue_over_time, color='blue', label='blue')
        ax.plot(range(len(weights_of_green_over_time)), weights_of_green_over_time, color='green', label='green')
        ax.plot(range(len(weights_of_red_over_time)), weights_of_red_over_time, color='red', label='red')
        ax.plot(range(len(weights_of_yellow_over_time)), weights_of_yellow_over_time, color='yellow', label='yellow')

        ax.set(xlabel='Round', ylabel="Belief Weight and Depth")
        ax.legend()
        ax.title.set_text(f"Robot's Beliefs of Human's Belief about robot: ")




















