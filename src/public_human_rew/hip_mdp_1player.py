import copy

import numpy as np
import operator
import random
import pdb

import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from human_agent import Human_Hypothesis

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3

# COLOR_LIST = [BLUE, GREEN, RED, YELLOW]


class HiMDP:

    def __init__(self, start_state, all_colors_list, task_reward, human_rew, h_rho, humans_beliefs, true_robot_rew, human_believed_robot_rew, r_rho, vi_type):
    # def __init__(self, start_state, all_colors_list, task_reward, human_rew, h_rho, true_robot_rew, r_rho, vi_type):
        # World parameters
        self.start_state = start_state
        self.all_colors_list = all_colors_list
        self.task_reward = task_reward
        self.NOP = len(all_colors_list)

        # Human model parameters
        self.human_rew = human_rew
        self.h_rho = h_rho

        # Robot model parameters
        self.vi_type = vi_type
        self.true_robot_rew = true_robot_rew

        self.human_believed_robot_rew = None
        self.humans_beliefs = None
        self.human_believed_robot_rew = human_believed_robot_rew
        self.humans_beliefs = humans_beliefs

        self.r_rho = r_rho
        # self.human_model_of_robot = None

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
            self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.0001
        self.gamma = 1.0
        self.maxiter = 10000

        # reset environment
        self.state = None
        self.reset()

    def set_robot_rew(self, robot_rew):
        self.true_robot_rew = robot_rew

    def set_human_rew(self, human_rew):
        self.human_rew = human_rew

    # def create_human_model_from_scratch(self):
    #     self.human_model_of_robot = Human_Hypothesis(self.human_rew, self.h_rho)
    #
    # def set_human_model_from_existing(self, human_model_of_robot):
    #     self.human_model_of_robot = copy.deepcopy(human_model_of_robot)

    def reset(self):
        self.initialize_game()

    def initialize_game(self):
        self.state = copy.deepcopy(self.start_state)

    def state_to_tuple(self, state):
        # return tuple([item for sublist in state for item in sublist])
        return tuple(state)

    def set_to_state(self, state):
        self.state = list(state)

    def get_available_actions_from_state(self, state):
        available = [self.NOP]
        for color in self.all_colors_list:
            if state[color] > 0:
                available.append(color)
        return available

    def is_done(self):
        return sum(self.state) == 0

    def is_done_given_state(self, state):
        return sum(state) == 0

    def robot_step_given_state(self, current_state, robot_action):
        # robot_action is a color
        next_state = copy.deepcopy(list(current_state))

        if robot_action == self.NOP:
            return next_state, 0, self.is_done_given_state(next_state)

        # have the robot act
        # update state
        rew = 0
        if next_state[robot_action] > 0:
            next_state[robot_action] -= 1
            rew += self.true_robot_rew[robot_action]

        return next_state, rew, self.is_done_given_state(next_state)

    def simulate_human_model_update_and_act(self, robot_state, robot_action, current_state):
        # state = [# blue, # green, # red, # yellow]
        # action = color
        new_beliefs_of_human = copy.deepcopy(self.humans_beliefs)
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
        for robot_model_params in new_beliefs_of_human:
            (weight_vector, _) = robot_model_params

            prob_theta = new_beliefs_of_human[robot_model_params]

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

            new_beliefs_of_human[robot_model_params] = posterior
            total_weight += posterior

        # Normalize all beliefs
        for robot_model_params in new_beliefs_of_human:
            new_beliefs_of_human[robot_model_params] = new_beliefs_of_human[robot_model_params] / total_weight

        new_beliefs_of_human = max(new_beliefs_of_human.items(), key=operator.itemgetter(1))[0][0]
        human_action = None
        max_reward = -10000
        candidate_actions = []

        for color in self.all_colors_list:
            rew = 0
            # hypothetical state update
            next_state = copy.deepcopy(current_state)

            if current_state[color] > 0:
                next_state[color] -= 1
                rew += self.human_rew[color]

                if self.h_rho > 0:
                    max_robot_rew = -10000
                    robot_color = None
                    for r_color in self.all_colors_list:
                        if next_state[r_color] > 0:
                            r_rew = new_beliefs_of_human[r_color]
                            if r_rew > max_robot_rew:
                                max_robot_rew = r_rew
                                robot_color = r_color

                    if robot_color is not None:
                        rew += (self.h_rho * new_beliefs_of_human[robot_color])

                if rew == max_reward:
                    max_reward = rew
                    # human_action = color
                    candidate_actions.append(color)

                elif rew > max_reward:
                    max_reward = rew
                    # human_action = color
                    candidate_actions = [color]

        if sum(current_state) > 0:
            human_action = np.random.choice(candidate_actions)
            human_action = candidate_actions[0]
        else:
            human_action = None
        return human_action

    def simulate_human_model_act(self, current_state):
        human_action = None
        max_reward = -10000
        candidate_actions = []

        for color in self.all_colors_list:
            rew = 0
            # hypothetical state update
            next_state = copy.deepcopy(current_state)

            if current_state[color] > 0:
                next_state[color] -= 1
                rew += self.human_rew[color]

                if self.h_rho > 0:
                    max_robot_rew = -10000
                    robot_color = None
                    for r_color in self.all_colors_list:
                        if next_state[r_color] > 0:
                            r_rew = self.human_believed_robot_rew[r_color]
                            if r_rew > max_robot_rew:
                                max_robot_rew = r_rew
                                robot_color = r_color

                    if robot_color is not None:
                        rew += (self.h_rho * self.human_believed_robot_rew[robot_color])

                if rew == max_reward:
                    max_reward = rew
                    # human_action = color
                    candidate_actions.append(color)

                elif rew > max_reward:
                    max_reward = rew
                    # human_action = color
                    candidate_actions = [color]

        if sum(current_state) > 0:
            human_action = np.random.choice(candidate_actions)
            human_action = candidate_actions[0]
        else:
            human_action = None
        return human_action

    def enumerate_states(self):
        self.reset()

        actions = copy.deepcopy(self.all_colors_list)
        actions.append(self.NOP)
        # actions = []
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state)]

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get available actions
            available_robot_actions = self.get_available_actions_from_state(state)

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_robot_actions):

                next_state, rew, done = self.robot_step_given_state(state, action)

                new_state_tup = self.state_to_tuple(next_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=rew, action=action)

        states = list(G.nodes)
        # print("NUMBER OF STATES", len(states))
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        for i in range(len(states)):
            # get all outgoing edges from current state
            edges = G.out_edges(states[i], data=True)
            for edge in edges:
                # get index of action in action_idx
                action_idx_i = action_to_idx[edge[2]['action']]
                # get index of next state in node list
                next_state_i = states.index(edge[1])
                # add edge to transition matrix
                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = edge[2]['weight']

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        # for i in range(len(states)):
        #     for j in range(len(actions)):
        #         print("np.sum(transition_mat[i, :, j])", np.sum(transition_mat[i, :, j]))
        #         print("np.sum(transition_mat[i, :, j]", np.sum(transition_mat[i, :, j]))
                # assert np.isclose(np.sum(transition_mat[i, :, j]), 1.0) or np.isclose(np.sum(transition_mat[i, :, j]),
                #                                                                       0.0)
        self.transitions, self.rewards, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx


    def value_iteration(self):
        if self.vi_type == 'cvi':
            vf, pi = self.collective_value_iteration()
        else:
            vf, pi = self.standard_value_iteration()
        return vf, pi

    # implementation of tabular value iteration
    def collective_value_iteration(self):
        """
        Parameters
        ----------
            transitions : array_like
                Transition probability matrix. Of size (# states, # states, # actions).
            rewards : array_like
                Reward matrix. Of size (# states, # actions).
            epsilson : float, optional
                The convergence threshold. The default is 0.0001.
            gamma : float, optional
                The discount factor. The default is 0.99.
            maxiter : int, optional
                The maximum number of iterations. The default is 10000.
        Returns
        -------
            value_function : array_like
                The value function. Of size (# states, 1).
            pi : array_like
                The optimal policy. Of size (# states, 1).
        """
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))

        for i in range(self.maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                # compute new Q values

                # Consider partner rew
                initial_state = copy.deepcopy(list(self.idx_to_state[s]))
                for action_idx in range(n_actions):
                    # have the robot act

                    current_state = copy.deepcopy(list(self.idx_to_state[s]))
                    robot_state = copy.deepcopy(current_state)
                    robot_action = self.idx_to_action[action_idx]

                    if robot_action == self.NOP:
                        if sum((list(self.idx_to_state[s]))) == 0:
                            r_sa = 0
                            r_s1aH = 0
                        else:
                            r_sa = -10
                            r_s1aH = -10
                    else:
                        # get robot reward
                        r_sa = (self.rewards[s][action_idx] if current_state[robot_action] > 0 else -10)
                        r_sa_check = (self.true_robot_rew[action_idx] if current_state[robot_action] > 0 else -10)
                        assert r_sa == r_sa_check

                        # get expected human reward
                        r_s1aH = 0
                        # update state
                        if current_state[robot_action] > 0:
                            current_state[robot_action] -= 1

                            # Comment in if you want to update human model's model of robot based on last robot action
                            # human_action = self.simulate_human_model_update_and_act(robot_state, robot_action, current_state)

                            # Comment in if you dont want to update human model's model of robot based on last robot action
                            human_action = self.simulate_human_model_act(current_state)

                            # update state and human's model of robot
                            if human_action is not None and current_state[human_action] > 0:
                                r_s1aH += self.human_rew[human_action]
                                current_state[human_action] -= 1

                    s11 = self.state_to_idx[tuple(current_state)]
                    joint_reward = r_sa + r_s1aH
                    Q[s, action_idx] = joint_reward + (self.gamma * vf[s11])

                vf[s] = np.max(Q[s,:], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))


            # check for convergence
            if delta < self.epsilson:
                # print("DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            # compute new Q values

            # Consider partner rew
            initial_state = copy.deepcopy(list(self.idx_to_state[s]))
            for action_idx in range(n_actions):
                # have the robot act

                current_state = copy.deepcopy(list(self.idx_to_state[s]))
                robot_action = self.idx_to_action[action_idx]

                if robot_action == self.NOP:
                    if sum((list(self.idx_to_state[s]))) == 0:
                        r_sa = 0
                        r_s1aH = 0
                    else:
                        r_sa = -10
                        r_s1aH = -10
                else:
                    # get robot reward
                    r_sa = (self.rewards[s][action_idx] if current_state[robot_action] > 0 else -10)
                    r_sa_check = (self.true_robot_rew[action_idx] if current_state[robot_action] > 0 else -10)
                    assert r_sa == r_sa_check

                    # get expected human reward
                    r_s1aH = 0
                    # update state
                    if current_state[robot_action] > 0:
                        current_state[robot_action] -= 1

                        # Comment in if you want to update human model's model of robot based on last robot action
                        # copy_human_model = copy.deepcopy(self.human_model_of_robot)
                        # copy_human_model.update_with_partner_action(initial_state, robot_action, False)
                        # human_action = copy_human_model.act(current_state, [], [])

                        # Comment in if you dont want to update human model's model of robot based on last robot action
                        human_action = self.simulate_human_model_act(current_state)

                        # update state and human's model of robot
                        if human_action is not None and current_state[human_action] > 0:
                            r_s1aH += self.human_rew[human_action]
                            current_state[human_action] -= 1

                s11 = self.state_to_idx[tuple(current_state)]
                joint_reward = r_sa + r_s1aH
                Q[s, action_idx] = joint_reward + (self.gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def standard_value_iteration(self):
        """
        Parameters
        ----------
            transitions : array_like
                Transition probability matrix. Of size (# states, # states, # actions).
            rewards : array_like
                Reward matrix. Of size (# states, # actions).
            epsilson : float, optional
                The convergence threshold. The default is 0.0001.
            gamma : float, optional
                The discount factor. The default is 0.99.
            maxiter : int, optional
                The maximum number of iterations. The default is 10000.
        Returns
        -------
            value_function : array_like
                The value function. Of size (# states, 1).
            pi : array_like
                The optimal policy. Of size (# states, 1).
        """
        self.gamma = 0.99
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))

        for i in range(self.maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                # compute new Q values

                # Consider partner rew
                initial_state = copy.deepcopy(list(self.idx_to_state[s]))
                for action_idx in range(n_actions):
                    # have the robot act

                    current_state = copy.deepcopy(list(self.idx_to_state[s]))
                    robot_action = self.idx_to_action[action_idx]

                    if robot_action == self.NOP:
                        if sum((list(self.idx_to_state[s]))) == 0:
                            r_sa = 0
                            r_s1aH = 0
                        else:
                            r_sa = -10
                            r_s1aH = -10
                    else:
                        # get robot reward
                        r_sa = (self.rewards[s][action_idx] if current_state[robot_action] > 0 else -10)
                        r_sa_check = (self.true_robot_rew[action_idx] if current_state[robot_action] > 0 else -10)
                        assert r_sa == r_sa_check

                        # get expected human reward
                        r_s1aH = 0
                        # update state
                        if current_state[robot_action] > 0:
                            current_state[robot_action] -= 1


                    s11 = self.state_to_idx[tuple(current_state)]
                    joint_reward = r_sa + r_s1aH
                    Q[s, action_idx] = joint_reward + (self.gamma * vf[s11])

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            if delta < self.epsilson:
                # print("DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            # compute new Q values

            # Consider partner rew
            initial_state = copy.deepcopy(list(self.idx_to_state[s]))
            for action_idx in range(n_actions):
                # have the robot act

                current_state = copy.deepcopy(list(self.idx_to_state[s]))
                robot_action = self.idx_to_action[action_idx]

                if robot_action == self.NOP:
                    if sum((list(self.idx_to_state[s]))) == 0:
                        r_sa = 0
                        r_s1aH = 0
                    else:
                        r_sa = -10
                        r_s1aH = -10
                else:
                    # get robot reward
                    r_sa = (self.rewards[s][action_idx] if current_state[robot_action] > 0 else -10)
                    r_sa_check = (self.true_robot_rew[action_idx] if current_state[robot_action] > 0 else -10)
                    assert r_sa == r_sa_check

                    # get expected human reward
                    r_s1aH = 0
                    # update state
                    if current_state[robot_action] > 0:
                        current_state[robot_action] -= 1

                s11 = self.state_to_idx[tuple(current_state)]
                joint_reward = r_sa + r_s1aH
                Q[s, action_idx] = joint_reward + (self.gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

