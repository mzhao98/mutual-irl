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


BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

import pickle
import json
import sys
import os



class Robot:
    def __init__(self, ind_rew, true_human_rew, starting_state, permutes, vi_type, is_collaborative_human):
        self.ind_rew = ind_rew
        self.true_human_rew = true_human_rew
        self.permutes = permutes

        self.beliefs = {}

        self.is_collaborative_human = is_collaborative_human
        self.total_reward = {'team': 0, 'robot': 0, 'human': 0}
        self.state_remaining_objects = {}
        self.possible_actions = [None]

        self.starting_objects = starting_state
        self.reset()
        self.possible_joint_actions = self.get_all_possible_joint_actions()

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
        self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.0001
        self.gamma = 1.0
        self.human_gamma = 0.01
        self.maxiter = 100
        self.beta = 15

        self.vi_type = vi_type

        self.true_human_rew_idx = None
        if self.true_human_rew is not None:
            self.set_beliefs_with_true_reward()
        else:
            self.set_beliefs_without_true_reward()

            print("Done setting up beliefs without true reward")

        self.history_of_human_beliefs = []

    def reset(self):
        self.state_remaining_objects = {}
        self.possible_actions = [None]
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.state_remaining_objects:
                self.state_remaining_objects[obj_tuple] = 1
                self.possible_actions.append(obj_tuple)
            else:
                self.state_remaining_objects[obj_tuple] += 1

    def set_beliefs_with_true_reward(self):
        # self.human_rew = copy.deepcopy(self.true_human_rew)

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.ind_rew.values())))
            permutes = list(set(permutes))
        # print("permutes = ", permutes)
        # print("len(permutes", len(permutes))
        object_keys = list(self.ind_rew.keys())
        for idx in range(len(permutes)):
            human_rew_values = permutes[idx]
            human_rew_dict = {object_keys[i]: list(human_rew_values)[i] for i in range(len(object_keys))}
            self.beliefs[idx] = {}
            self.beliefs[idx]['reward_dict'] = human_rew_dict
            self.beliefs[idx]['prob'] = 0
            if human_rew_dict == self.true_human_rew:
                # print("equals")
                self.beliefs[idx]['prob'] = 1
                self.true_human_rew_idx = idx
        # if self.true_human_rew not in [self.beliefs[idx]['reward_dict'] for idx in self.beliefs.keys()]:
        #     print("beliefs = ", self.beliefs)
        #     print("self.true_human_rew", self.true_human_rew)

        print("sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) = ", sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]))
        assert abs(sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) - 1) < 0.01

    def set_beliefs_without_true_reward(self):
        # self.human_rew = copy.deepcopy(self.true_human_rew)
        self.enumerate_states()

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.ind_rew.values())))
            permutes = list(set(permutes))

        object_keys = list(self.ind_rew.keys())
        self.belief_idx_to_q_values = {}
        for idx in range(len(permutes)):
            print("starting idx = ", idx)
            human_rew_values = permutes[idx]
            human_rew_dict = {object_keys[i]: list(human_rew_values)[i] for i in range(len(object_keys))}

            q_values_table = self.human_candidate_value_iteration(human_rew_dict)
            self.belief_idx_to_q_values[idx] = {}
            self.belief_idx_to_q_values[idx]['reward_dict'] = human_rew_dict
            self.belief_idx_to_q_values[idx] = q_values_table


            self.beliefs[idx] = {}
            self.beliefs[idx]['reward_dict'] = human_rew_dict
            self.beliefs[idx]['prob'] = 1/len(permutes)
            print("done with idx = ", idx)

        # assert sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) == 1

        # with open('data/q_values.pkl', 'wb') as file:
        #     pickle.dump(self.belief_idx_to_q_values, file)

    def reset_belief_history(self):
        self.history_of_human_beliefs = []

    def update_based_on_h_action(self, current_state, robot_action, human_action):

        # print("current_state, robot_action, human_action", (current_state, robot_action, human_action))

        if self.true_human_rew is not None:
            return

        self.history_of_human_beliefs.append(copy.deepcopy(self.beliefs))

        joint_action = (robot_action, human_action)
        joint_action_idx = self.action_to_idx[joint_action]

        current_state_tup = self.state_to_tuple(current_state)
        state_idx = self.state_to_idx[current_state_tup]

        # print("current_state_tup", current_state_tup)

        normalize_Z = 0

        dict_prob_obs_given_theta = {}

        for idx in self.beliefs:
            # human_rew_dict = self.beliefs[idx]['reward_dict']
            q_values_table = self.belief_idx_to_q_values[idx]

            # prob_theta = self.beliefs[idx]['prob']

            q_value_for_obs = q_values_table[state_idx, joint_action_idx]

            next_state, (team_rew, robot_rew, human_rew), done = self.human_step_given_state(
                current_state_remaining_objects, joint_action, human_rew_dict)
            r_sa = robot_rew + human_rew + team_rew

            exp_q_value_for_obs = np.exp(self.beta * q_value_for_obs)
            # exp_q_value_for_obs = q_value_for_obs

            # print()
            # print("belief", self.beliefs[idx]['reward_dict'])
            # print("q_value_for_obs", q_value_for_obs)
            # print("exp_q_value_for_obs", exp_q_value_for_obs)

            normalize_Z += exp_q_value_for_obs

            dict_prob_obs_given_theta[idx] = exp_q_value_for_obs

        if normalize_Z == 0:
            print("PROBLEM WITH Z=0 at dict_prob_obs_given_theta")
            normalize_Z = 0.01
        for idx in dict_prob_obs_given_theta:
            dict_prob_obs_given_theta[idx] = dict_prob_obs_given_theta[idx] / normalize_Z

        normalization_denominator = 0
        for idx in self.beliefs:
            prob_theta = self.beliefs[idx]['prob']
            prob_obs_given_theta_normalized = dict_prob_obs_given_theta[idx]
            self.beliefs[idx]['prob'] = prob_theta * prob_obs_given_theta_normalized
            normalization_denominator += prob_theta * prob_obs_given_theta_normalized

        if normalization_denominator == 0:
            print("PROBLEM WITH Z=0 at beliefs")
            normalization_denominator = 0.01
        for idx in self.beliefs:
            self.beliefs[idx]['prob'] = self.beliefs[idx]['prob'] / normalization_denominator

            # print(f"belief {self.beliefs[idx]['reward_dict']} likelihood is {self.beliefs[idx]['prob']}")


    def get_all_possible_joint_actions(self):
        possible_joint_actions = []
        for r_act in self.possible_actions:
            for h_act in self.possible_actions:
                joint_action = {'robot': r_act, 'human': h_act}
                possible_joint_actions.append(joint_action)
        return possible_joint_actions

    def is_done(self):
        if sum(self.state_remaining_objects.values()) == 0:
            return True
        return False

    def get_all_possible_joint_actions(self):
        possible_joint_actions = []
        for r_act in self.possible_actions:
            for h_act in self.possible_actions:
                joint_action = {'robot': r_act, 'human': h_act}
                possible_joint_actions.append(joint_action)
        return possible_joint_actions

    def greedy_step_given_state(self, input_state, joint_action, human_reward):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = -2
        # robot_rew_given_human, human_rew_given_robot = -1, -1
        if robot_action == human_action and human_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                if robot_action in state_remaining_objects:
                    if robot_action in state_remaining_objects and state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        # robot_rew += self.ind_rew[robot_action]
                        # human_rew += human_reward[robot_action]
                        team_rew += (self.ind_rew[robot_action])

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = 0, 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1

                        pickup_agent = np.random.choice(['r', 'h'])
                        if pickup_agent == 'r':
                            robot_rew += self.ind_rew[robot_action]
                            # human_rew = -1
                        else:
                            # robot_rew = -1
                            human_rew += human_reward[human_action]
                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.ind_rew[robot_action]
                        human_rew += human_reward[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.ind_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += human_reward[human_action]

        done = False
        if sum(state_remaining_objects.values()) == 0:
            done = True
        # team_rew = robot_rew + human_rew
        return state_remaining_objects, (team_rew, robot_rew, human_rew), done

    def step_given_state(self, input_state, joint_action, human_reward):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = -2
        # robot_rew_given_human, human_rew_given_robot = -1, -1
        if robot_action == human_action and human_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                if robot_action in state_remaining_objects:
                    if robot_action in state_remaining_objects and state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        # robot_rew += self.ind_rew[robot_action]
                        # human_rew += human_reward[robot_action]
                        team_rew += (self.ind_rew[robot_action] + human_reward[robot_action])

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = 0, 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1

                        pickup_agent = np.random.choice(['r', 'h'])
                        if pickup_agent == 'r':
                            robot_rew += self.ind_rew[robot_action]
                            # human_rew = -1
                        else:
                            # robot_rew = -1
                            human_rew += human_reward[human_action]
                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.ind_rew[robot_action]
                        human_rew += human_reward[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.ind_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += human_reward[human_action]

        done = False
        if sum(state_remaining_objects.values()) == 0:
            done = True
        # team_rew = robot_rew + human_rew
        return state_remaining_objects, (team_rew, robot_rew, human_rew), done

    def human_step_given_state(self, input_state, joint_action, human_reward):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = -2
        # robot_rew_given_human, human_rew_given_robot = -1, -1
        if robot_action == human_action and human_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                if robot_action in state_remaining_objects:
                    if robot_action in state_remaining_objects and state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        # robot_rew += self.ind_rew[robot_action]
                        # human_rew += human_reward[robot_action]
                        team_rew += (self.ind_rew[robot_action] + human_reward[robot_action])

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = 0, 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1

                        pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     robot_rew += self.ind_rew[robot_action]
                        #     # human_rew = -1
                        # else:
                        #     # robot_rew = -1
                        human_rew += human_reward[human_action]
                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.ind_rew[robot_action]
                        human_rew += human_reward[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.ind_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += human_reward[human_action]

        done = False
        if sum(state_remaining_objects.values()) == 0:
            done = True
        # team_rew = robot_rew + human_rew
        return state_remaining_objects, (team_rew, robot_rew, human_rew), done

    def human_imagine_step_given_state(self, input_state, joint_action, human_reward):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = -2
        # robot_rew_given_human, human_rew_given_robot = -1, -1
        if human_action is not None:
            if human_action in state_remaining_objects:
                if state_remaining_objects[human_action] > 0:
                    state_remaining_objects[human_action] -= 1
                    human_rew += human_reward[human_action]

        done = False
        if sum(state_remaining_objects.values()) == 0:
            done = True
        # team_rew = robot_rew + human_rew
        return state_remaining_objects, (team_rew, robot_rew, human_rew), done

    def state_to_tuple(self, remaining_objs_dict):
        remaining_objs = []
        for objname in remaining_objs_dict:
            for i in range(remaining_objs_dict[objname]):
                remaining_objs.append(objname)
        return tuple(remaining_objs)


    def enumerate_states(self):
        self.reset()

        actions = self.get_all_possible_joint_actions()
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state_remaining_objects)]

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(actions):

                next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(state, action, self.ind_rew)

                new_state_tup = self.state_to_tuple(next_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=team_rew, action=(action['robot'], action['human']))

        states = list(G.nodes)
        # print("NUMBER OF STATES", len(states))
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {(action['robot'], action['human']): i for i, action in enumerate(actions)}
        idx_to_action = {i: (action['robot'], action['human']) for i, action in enumerate(actions)}

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

    def get_human_action_under_greedy_hypothesis(self, current_state_remaining_objects, human_reward):
        best_human_act = []
        max_reward = -100
        for candidate_h_act in self.possible_actions:
            if candidate_h_act is not None:
                if current_state_remaining_objects[candidate_h_act] > 0:
                    candidate_rew = human_reward[candidate_h_act]
                    if candidate_rew == max_reward:
                        if candidate_h_act not in best_human_act:
                            best_human_act.append(candidate_h_act)

                    elif candidate_rew > max_reward:
                        max_reward = candidate_rew
                        best_human_act = [candidate_h_act]

        if len(best_human_act) == 0:
            h_action = None
        else:
            # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
            h_action = best_human_act[0]

        possible_h_action_to_prob = {}
        for candidate_h_act in self.possible_actions:
            if candidate_h_act == h_action:
                possible_h_action_to_prob[candidate_h_act] = 1
            else:
                possible_h_action_to_prob[candidate_h_act] = 0

        return possible_h_action_to_prob

    def get_human_action_under_collaborative_hypothesis(self, current_state_remaining_objects, human_reward):
        best_human_act = []
        max_reward = -100
        # print("self.possible_joint_actions", self.possible_joint_actions)
        for joint_act in self.possible_joint_actions:
            candidate_r_act = joint_act['robot']
            candidate_h_act = joint_act['human']
            # joint_act = {'robot': candidate_r_act, 'human': candidate_h_act}
            # print("joint_act", joint_act)
            # print("candidate_h_act", candidate_h_act)
            _, (team_r, robot_r, human_r), _ = self.human_step_given_state(current_state_remaining_objects, joint_act, human_reward)
            # print("candidate_rew", candidate_rew)
            candidate_rew = team_r + robot_r + human_r
            if candidate_h_act is not None:
                if candidate_rew == max_reward:
                    if candidate_h_act not in best_human_act:
                        best_human_act.append(candidate_h_act)

                elif candidate_rew > max_reward:
                    max_reward = candidate_rew
                    best_human_act = [candidate_h_act]

        if len(best_human_act) == 0:
            h_action = None
        else:
            # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
            h_action = best_human_act[0]

        possible_h_action_to_prob = {}
        for candidate_h_act in self.possible_actions:
            if candidate_h_act == h_action:
                possible_h_action_to_prob[candidate_h_act] = 1
            else:
                possible_h_action_to_prob[candidate_h_act] = 0

        return possible_h_action_to_prob

    def get_human_action_under_hypothesis(self, current_state_remaining_objects, human_reward):
        if self.is_collaborative_human is True:
            possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(current_state_remaining_objects, human_reward)
        else:
            possible_h_action_to_prob = self.get_human_action_under_greedy_hypothesis(current_state_remaining_objects, human_reward)
        return possible_h_action_to_prob



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
            # print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                current_state = copy.deepcopy(list(self.idx_to_state[s]))

                current_state_remaining_objects = {}
                for obj_tuple in current_state:
                    if obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 1
                    else:
                        current_state_remaining_objects[obj_tuple] += 1
                for obj_tuple in self.possible_actions:
                    if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 0

                if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                    for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                        Q[s, action_idx] = vf[s]

                else:
                    # compute new Q values
                    # for action_idx in range(n_actions):

                    for action_idx in self.idx_to_action:
                        # pdb.set_trace()
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        r_act = joint_action[0]
                        h_act = joint_action[1]
                        joint_action = {'robot': r_act, 'human': h_act}

                        expected_reward_sa = 0
                        robot_only_reward = 0
                        belief_added_reward = 0
                        for h_reward_idx in self.beliefs:
                            h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                            probability_of_hyp = self.beliefs[h_reward_idx]['prob']
                            probability_of_hyp = 1 / (1 + np.exp(-60 * (probability_of_hyp - 0.5)))
                            # if probability_of_hyp == 0:
                            #     continue

                            possible_h_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects, h_reward_hypothesis)
                            h_prob = possible_h_action_to_prob[h_act]
                            h_prob = 1 / (1 + np.exp(-60 * (h_prob - 0.5)))
                            # if h_prob == 0:
                            #     continue

                            next_state, (team_rew, robot_rew, human_rew), done = \
                                self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)
                            #
                            # print()
                            # print("current_state_remaining_objects", current_state_remaining_objects)
                            # print("joint_action", joint_action)
                            # print(f"h_reward_hypothesis = {h_reward_hypothesis}")
                            # print("probability_of_hyp", probability_of_hyp)
                            # print("h_prob", h_prob)
                            # print(f"team_rew = {team_rew}")
                            # print(f"robot_rew = {robot_rew}")
                            # print(f"human_rew = {human_rew}")


                            # r_sa = team_rew + robot_rew + human_rew
                            #
                            # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                            # # r_sa += (self.gamma * vf[s11])
                            # expected_reward_sa += (r_sa * probability_of_hyp * h_prob)
                            expected_reward_sa += (robot_rew * probability_of_hyp) + ((team_rew + human_rew) * probability_of_hyp * h_prob)
                            robot_only_reward += (robot_rew * probability_of_hyp)
                            belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * h_prob)
                            # expected_reward_sa += r_sa
                            # break

                        # if expected_reward_sa == 0:
                        #     expected_reward_sa = -2
                        # next_state, (_, _, _), done = \
                        #     self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                        # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # print()
                        # print("current_state_remaining_objects", current_state_remaining_objects)
                        # print("joint_action", joint_action)
                        # print(f"robot_only_reward = {robot_only_reward}")
                        # print(f"belief_added_reward = {belief_added_reward}")

                        next_state, (_, _, _), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # if expected_reward_sa == 0:
                        #     expected_reward_sa = -2
                        #     expected_reward_sa += (self.gamma * vf[s11])
                        expected_reward_sa += (self.gamma * vf[s11])
                        Q[s, action_idx] = expected_reward_sa

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            # print("i = ", i)
            if delta < self.epsilson:
                print("CVI DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            current_state = copy.deepcopy(list(self.idx_to_state[s]))

            current_state_remaining_objects = {}
            for obj_tuple in current_state:
                if obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 1
                else:
                    current_state_remaining_objects[obj_tuple] += 1
            for obj_tuple in self.possible_actions:
                if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 0

            if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

            else:
                for action_idx in self.idx_to_action:
                    # pdb.set_trace()
                    # check joint action
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    robot_only_reward = 0
                    belief_added_reward = 0

                    expected_reward_sa = 0
                    for h_reward_idx in self.beliefs:
                        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                        probability_of_hyp = self.beliefs[h_reward_idx]['prob']
                        probability_of_hyp = 1 / (1 + np.exp(-60 * (probability_of_hyp - 0.5)))
                        # if probability_of_hyp == 0:
                        #     continue

                        possible_h_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects, h_reward_hypothesis)
                        h_prob = possible_h_action_to_prob[h_act]
                        h_prob = 1/(1+np.exp(-60*(h_prob - 0.5)))
                        # if h_prob == 0:
                        #     continue

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        # r_sa = team_rew + robot_rew + human_rew
                        #
                        # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # r_sa += (self.gamma * vf[s11])
                        # expected_reward_sa += (r_sa * probability_of_hyp * h_prob)
                        expected_reward_sa += (robot_rew * probability_of_hyp) + ((team_rew + human_rew) * probability_of_hyp * h_prob)
                        robot_only_reward +=  (robot_rew * probability_of_hyp)
                        belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * h_prob)
                        # expected_reward_sa += r_sa
                        # break

                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2




                    next_state, (_, _, _), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2
                    expected_reward_sa += (self.gamma * vf[s11])
                    Q[s, action_idx] = expected_reward_sa

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]


        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def collective_value_iteration_with_true_human_reward(self):
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
            # print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                current_state = copy.deepcopy(list(self.idx_to_state[s]))

                current_state_remaining_objects = {}
                for obj_tuple in current_state:
                    if obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 1
                    else:
                        current_state_remaining_objects[obj_tuple] += 1
                for obj_tuple in self.possible_actions:
                    if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 0

                if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                    for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                        Q[s, action_idx] = vf[s]

                else:
                    # compute new Q values

                    h_reward_hypothesis = self.beliefs[self.true_human_rew_idx]['reward_dict']
                    probability_of_hyp = self.beliefs[self.true_human_rew_idx]['prob']

                    possible_h_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects,
                                                                                       h_reward_hypothesis)

                    for action_idx in self.idx_to_action:
                        # pdb.set_trace()
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        r_act = joint_action[0]
                        h_act = joint_action[1]
                        joint_action = {'robot': r_act, 'human': h_act}

                        h_prob = possible_h_action_to_prob[h_act]

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        r_sa = team_rew + robot_rew + human_rew
                        # print("r_sa = ", r_sa)
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        expected_reward_sa = (probability_of_hyp * h_prob) * (r_sa + (self.gamma * vf[s11]))

                        # if expected_reward_sa == 0:
                        #     expected_reward_sa = -2
                        Q[s, action_idx] = expected_reward_sa

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            # print("i = ", i)
            if delta < self.epsilson:
                print("CVI DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            current_state = copy.deepcopy(list(self.idx_to_state[s]))

            current_state_remaining_objects = {}
            for obj_tuple in current_state:
                if obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 1
                else:
                    current_state_remaining_objects[obj_tuple] += 1
            for obj_tuple in self.possible_actions:
                if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 0

            if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

            else:
                h_reward_hypothesis = self.beliefs[self.true_human_rew_idx]['reward_dict']
                probability_of_hyp = self.beliefs[self.true_human_rew_idx]['prob']

                possible_h_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects,
                                                                                   h_reward_hypothesis)

                for action_idx in self.idx_to_action:
                    # pdb.set_trace()
                    # check joint action
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    h_prob = possible_h_action_to_prob[h_act]

                    next_state, (team_rew, robot_rew, human_rew), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                    r_sa = team_rew + robot_rew + human_rew
                    # print("r_sa = ", r_sa)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    expected_reward_sa = (probability_of_hyp * h_prob) * (r_sa + (self.gamma * vf[s11]))

                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2
                    Q[s, action_idx] = expected_reward_sa

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]


        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def greedy_value_iteration(self):
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

        zeroed_human_reward = copy.deepcopy(self.ind_rew)
        for keyname in zeroed_human_reward:
            zeroed_human_reward[keyname] = 0

        for i in range(self.maxiter):
            # print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                current_state = copy.deepcopy(list(self.idx_to_state[s]))

                current_state_remaining_objects = {}
                for obj_tuple in current_state:
                    if obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 1
                    else:
                        current_state_remaining_objects[obj_tuple] += 1
                for obj_tuple in self.possible_actions:
                    if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 0

                if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                    for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                        Q[s, action_idx] = vf[s]

                else:
                    # compute new Q values
                    #
                    for action_idx in self.idx_to_action:
                        # pdb.set_trace()
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        r_act = joint_action[0]
                        h_act = joint_action[1]
                        joint_action = {'robot': r_act, 'human': h_act}


                        # print("current_state_remaining_objects = ", current_state_remaining_objects)
                        # print("joint_action = ", joint_action)
                        next_state, (team_rew, robot_rew, human_rew), done = self.greedy_step_given_state(
                            current_state_remaining_objects, joint_action, zeroed_human_reward)
                        # print(f"current_state = ", current_state_remaining_objects)
                        # print("action=  ", joint_action)
                        # print("r_sa = ", r_sa)
                        # print("next_state = ", next_state)
                        # print("done = ", done)



                        # if joint_action['human'] == h_action:
                        #     print(f"h_action,joint_action['human'] = {(h_action, joint_action['human'])}")
                        #     r_sa = team_rew
                        # else:
                        #     r_sa = -1

                        r_sa = robot_rew + team_rew
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # action_idx = self.action_to_idx[(r_act, h_action)]
                        Q[s, action_idx] = r_sa + (self.gamma * vf[s11])

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            if delta < self.epsilson:
                print("Std VI DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            current_state = copy.deepcopy(list(self.idx_to_state[s]))

            current_state_remaining_objects = {}
            for obj_tuple in current_state:
                if obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 1
                else:
                    current_state_remaining_objects[obj_tuple] += 1
            for obj_tuple in self.possible_actions:
                if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 0

            if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

            else:

                # compute new Q values
                for action_idx in self.idx_to_action:
                    # check joint action
                    # joint_action = self.idx_to_action[action_idx]
                    # joint_action = {'robot': r_act, 'human': h_action}
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    next_state, (team_rew, robot_rew, human_rew), done = self.greedy_step_given_state(current_state_remaining_objects,
                                                                                           joint_action, zeroed_human_reward)

                    r_sa = robot_rew + team_rew
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # action_idx = self.action_to_idx[(r_act, h_action)]
                    Q[s, action_idx] = r_sa + (self.gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]


        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def human_candidate_value_iteration(self, human_rew_dict):
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
            # print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                current_state = copy.deepcopy(list(self.idx_to_state[s]))

                current_state_remaining_objects = {}
                for obj_tuple in current_state:
                    if obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 1
                    else:
                        current_state_remaining_objects[obj_tuple] += 1
                for obj_tuple in self.possible_actions:
                    if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 0

                if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                    for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                        Q[s, action_idx] = vf[s]

                else:
                    # compute new Q values
                    #
                    for action_idx in self.idx_to_action:
                        # pdb.set_trace()
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        r_act = joint_action[0]
                        h_act = joint_action[1]
                        joint_action = {'robot': r_act, 'human': h_act}


                        # print("current_state_remaining_objects = ", current_state_remaining_objects)
                        # print("joint_action = ", joint_action)
                        if self.is_collaborative_human is True:
                            next_state, (team_rew, robot_rew, human_rew), done = self.human_step_given_state(
                                current_state_remaining_objects, joint_action, human_rew_dict)
                            r_sa = robot_rew + human_rew + team_rew
                        else:
                            next_state, (team_rew, robot_rew, human_rew), done = self.human_imagine_step_given_state(
                            current_state_remaining_objects, joint_action, human_rew_dict)
                            r_sa = human_rew


                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # action_idx = self.action_to_idx[(r_act, h_action)]
                        Q[s, action_idx] = r_sa + (self.human_gamma * vf[s11])

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            if delta < self.epsilson:
                print("Std VI DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            current_state = copy.deepcopy(list(self.idx_to_state[s]))

            current_state_remaining_objects = {}
            for obj_tuple in current_state:
                if obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 1
                else:
                    current_state_remaining_objects[obj_tuple] += 1
            for obj_tuple in self.possible_actions:
                if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                    current_state_remaining_objects[obj_tuple] = 0

            if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                for action_idx in range(n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

            else:

                # compute new Q values
                for action_idx in self.idx_to_action:
                    # check joint action
                    # joint_action = self.idx_to_action[action_idx]
                    # joint_action = {'robot': r_act, 'human': h_action}
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    if self.is_collaborative_human is True:
                        next_state, (team_rew, robot_rew, human_rew), done = self.human_step_given_state(
                            current_state_remaining_objects, joint_action, human_rew_dict)
                        r_sa = robot_rew + human_rew + team_rew
                    else:
                        next_state, (team_rew, robot_rew, human_rew), done = self.human_imagine_step_given_state(
                            current_state_remaining_objects, joint_action, human_rew_dict)
                        r_sa = human_rew
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # action_idx = self.action_to_idx[(r_act, h_action)]
                    Q[s, action_idx] = r_sa + (self.human_gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        return Q

    def setup_value_iteration(self):
        self.enumerate_states()

        if self.vi_type == 'cvi':
            if self.true_human_rew is None:
                # print("Running collective_value_iteration")
                self.collective_value_iteration()
            else:
                # print("Running collective_value_iteration_with_true_human_reward")
                # self.collective_value_iteration()
                self.collective_value_iteration_with_true_human_reward()
        else:
            self.greedy_value_iteration()
        return


    def act(self, state):
        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        action_distribution = self.policy[state_idx]

        # for a_idx in self.idx_to_action:
        #     print(f"action {self.idx_to_action[a_idx]} --> value = {action_distribution[a_idx]}")

        # print("action_distribution", action_distribution)
        action = np.argmax(action_distribution)
        action = self.idx_to_action[action]

        # print("idx_to_action = ", self.idx_to_action)
        # print("action_distribution = ", action_distribution)
        # print("action", action)

        r_action = action[0]
        return r_action


    # def resolve_heavy_pickup(self, rh_action):
    #     robot_rew, human_rew = -1, -1
    #     if rh_action in self.state_remaining_objects and self.state_remaining_objects[rh_action] > 0:
    #         self.state_remaining_objects[rh_action] -= 1
    #         robot_rew = self.ind_rew[rh_action]
    #         human_rew = self.human_rew[rh_action]
    #
    #     self.total_reward['team'] += (robot_rew + human_rew)
    #     self.total_reward['robot'] += robot_rew
    #     self.total_reward['human'] += human_rew
    #     return (robot_rew + human_rew), robot_rew, human_rew
    #
    # def resolve_two_agents_same_item(self, robot_action, human_action):
    #     (robot_action_color, robot_action_weight) = robot_action
    #     (human_action_color, human_action_weight) = human_action
    #     robot_rew, human_rew = -1, -1
    #     if robot_action in self.state_remaining_objects:
    #         if self.state_remaining_objects[robot_action] == 0:
    #             robot_rew, human_rew = -1, -1
    #         elif self.state_remaining_objects[robot_action] == 1:
    #             self.state_remaining_objects[robot_action] -= 1
    #             robot_rew = self.ind_rew[robot_action]
    #             human_rew = self.human_rew[human_action]
    #             pickup_agent = np.random.choice(['r', 'h'])
    #             if pickup_agent == 'r':
    #                 human_rew = -1
    #             else:
    #                 robot_rew = -1
    #         else:
    #             self.state_remaining_objects[robot_action] -= 1
    #             self.state_remaining_objects[human_action] -= 1
    #             robot_rew = self.ind_rew[robot_action]
    #             human_rew = self.human_rew[human_action]
    #
    #     self.total_reward['team'] += (robot_rew + human_rew)
    #     self.total_reward['robot'] += robot_rew
    #     self.total_reward['human'] += human_rew
    #
    #     return (robot_rew + human_rew), robot_rew, human_rew
    #
    # def resolve_two_agents_diff_item(self, robot_action, human_action):
    #
    #     robot_rew, human_rew = -1, -1
    #
    #     if robot_action is not None and robot_action in self.state_remaining_objects:
    #         (robot_action_color, robot_action_weight) = robot_action
    #
    #         if robot_action_weight == 0:
    #             if self.state_remaining_objects[robot_action] > 0:
    #                 self.state_remaining_objects[robot_action] -= 1
    #                 robot_rew = self.ind_rew[robot_action]
    #
    #     if human_action is not None and human_action in self.state_remaining_objects:
    #         (human_action_color, human_action_weight) = human_action
    #         if human_action_weight == 0:
    #             if self.state_remaining_objects[human_action] > 0:
    #                 self.state_remaining_objects[human_action] -= 1
    #                 human_rew = self.human_rew[human_action]
    #
    #     self.total_reward['team'] += (robot_rew + human_rew)
    #     self.total_reward['robot'] += robot_rew
    #     self.total_reward['human'] += human_rew
    #     return (robot_rew + human_rew), robot_rew, human_rew
