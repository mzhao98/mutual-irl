import pdb

import numpy as np
import operator
import copy
import networkx as nx
import random
# import matplotlib.pyplot as plt
import itertools
from scipy import stats
from multiprocessing import Pool, freeze_support
from maxent import irl

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

import pickle
import json
import sys
import os
from datetime import datetime
from scipy.stats import entropy


class Robot:
    def __init__(self, team_rew, ind_rew, human_rew, starting_state, robot_knows_human_rew, permutes, vi_type,
                 is_collaborative_human, update_threshold=0.9):
        self.team_rew = team_rew
        self.ind_rew = ind_rew
        self.true_human_rew = human_rew
        self.permutes = permutes
        self.robot_knows_human_rew = robot_knows_human_rew

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
        self.gamma = 0.9
        self.greedy_gamma = 0.9
        self.human_gamma = 0.0001
        self.maxiter = 100
        self.small_maxiter = 10
        self.beta = 0.9
        self.confidence_threshold = 0.6
        self.update_threshold = update_threshold

        self.vi_type = vi_type

        self.true_human_rew_idx = None
        if self.robot_knows_human_rew is True:
            # print("Start setting up beliefs WITH true reward")
            self.set_beliefs_with_true_reward()
            # print("Done setting up beliefs WITH true reward")
        else:
            # print("Start setting up beliefs WITHOUT true reward")
            self.set_beliefs_without_true_reward()
            # print("Done setting up beliefs WITHOUT true reward")

        # print("self.beliefs = ", self.beliefs)

        self.history_of_human_beliefs = []
        self.history_of_robot_beliefs_of_true_human_rew = []
        self.history_of_robot_beliefs_of_max_human_rew = []
        self.episode_history = []
        self.believed_human_reward_dict = {elem: 0 for elem in self.start_remaining_objects.keys()}
        self.feature_matrix = None
        self.believed_human_reward = [0,0,0,0]


    def reset(self):
        self.state_remaining_objects = {}
        self.possible_actions = [None]
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.state_remaining_objects:
                self.state_remaining_objects[obj_tuple] = 1
                self.possible_actions.append(obj_tuple)
            else:
                self.state_remaining_objects[obj_tuple] += 1

        self.start_remaining_objects = copy.deepcopy(self.state_remaining_objects)

    def set_beliefs_with_true_reward(self):
        # self.human_rew = copy.deepcopy(self.true_human_rew)

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.true_human_rew.values())))
            permutes = list(set(permutes))
        # print("permutes = ", permutes)
        # print("len(permutes", len(permutes))
        object_keys = list(self.true_human_rew.keys())
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

        # print("sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) = ", sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]))
        assert abs(sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) - 1) < 0.01

    def set_beliefs_without_true_reward(self):
        # self.human_rew = copy.deepcopy(self.true_human_rew)
        self.enumerate_states()

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.true_human_rew.values())))
            permutes = list(set(permutes))

        object_keys = list(self.true_human_rew.keys())
        self.belief_idx_to_q_values = {}
        for idx in range(len(permutes)):
            # print("starting idx = ", idx)
            human_rew_values = permutes[idx]
            human_rew_dict = {object_keys[i]: list(human_rew_values)[i] for i in range(len(object_keys))}

            # q_values_table = self.human_candidate_value_iteration(human_rew_dict)
            self.belief_idx_to_q_values[idx] = {}
            self.belief_idx_to_q_values[idx]['reward_dict'] = human_rew_dict
            self.belief_idx_to_q_values[idx] = None

            self.beliefs[idx] = {}
            self.beliefs[idx]['reward_dict'] = human_rew_dict
            self.beliefs[idx]['prob'] = 1 / len(permutes)
            # print("done with idx = ", idx)

            if human_rew_dict == self.true_human_rew:
                self.true_human_rew_idx = idx

        # print("sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) = ", sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]))
        assert abs(sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) - 1) < 0.01

        # with open('data/q_values.pkl', 'wb') as file:
        #     pickle.dump(self.belief_idx_to_q_values, file)

    def reset_belief_history(self):
        self.history_of_human_beliefs = []
        self.history_of_robot_beliefs_of_true_human_rew = []
        self.history_of_robot_beliefs_of_max_human_rew = []
        self.episode_history = []

    def hypothesize_updated_belief(self, current_state, current_human_action, current_robot_action, next_state, next_human_action):
        # state, robot_action, human_action, next_state, human_action_next_state
        #  Create trajectories matrix
        trajectories = []
        for state, robot_action, human_action in self.episode_history:
            joint_action_idx = self.action_to_idx[(robot_action, human_action)]
            state_idx = self.state_to_idx[self.state_to_tuple(state)]
            trajectories.append([state_idx, joint_action_idx])

        current_state_idx = self.state_to_idx[self.state_to_tuple(current_state)]
        joint_action_idx = self.action_to_idx[(current_robot_action, current_human_action)]
        trajectories.append([current_state_idx, joint_action_idx])

        next_state_idx = self.state_to_idx[self.state_to_tuple(next_state)]
        next_joint_action_idx = self.action_to_idx[(None, next_human_action)]
        trajectories.append([next_state_idx, next_joint_action_idx])

        trajectories = np.array([trajectories])
        # print("trajectories", trajectories.shape)

        # create feature matrix
        if self.feature_matrix is None:
            feature_matrix = []
            for state in self.state_to_idx:
                state_as_dict = {}
                for object_type in self.start_remaining_objects:
                    state_as_dict[object_type] = 0
                for object_type in list(state):
                    state_as_dict[object_type] += 1

                featurized_state = []
                for object_type in self.start_remaining_objects:
                    featurized_state.append(self.start_remaining_objects[object_type] - state_as_dict[object_type])
                feature_matrix.append(featurized_state)
            self.feature_matrix = np.array(feature_matrix)
        # print("feature_matrix", feature_matrix.shape)

        transition_probability = self.transitions
        discount = self.gamma
        n_actions = len(self.action_to_idx)
        epochs = 10
        learning_rate = 0.01
        _,  updated_believed_human_reward = irl(self.feature_matrix, n_actions, discount, transition_probability,
                                            trajectories, epochs, learning_rate)

        return updated_believed_human_reward

    def update_based_on_h_action(self, current_state, robot_action, human_action):
        # print("current_state, robot_action, human_action", (current_state, robot_action, human_action))
        self.episode_history.append((current_state, robot_action, human_action))


    def run_max_ent(self):
        if self.robot_knows_human_rew is True:
            return

        self.history_of_human_beliefs.append(copy.deepcopy(self.beliefs))
        self.history_of_robot_beliefs_of_true_human_rew.append(self.beliefs[self.true_human_rew_idx]['prob'])

        """
            Find the reward function for the given trajectories.

            feature_matrix: Matrix with the nth row representing the nth state. NumPy
                array with shape (N, D) where N is the number of states and D is the
                dimensionality of the state.
            n_actions: Number of actions A. int.
            discount: Discount factor of the MDP. float.
            transition_probability: NumPy array mapping (state_i, action, state_k) to
                the probability of transitioning from state_i to state_k under action.
                Shape (N, A, N).
            trajectories: 3D array of state/action pairs. States are ints, actions
                are ints. NumPy array with shape (T, L, 2) where T is the number of
                trajectories and L is the trajectory length.
            epochs: Number of gradient descent steps. int.
            learning_rate: Gradient descent learning rate. float.
            -> Reward vector with shape (N,).
            """

        #  Create trajectories matrix
        trajectories = []
        for state, robot_action, human_action in self.episode_history:
            joint_action_idx = self.action_to_idx[(robot_action, human_action)]
            state_idx = self.state_to_idx[self.state_to_tuple(state)]
            trajectories.append([state_idx, joint_action_idx])
        trajectories = np.array([trajectories])
        # print("trajectories", trajectories.shape)

        # create feature matrix
        if self.feature_matrix is None:
            feature_matrix = []
            for state in self.state_to_idx:
                state_as_dict = {}
                for object_type in self.start_remaining_objects:
                    state_as_dict[object_type] = 0
                for object_type in list(state):
                    state_as_dict[object_type] += 1

                featurized_state = []
                for object_type in self.start_remaining_objects:
                    featurized_state.append(self.start_remaining_objects[object_type] - state_as_dict[object_type])
                feature_matrix.append(featurized_state)
            self.feature_matrix = np.array(feature_matrix)
        # print("feature_matrix", feature_matrix.shape)


        transition_probability = self.transitions
        discount = self.gamma
        n_actions = len(self.action_to_idx)
        epochs = 10
        learning_rate = 0.01
        _, self.believed_human_reward = irl(self.feature_matrix, n_actions, discount, transition_probability,
            trajectories, epochs, learning_rate)

        # print("self.believed_human_reward", self.believed_human_reward)

        # pdb.set_trace()
        self.believed_human_reward_dict = {}
        for i in range(len(self.believed_human_reward)):
            self.believed_human_reward_dict[list(self.start_remaining_objects.keys())[i]] = self.believed_human_reward[i] * 100

        # print("self.believed_human_reward", self.believed_human_reward_dict)
        # pdb.set_trace()

        return
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
        team_rew = self.team_rew[robot_action] + self.team_rew[human_action]

        # if robot_action is None or state_remaining_objects[robot_action] == 0:
        #     team_rew -= 2
        # if human_action is None or state_remaining_objects[human_action] == 0:
        #     team_rew -= 2

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
                        # team_rew += (self.ind_rew[robot_action] + human_reward[robot_action])
                        human_rew += human_reward[human_action]
                        robot_rew += self.ind_rew[robot_action]

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew += 0
                        human_rew += 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1
                        human_rew += human_reward[human_action]
                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     robot_rew += self.ind_rew[robot_action]
                        #     # human_rew = -1
                        # else:
                        #     # robot_rew = -1
                        #     human_rew += human_reward[human_action]
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

    def step_given_state_w_human_hyp_for_all(self, input_state, joint_action, human_reward):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = 0

        # if robot_action is None or state_remaining_objects[robot_action] == 0:
        #     team_rew -= 2
        # if human_action is None or state_remaining_objects[human_action] == 0:
        #     team_rew -= 2

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
                        # team_rew += (self.ind_rew[robot_action] + human_reward[robot_action])
                        human_rew += human_reward[human_action]
                        robot_rew += human_reward[robot_action]

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew += 0
                        human_rew += 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1
                        human_rew += human_reward[human_action]
                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     robot_rew += human_reward[robot_action]
                        #     # human_rew = -1
                        # else:
                        #     # robot_rew = -1
                        #     human_rew += human_reward[human_action]
                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += human_reward[robot_action]
                        human_rew += human_reward[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += human_reward[robot_action]

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
        team_rew = self.team_rew[robot_action] + self.team_rew[human_action]
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
                        human_rew += human_reward[human_action]
                        robot_rew += self.ind_rew[robot_action]

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew += 0
                        human_rew += 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1
                        human_rew += human_reward[human_action]
                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     robot_rew += self.ind_rew[robot_action]
                        #     # human_rew = -1
                        # else:
                        #     # robot_rew = -1
                        #     human_rew += human_reward[human_action]
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
        team_rew = self.team_rew[robot_action] + self.team_rew[human_action]
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
        transition_mat = np.zeros([len(states), len(actions), len(states)])
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
                transition_mat[i, action_idx_i, next_state_i] = 1.0
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
        h_action = None
        max_reward = -100
        for candidate_h_act in self.possible_actions:
            candidate_rew = -2
            if candidate_h_act is not None:
                if current_state_remaining_objects[candidate_h_act] > 0:
                    # print("human_reward = ", human_reward)
                    # print("candidate_h_act", candidate_h_act)
                    candidate_rew = human_reward[candidate_h_act]

            if candidate_rew > max_reward:
                max_reward = candidate_rew
                h_action = [candidate_h_act]

        possible_h_action_to_prob = {}
        for candidate_h_act in self.possible_actions:
            if candidate_h_act == h_action:
                possible_h_action_to_prob[candidate_h_act] = 1
            else:
                possible_h_action_to_prob[candidate_h_act] = 0

        return possible_h_action_to_prob

    def get_human_action_under_collaborative_hypothesis(self, current_state_remaining_objects, human_reward):
        # state_remaining_objects = copy.deepcopy(current_state_remaining_objects)
        best_human_act = []
        max_reward = -100
        h_action = None
        human_r = 0
        robot_r = 0
        best_joint_action = None
        # print("self.possible_joint_actions", self.possible_joint_actions)

        # for joint_act in self.possible_joint_actions:
        #     candidate_r_act = joint_act['robot']
        #     candidate_h_act = joint_act['human']
        #
        #     _, (team_rew, r_rew, h_rew), _ = self.step_given_state(state_remaining_objects, joint_act, human_reward)
        #     candidate_rew = team_rew + r_rew + h_rew
        #     if candidate_h_act is not None:
        #         # if candidate_rew == max_reward:
        #         #     if candidate_h_act not in best_human_act:
        #         #         best_human_act.append(candidate_h_act)
        #         #         best_human_act =
        #
        #         # elif candidate_rew > max_reward:
        #         #     max_reward = candidate_rew
        #         #     best_human_act = [candidate_h_act]
        #         #
        #         # else:
        #         #     other_actions.append(candidate_h_act)
        #         if candidate_rew > max_reward:
        #             h_action = candidate_h_act
        #             max_reward = candidate_rew

        best_reward_for_human_action = {}

        for joint_act in self.possible_joint_actions:
            state_remaining_objects = copy.deepcopy(current_state_remaining_objects)
            human_r = 0
            robot_r = 0
            candidate_r_act = joint_act['robot']
            candidate_h_act = joint_act['human']
            joint_act = {'robot': candidate_r_act, 'human': candidate_h_act}
            # print("joint_act", joint_act)
            # print("candidate_h_act", candidate_h_act)
            # _, (team_r, robot_r, human_r), _ = self.human_step_given_state(current_state_remaining_objects, joint_act, human_reward)
            if candidate_h_act is not None:
                if candidate_h_act[1] == 1 and candidate_h_act == candidate_r_act and state_remaining_objects[
                    candidate_h_act] > 0:
                    state_remaining_objects[candidate_h_act] -= 1
                    human_r = human_reward[candidate_h_act]
                    robot_r = self.ind_rew[candidate_r_act]
                    # print(f"1: {human_r}, {robot_r}")

                if candidate_h_act[1] == 0 and state_remaining_objects[candidate_h_act] > 0:
                    state_remaining_objects[candidate_h_act] -= 1
                    human_r = human_reward[candidate_h_act]
                    # print(f"2: {human_r}, {robot_r}")

            if candidate_r_act is not None:
                if candidate_r_act[1] == 0 and state_remaining_objects[candidate_r_act] > 0:
                    state_remaining_objects[candidate_r_act] -= 1
                    robot_r = self.ind_rew[candidate_r_act]
                    # print(f"3:  {robot_r}")

            # print(f"1: {human_r}, {robot_r}")
            team_r = -2
            candidate_rew = team_r + robot_r + human_r
            # print("candidate_rew", candidate_rew)
            # pdb.set_trace()
            if candidate_h_act is not None:
                # if candidate_rew == max_reward:
                #     if candidate_h_act not in best_human_act:
                #         best_human_act.append(candidate_h_act)

                if candidate_rew > max_reward:
                    max_reward = candidate_rew
                    h_action = candidate_h_act
                    best_joint_action = (candidate_r_act, candidate_h_act)

                if candidate_h_act not in best_reward_for_human_action:
                    best_reward_for_human_action[candidate_h_act] = candidate_rew
                else:
                    if candidate_rew > best_reward_for_human_action[candidate_h_act]:
                        best_reward_for_human_action[candidate_h_act] = candidate_rew

        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]

        possible_h_action_to_prob = {}
        for candidate_h_act in self.possible_actions:
            if candidate_h_act == h_action:
                possible_h_action_to_prob[candidate_h_act] = 1
            elif candidate_h_act is None:
                possible_h_action_to_prob[candidate_h_act] = 1
            elif best_reward_for_human_action[candidate_h_act] == max_reward:
                possible_h_action_to_prob[candidate_h_act] = 1
            else:
                possible_h_action_to_prob[candidate_h_act] = 0

        # pdb.set_trace()
        # possible_j_action_to_prob = {}
        # for joint_act in self.possible_joint_actions:
        #     candidate_r_act = joint_act['robot']
        #     candidate_h_act = joint_act['human']
        #     if best_joint_action == (candidate_r_act, candidate_h_act):
        #         possible_j_action_to_prob[(candidate_r_act, candidate_h_act)] = 1
        #     else:
        #         possible_j_action_to_prob[(candidate_r_act, candidate_h_act)] = 0

        return possible_h_action_to_prob

    def get_human_action_under_hypothesis(self, current_state_remaining_objects, human_reward):
        # print("human_reward", human_reward)
        if self.is_collaborative_human is True:
            possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(
                current_state_remaining_objects, human_reward)
        else:
            possible_h_action_to_prob = self.get_human_action_under_greedy_hypothesis(current_state_remaining_objects,
                                                                                      human_reward)
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
        self.n_states = self.transitions.shape[0]
        self.n_actions = self.transitions.shape[1]

        # initialize value function
        pi = np.zeros((self.n_states, 1))
        vf = np.zeros((self.n_states, 1))
        Q = np.zeros((self.n_states, self.n_actions))
        # print("true belief prob = ", self.beliefs[self.true_human_rew_idx]['prob'])

        for i in range(self.maxiter):
            # print("i=", i)
            # initalize delta
            delta = 0
            previous_pi = copy.deepcopy(pi)
            # perform Bellman update
            for s in range(self.n_states):
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
                    for action_idx in range(self.n_actions):
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

                            possible_joint_action_to_prob = self.get_human_action_under_hypothesis(
                                current_state_remaining_objects, h_reward_hypothesis)
                            j_prob = possible_joint_action_to_prob[h_act]
                            if r_act is None:
                                j_prob = 1
                            # j_prob = 1

                            next_state, (team_rew, robot_rew, human_rew), done = \
                                self.step_given_state(current_state_remaining_objects, joint_action,
                                                      h_reward_hypothesis)

                            # if r_act is None:
                            #     team_rew -= 2
                            # if h_act is None:
                            #     team_rew -= 2
                            # #
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

                            # if r_act is None:
                            #     robot_rew -= 1
                            # elif current_state_remaining_objects[r_act] == 0:
                            #     robot_rew -= 1
                            # if h_act is None:
                            #     robot_rew -= 1
                            # elif current_state_remaining_objects[r_act] == 0:
                            #     robot_rew -= 1

                            expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp * j_prob)

                            # if r_act is None:
                            #     expected_reward_sa -= 100
                            # elif current_state_remaining_objects[r_act] == 0:
                            #     expected_reward_sa -= 100

                            robot_only_reward += (robot_rew * probability_of_hyp)
                            belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * j_prob)
                            # expected_reward_sa
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

                pi[s] = np.argmax(Q[s, :], 0)

                # compute delta
                if pi[s] != previous_pi[s]:
                    delta += 1

            # check for convergence
            # print("i = ", i)
            if delta < self.epsilson:
                # print("CVI DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(self.n_states):
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
                for action_idx in range(self.n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

            else:
                # print("\ncurrent_state", current_state)
                sum_of_probs = 0
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
                    # print("SUM OF PROBS", sum([self.beliefs[h_idx]['prob'] for h_idx in self.beliefs]))
                    # pdb.set_trace()

                    for h_reward_idx in self.beliefs:
                        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                        probability_of_hyp = self.beliefs[h_reward_idx]['prob']
                        # probability_of_hyp = 1 / (1 + np.exp(-60 * (probability_of_hyp - self.confidence_threshold)))
                        # if probability_of_hyp == 0:
                        #     continue

                        possible_joint_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects, h_reward_hypothesis)
                        j_prob = possible_joint_action_to_prob[h_act]

                        if r_act is None:
                            j_prob = 1
                        # j_prob = 1
                        # h_prob = 1/(1+np.exp(-60*(h_prob - self.confidence_threshold)))
                        # if h_prob == 0:
                        #     continue
                        sum_of_probs += probability_of_hyp * j_prob

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        # if r_act is None:
                        #     team_rew -= 2
                        # if h_act is None:
                        #     team_rew -= 2

                        # r_sa = team_rew + robot_rew + human_rew
                        #
                        # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # r_sa += (self.gamma * vf[s11])
                        # expected_reward_sa += (r_sa * probability_of_hyp * h_prob)
                        # expected_reward_sa += (robot_rew * probability_of_hyp) + ((team_rew + human_rew) * probability_of_hyp * h_prob)
                        # if r_act is None:
                        #     robot_rew -= 100
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     robot_rew -= 100

                        expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp * j_prob)

                        # if r_act is None:
                        #     expected_reward_sa -= 100
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     expected_reward_sa -= 100

                        robot_only_reward += (robot_rew * probability_of_hyp)
                        belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * j_prob)
                        # expected_reward_sa += r_sa
                        # break

                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2

                    next_state, (_, _, _), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2

                    # print(f"joint_action {joint_action}: expected = {expected_reward_sa}, future = {(self.gamma * vf[s11])}")
                    expected_reward_sa += (self.gamma * vf[s11])
                    Q[s, action_idx] = expected_reward_sa
                    # print(f"\t Q = {Q[s, action_idx]}")

                # print("sum_of_probs = ", sum_of_probs)
            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def collective_policy_iteration(self):
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
        self.n_states = self.transitions.shape[0]
        self.n_actions = self.transitions.shape[1]

        # initialize value function
        self.pi = np.zeros((self.n_states, 1))
        self.vf = np.zeros((self.n_states, 1))
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = {}
        # print("true belief prob = ", self.beliefs[self.true_human_rew_idx]['prob'])

        # self.collective_policy_improvement()

        for i in range(self.small_maxiter):
            v, p = self.collective_policy_evaluation()
            policy_stable = self.collective_policy_improvement()
            # print(f"CPI ITERATION {i}")
            if policy_stable is True:
                break

    def collective_policy_evaluation(self):
        for i in range(self.small_maxiter):
            # print("CPE i=", i)
            # print(f"beliefs = {len(self.beliefs)}")
            # print(f"actions = {len(self.idx_to_action)}")
            # print(f"state = {self.n_states}")
            # initalize delta
            delta = 0

            # perform Bellman update
            for s in range(self.n_states):
                # print(f"s = {s}")
                # store old value function
                start_time = datetime.now()
                old_v = self.vf[s].copy()

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
                    for action_idx in range(self.n_actions):
                        # action_idx = self.action_to_idx[(None, None)]
                        self.Q[s, action_idx] = self.vf[s]

                else:
                    # compute new Q values
                    # for action_idx in range(n_actions):

                    for action_idx in self.idx_to_action:
                        # print(f"action_idx = {action_idx}")
                        # pdb.set_trace()
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        r_act = joint_action[0]
                        h_act = joint_action[1]
                        joint_action = {'robot': r_act, 'human': h_act}

                        expected_reward_sa = 0

                        for h_reward_idx in self.beliefs:
                            # print(f"h_reward_idx = {h_reward_idx} of total # beliefs {len(self.beliefs)}")
                            h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                            probability_of_hyp = self.beliefs[h_reward_idx]['prob']
                            # print(f"probability_of_hyp = {probability_of_hyp}, h_reward_idx = {h_reward_idx}")

                            # possible_joint_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects, h_reward_hypothesis)
                            # j_prob = possible_joint_action_to_prob[h_act]
                            # if r_act is None:
                            #     j_prob = 1
                            # j_prob = possible_joint_action_to_prob[(r_act, h_act)]
                            # h_prob = 1

                            next_state, (team_rew, robot_rew, human_rew), done = \
                                self.step_given_state(current_state_remaining_objects, joint_action,
                                                      h_reward_hypothesis)

                            expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp)

                        next_state, (_, _, _), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]

                        expected_reward_sa += (self.gamma * self.vf[s11])
                        self.Q[s, action_idx] = expected_reward_sa

                self.vf[s] = np.max(self.Q[s, :], 0)

                delta = max(delta, abs(old_v - self.vf[s]))

                end_time = datetime.now()

            # check for convergence
            # print("i completed = ", i)

            # print("time for 1 state iter", end_time - start_time)
            # print("delta = ", delta)
            if delta < self.epsilson:
                # print("CPE DONE at iteration ", i)
                break

        return self.vf, self.pi

    def collective_policy_improvement(self):
        policy_stable = True

        # compute optimal policy
        # policy = {}
        wait_for_policy_creation = False
        if len(self.policy) == 0:
            wait_for_policy_creation = True

        for s in range(self.n_states):
            if not wait_for_policy_creation:
                old_policy_at_s = copy.deepcopy(self.policy[s])
            # store old value function
            # old_v = self.vf[s].copy()

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
                for action_idx in range(self.n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    self.Q[s, action_idx] = self.vf[s]

            else:
                # print("\ncurrent_state", current_state)
                sum_of_probs = 0
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
                    # print("SUM OF PROBS", sum([self.beliefs[h_idx]['prob'] for h_idx in self.beliefs]))
                    # pdb.set_trace()

                    for h_reward_idx in self.beliefs:
                        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                        probability_of_hyp = self.beliefs[h_reward_idx]['prob']

                        # possible_joint_action_to_prob = self.get_human_action_under_hypothesis(
                        #     current_state_remaining_objects, h_reward_hypothesis)
                        # j_prob = possible_joint_action_to_prob[h_act]
                        #
                        # if r_act is None:
                        #     j_prob = 1
                        # j_prob = possible_joint_action_to_prob[(r_act, h_act)]
                        # j_prob = 1
                        # h_prob = 1/(1+np.exp(-60*(h_prob - self.confidence_threshold)))
                        # if h_prob == 0:
                        #     continue
                        # sum_of_probs += probability_of_hyp * j_prob

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp)

                        # if r_act is None:
                        #     expected_reward_sa -= 0
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     expected_reward_sa -= 100

                    next_state, (_, _, _), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2

                    # print(f"joint_action {joint_action}: expected = {expected_reward_sa}, future = {(self.gamma * self.vf[s11])}")
                    expected_reward_sa += (self.gamma * self.vf[s11])
                    self.Q[s, action_idx] = expected_reward_sa
                    # print(f"\t Q = {self.Q[s, action_idx]}")

                # print("sum_of_probs = ", sum_of_probs)
            self.pi[s] = np.argmax(self.Q[s, :], 0)
            self.policy[s] = self.Q[s, :]
            if wait_for_policy_creation:
                policy_stable = False
            elif old_policy_at_s.all() != self.policy[s].all():
                policy_stable = False

        return policy_stable

    def collective_policy_iteration_w_h_alpha(self, h_alpha):
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
        self.n_states = self.transitions.shape[0]
        self.n_actions = self.transitions.shape[1]

        # initialize value function
        self.pi = np.zeros((self.n_states, 1))
        self.vf = np.zeros((self.n_states, 1))
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = {}
        # print("true belief prob = ", self.beliefs[self.true_human_rew_idx]['prob'])

        # self.collective_policy_improvement()

        for i in range(self.small_maxiter):
            v, p = self.collective_policy_evaluation_w_h_alpha(h_alpha)
            policy_stable = self.collective_policy_improvement_w_h_alpha(h_alpha)
            # print(f"CPI ITERATION {i}")
            if policy_stable is True:
                break

    def collective_policy_evaluation_w_h_alpha(self, h_alpha):
        for i in range(self.small_maxiter):
            print("CPE i=", i)
            # print(f"beliefs = {len(self.beliefs)}")
            # print(f"actions = {len(self.idx_to_action)}")
            # print(f"state = {self.n_states}")
            # initalize delta
            delta = 0

            # perform Bellman update
            for s in range(self.n_states):
                # print(f"s = {s}")
                # store old value function
                start_time = datetime.now()
                old_v = self.vf[s].copy()

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
                    for action_idx in range(self.n_actions):
                        # action_idx = self.action_to_idx[(None, None)]
                        self.Q[s, action_idx] = self.vf[s]

                else:
                    # compute new Q values
                    # for action_idx in range(n_actions):

                    for action_idx in self.idx_to_action:
                        # print(f"action_idx = {action_idx}")
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
                            # print(f"h_reward_idx = {h_reward_idx} of total # beliefs {len(self.beliefs)}")
                            h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                            probability_of_hyp = self.beliefs[h_reward_idx]['prob']

                            possible_joint_action_to_prob = self.get_human_action_under_hypothesis(
                                current_state_remaining_objects, h_reward_hypothesis)
                            j_prob = possible_joint_action_to_prob[h_act]
                            # if r_act is None:
                            #     j_prob = 1
                            # j_prob = possible_joint_action_to_prob[(r_act, h_act)]
                            # j_prob = 1
                            if j_prob == 1:
                                j_prob = 1 - h_alpha
                            else:
                                j_prob = h_alpha

                            next_state, (team_rew, robot_rew, human_rew), done = \
                                self.step_given_state(current_state_remaining_objects, joint_action,
                                                      h_reward_hypothesis)

                            # if r_act is None:
                            #     team_rew -= 2

                            # if h_act is None:
                            #     team_rew -= 2
                            # #
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

                            # if r_act is None:
                            #     robot_rew -= 1
                            # elif current_state_remaining_objects[r_act] == 0:
                            #     robot_rew -= 1
                            # if h_act is None:
                            #     robot_rew -= 1
                            # elif current_state_remaining_objects[r_act] == 0:
                            #     robot_rew -= 1

                            expected_reward_sa += team_rew + ((robot_rew + human_rew) * probability_of_hyp * j_prob)

                            if r_act is None:
                                expected_reward_sa -= 0
                            elif current_state_remaining_objects[r_act] == 0:
                                expected_reward_sa -= 100

                            robot_only_reward += (robot_rew * probability_of_hyp)
                            belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * j_prob)
                            # expected_reward_sa
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
                        expected_reward_sa += (self.gamma * self.vf[s11])
                        self.Q[s, action_idx] = expected_reward_sa

                self.vf[s] = np.max(self.Q[s, :], 0)

                delta = max(delta, abs(old_v - self.vf[s]))

                end_time = datetime.now()

            # check for convergence
            # print("i completed = ", i)

            # print("time for 1 state iter", end_time - start_time)
            # print("delta = ", delta)
            if delta < self.epsilson:
                print("CPE DONE at iteration ", i)
                break

        return self.vf, self.pi

    def collective_policy_improvement_w_h_alpha(self, h_alpha):
        policy_stable = True

        # compute optimal policy
        # policy = {}
        wait_for_policy_creation = False
        if len(self.policy) == 0:
            wait_for_policy_creation = True

        for s in range(self.n_states):
            if not wait_for_policy_creation:
                old_policy_at_s = copy.deepcopy(self.policy[s])
            # store old value function
            # old_v = self.vf[s].copy()

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
                for action_idx in range(self.n_actions):
                    # action_idx = self.action_to_idx[(None, None)]
                    self.Q[s, action_idx] = self.vf[s]

            else:
                # print("\ncurrent_state", current_state)
                sum_of_probs = 0
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
                    # print("SUM OF PROBS", sum([self.beliefs[h_idx]['prob'] for h_idx in self.beliefs]))
                    # pdb.set_trace()

                    for h_reward_idx in self.beliefs:
                        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                        probability_of_hyp = self.beliefs[h_reward_idx]['prob']
                        # probability_of_hyp = 1 / (1 + np.exp(-60 * (probability_of_hyp - self.confidence_threshold)))
                        # if probability_of_hyp == 0:
                        #     continue

                        possible_joint_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects, h_reward_hypothesis)
                        j_prob = possible_joint_action_to_prob[h_act]
                        #
                        # if r_act is None:
                        #     j_prob = 1
                        # j_prob = possible_joint_action_to_prob[(r_act, h_act)]
                        # j_prob = 1
                        if j_prob == 1:
                            j_prob = 1 - h_alpha
                        else:
                            j_prob = h_alpha
                        # h_prob = 1/(1+np.exp(-60*(h_prob - self.confidence_threshold)))
                        # if h_prob == 0:
                        #     continue
                        sum_of_probs += probability_of_hyp * j_prob

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        # if r_act is None:
                        #     team_rew -= 2
                        # if h_act is None:
                        #     team_rew -= 2

                        # r_sa = team_rew + robot_rew + human_rew
                        #
                        # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # r_sa += (self.gamma * vf[s11])
                        # expected_reward_sa += (r_sa * probability_of_hyp * h_prob)
                        # expected_reward_sa += (robot_rew * probability_of_hyp) + ((team_rew + human_rew) * probability_of_hyp * h_prob)
                        # if r_act is None:
                        #     robot_rew -= 100
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     robot_rew -= 100

                        # expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp * j_prob)
                        expected_reward_sa += team_rew + ((robot_rew + human_rew) * probability_of_hyp * j_prob)

                        if r_act is None:
                            expected_reward_sa -= 0
                        elif current_state_remaining_objects[r_act] == 0:
                            expected_reward_sa -= 100

                        robot_only_reward += (robot_rew * probability_of_hyp)
                        belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * j_prob)
                        # expected_reward_sa += r_sa
                        # break

                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2

                    next_state, (_, _, _), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, self.ind_rew)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2

                    # print(f"joint_action {joint_action}: expected = {expected_reward_sa}, future = {(self.gamma * self.vf[s11])}")
                    expected_reward_sa += (self.gamma * self.vf[s11])
                    self.Q[s, action_idx] = expected_reward_sa
                    # print(f"\t Q = {self.Q[s, action_idx]}")

                # print("sum_of_probs = ", sum_of_probs)
            self.pi[s] = np.argmax(self.Q[s, :], 0)
            self.policy[s] = self.Q[s, :]
            if wait_for_policy_creation:
                policy_stable = False
            elif old_policy_at_s.all() != self.policy[s].all():
                policy_stable = False

        return policy_stable

    def old_collective_value_iteration(self):
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
        n_actions = self.transitions.shape[1]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        # print("true belief prob = ", self.beliefs[self.true_human_rew_idx]['prob'])

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

                            possible_h_action_to_prob = self.get_human_action_under_hypothesis(
                                current_state_remaining_objects, h_reward_hypothesis)
                            h_prob = possible_h_action_to_prob[h_act]

                            next_state, (team_rew, robot_rew, human_rew), done = \
                                self.step_given_state(current_state_remaining_objects, joint_action,
                                                      h_reward_hypothesis)
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

                            if r_act is None:
                                robot_rew -= 100
                            elif current_state_remaining_objects[r_act] == 0:
                                robot_rew -= 100

                            expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp * h_prob)

                            robot_only_reward += (robot_rew * probability_of_hyp)
                            belief_added_reward += ((team_rew + human_rew) * probability_of_hyp * h_prob)
                            # expected_reward_sa
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
                # print("CVI DONE at iteration ", i)
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
                        # probability_of_hyp = 1 / (1 + np.exp(-60 * (probability_of_hyp - self.confidence_threshold)))
                        # if probability_of_hyp == 0:
                        #     continue

                        possible_h_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects, h_reward_hypothesis)
                        h_prob = possible_h_action_to_prob[h_act]
                        # h_prob = 1/(1+np.exp(-60*(h_prob - self.confidence_threshold)))
                        # if h_prob == 0:
                        #     continue

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        # r_sa = team_rew + robot_rew + human_rew
                        #
                        # s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # r_sa += (self.gamma * vf[s11])
                        # expected_reward_sa += (r_sa * probability_of_hyp * h_prob)
                        # expected_reward_sa += (robot_rew * probability_of_hyp) + ((team_rew + human_rew) * probability_of_hyp * h_prob)
                        if r_act is None:
                            robot_rew -= 100
                        elif current_state_remaining_objects[r_act] == 0:
                            robot_rew -= 100

                        expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp * h_prob)

                        # if r_act is None:
                        #     expected_reward_sa -= 100
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     expected_reward_sa -= 100

                        robot_only_reward += (robot_rew * probability_of_hyp)
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

    def collective_value_iteration_argmax_old(self):
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
        n_actions = self.transitions.shape[1]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        # print("true belief prob = ", self.beliefs[self.true_human_rew_idx]['prob'])

        most_probable_h_reward_idx = None
        max_prob = 0
        for candidate_idx in self.beliefs:
            if self.beliefs[candidate_idx]['prob'] > max_prob:
                max_prob = self.beliefs[candidate_idx]['prob']
                most_probable_h_reward_idx = candidate_idx

        probability_of_hyp = 1
        most_probable_h_reward = self.beliefs[most_probable_h_reward_idx]['reward_dict']

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

                        possible_h_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects, most_probable_h_reward)
                        h_prob = possible_h_action_to_prob[h_act]

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state(current_state_remaining_objects, joint_action, most_probable_h_reward)

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
                for action_idx in self.idx_to_action:
                    # pdb.set_trace()
                    # check joint action
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    possible_h_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects,
                                                                                       most_probable_h_reward)
                    h_prob = possible_h_action_to_prob[h_act]

                    next_state, (team_rew, robot_rew, human_rew), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action, most_probable_h_reward)

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

    def collective_value_iteration_argmax(self):
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
        n_actions = self.transitions.shape[1]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))

        most_probable_h_reward_idx = None
        max_prob = 0
        for candidate_idx in self.beliefs:
            if self.beliefs[candidate_idx]['prob'] > max_prob:
                max_prob = self.beliefs[candidate_idx]['prob']
                most_probable_h_reward_idx = candidate_idx

        probability_of_hyp = 1
        # h_reward_hypothesis = self.beliefs[most_probable_h_reward_idx]['reward_dict']
        h_reward_hypothesis = self.believed_human_reward_dict
        # print("h_reward_hypothesis", h_reward_hypothesis)

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

                    # h_reward_hypothesis = self.beliefs[self.true_human_rew_idx]['reward_dict']
                    # probability_of_hyp = self.beliefs[self.true_human_rew_idx]['prob']

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
                        # if r_act is None:
                        #     h_prob = 1
                        h_prob = 1

                        next_state, (team_rew, robot_rew, human_rew), done = \
                            self.step_given_state_w_human_hyp_for_all(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                        r_sa = team_rew + robot_rew + human_rew

                        # print("r_sa = ", r_sa)
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]

                        expected_reward_sa = ((team_rew + robot_rew + human_rew) * probability_of_hyp * h_prob)
                        expected_reward_sa += (self.gamma * vf[s11])

                        # if r_act is None:
                        #     expected_reward_sa -= 0
                        # elif current_state_remaining_objects[r_act] == 0:
                        #     expected_reward_sa -= 100
                        # if expected_reward_sa == 0:
                        #     expected_reward_sa = -2
                        Q[s, action_idx] = expected_reward_sa

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            # print("i = ", i)
            if delta < self.epsilson:
                # print("CVI DONE at iteration ", i)
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
                # h_reward_hypothesis = self.beliefs[self.true_human_rew_idx]['reward_dict']
                # probability_of_hyp = self.beliefs[self.true_human_rew_idx]['prob']

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
                    # if r_act is None:
                    #     h_prob = 1
                    h_prob = 1

                    next_state, (team_rew, robot_rew, human_rew), done = \
                        self.step_given_state_w_human_hyp_for_all(current_state_remaining_objects, joint_action, h_reward_hypothesis)

                    r_sa = team_rew + robot_rew + human_rew
                    # print("r_sa = ", r_sa)
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]

                    expected_reward_sa = ((team_rew + robot_rew + human_rew) * probability_of_hyp * h_prob)
                    expected_reward_sa += (self.gamma * vf[s11])

                    # if expected_reward_sa == 0:
                    #     expected_reward_sa = -2
                    # if r_act is None:
                    #     expected_reward_sa -= 0
                    # elif current_state_remaining_objects[r_act] == 0:
                    #     expected_reward_sa -= 100
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
        n_actions = self.transitions.shape[1]

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
                # print("CVI DONE at iteration ", i)
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
        n_actions = self.transitions.shape[1]

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

                        r_sa = robot_rew + team_rew + human_rew
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        # action_idx = self.action_to_idx[(r_act, h_action)]
                        Q[s, action_idx] = r_sa + (self.greedy_gamma * vf[s11])

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            if delta < self.epsilson:
                # print("Std VI DONE at iteration ", i)
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

                    next_state, (team_rew, robot_rew, human_rew), done = self.greedy_step_given_state(
                        current_state_remaining_objects,
                        joint_action, zeroed_human_reward)

                    r_sa = robot_rew + team_rew + human_rew
                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                    # action_idx = self.action_to_idx[(r_act, h_action)]
                    Q[s, action_idx] = r_sa + (self.greedy_gamma * vf[s11])

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
        n_actions = self.transitions.shape[1]

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
                # print("Std VI DONE at iteration ", i)
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

    def setup_value_iteration(self, h_alpha=0.0):
        # print("Enumerating states")
        self.enumerate_states()
        # print("Done enumerating states")

        if self.episode_history is not None and len(self.episode_history) > 0:
            # print('Running max ent')
            self.run_max_ent()
            # print("Done with max ent", self.believed_human_reward_dict)

        if self.vi_type == 'cvi':
            if self.robot_knows_human_rew is False:
                # print("Running collective_value_iteration")
                # self.collective_value_iteration()
                # self.collective_policy_iteration()
                # print("Running CVI")
                self.collective_value_iteration_argmax()
                # print("Done running CVI")
                # self.collective_policy_iteration_w_h_alpha(h_alpha)
                # self.collective_value_iteration_argmax()
                # print("Done running collective_value_iteration")
            else:
                # print("Running collective_value_iteration_with_true_human_reward")
                # self.collective_value_iteration()
                # self.collective_policy_iteration()
                # self.collective_policy_iteration()
                self.collective_value_iteration_argmax()
                # self.collective_value_iteration_with_true_human_reward()
        else:
            self.greedy_value_iteration()
        return

    def act_old(self, state, is_start=False):
        # max_key = max(self.beliefs, key=lambda k: self.beliefs[k]['prob'])
        # print("max prob belief", self.beliefs[max_key]['reward_dict'])

        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        action_distribution = self.policy[state_idx]

        # for a_idx in self.idx_to_action:
        #     print(f"action {self.idx_to_action[a_idx]} --> value = {action_distribution[a_idx]}")

        # print("action_distribution", action_distribution)
        # action = np.random.choice(np.flatnonzero(action_distribution == action_distribution.max()))
        action = np.argmax(action_distribution)
        # action = np.flatnonzero(action_distribution == action_distribution.max())[0]
        action = self.idx_to_action[action]

        # print("idx_to_action = ", self.idx_to_action)
        # print("action_distribution = ", action_distribution)
        # print("action", action)

        r_action = action[0]
        return r_action

    def take_explore_action(self, state, human_action_to_prob_for_state):
        entropy_of_current_state = entropy([self.beliefs[i]['prob'] for i in self.beliefs], base=2)

        robot_action_to_info_gain = {}
        for robot_action in self.possible_actions:
            if robot_action is not None and state[robot_action] == 0:
                continue
            robot_action_to_info_gain[robot_action] = 0
            for human_action in self.possible_actions:
                if human_action is not None and state[human_action] == 0:
                    prob_a_h_t = 0
                else:
                    prob_a_h_t = human_action_to_prob_for_state[human_action]

                joint_action = {'robot': robot_action, 'human': human_action}

                next_state, (_, _, _), done = self.step_given_state(state, joint_action, self.ind_rew)
                if done:
                    break

                human_action_to_prob = {}
                for h_reward_idx in self.beliefs:
                    # print(f"h_reward_idx = {h_reward_idx} of total # beliefs {len(self.beliefs)}")
                    h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                    probability_of_hyp = self.beliefs[h_reward_idx]['prob']

                    possible_human_action_to_prob = self.get_human_action_under_hypothesis(next_state,
                                                                                           h_reward_hypothesis)
                    # print("possible_human_action_to_prob", possible_human_action_to_prob)
                    # print("probability_of_hyp", probability_of_hyp)

                    for h_act in possible_human_action_to_prob:

                        # else:
                        h_prob = possible_human_action_to_prob[h_act]
                        if h_act is None:
                            # h_prob = 0
                            human_action_to_prob[h_act] = 0
                        else:
                            if h_act not in human_action_to_prob:
                                # h_prob = 0
                                human_action_to_prob[h_act] = 0

                            human_action_to_prob[h_act] += (probability_of_hyp * h_prob)

                # print("human_action_to_prob", human_action_to_prob)
                # pdb.set_trace()
                updated_belief = self.hypothesize_updated_belief(self.beliefs, state, human_action)

                best_entropy_of_next_state = None
                best_info_gain = -10000000
                best_human_action = None
                best_belief = None
                for human_action_next_state in self.possible_actions:
                    if human_action_next_state is not None and next_state[human_action_next_state] > 0:
                        # prob_a_h_t1 = human_action_to_prob[human_action_next_state]
                        next_updated_belief = self.hypothesize_updated_belief(updated_belief, next_state,
                                                                              human_action_next_state)
                        entropy_of_next_state = entropy([next_updated_belief[i]['prob'] for i in next_updated_belief],
                                                        base=2)
                        info_gain = entropy_of_current_state - entropy_of_next_state
                        if info_gain > best_info_gain:
                            best_info_gain = info_gain
                            best_human_action = human_action_next_state
                            best_entropy_of_next_state = entropy_of_next_state
                            best_belief = next_updated_belief

                # if human_action is None:
                #     prob_a_h_t = 0
                # print("\nupdated belief = ", best_belief)
                # print(f"robot: {robot_action}, human action {human_action}, prob= {prob_a_h_t}, best next h={best_human_action}: best info gain = {best_info_gain}\n init entropy = {entropy_of_current_state}, next entropy = {best_entropy_of_next_state}")
                robot_action_to_info_gain[robot_action] += prob_a_h_t * best_entropy_of_next_state
                # pdb.set_trace()

        max_info_gain = -10000
        best_action = None
        best_actions_list = []
        for robot_action in robot_action_to_info_gain:
            # if robot_action is None:
            #     continue
            robot_action_to_info_gain[robot_action] = entropy_of_current_state - robot_action_to_info_gain[robot_action]
            if robot_action_to_info_gain[robot_action] > max_info_gain:
                max_info_gain = robot_action_to_info_gain[robot_action]
                # best_action = robot_action
                best_actions_list = []
                best_actions_list.append(robot_action)

            elif robot_action_to_info_gain[robot_action] == max_info_gain:
                best_actions_list.append(robot_action)

        # print("robot_action_to_info_gain", robot_action_to_info_gain)
        print("robot_action_to_info_gain", robot_action_to_info_gain)
        best_action_idx = np.random.choice(np.arange(len(best_actions_list)))
        best_action = best_actions_list[best_action_idx]
        # max_key = max(robot_action_to_info_gain, key=lambda k: robot_action_to_info_gain[k])
        return best_action

    def get_info_gain(self, state, human_action_to_prob_for_state):
        starting_belief = copy.deepcopy(self.believed_human_reward)

        robot_action_to_info_gain = {}
        for robot_action in self.possible_actions:
            if robot_action is not None and state[robot_action] == 0:
                continue
            robot_action_to_info_gain[robot_action] = 0
            for human_action in self.possible_actions:
                if human_action is not None and state[human_action] == 0:
                    prob_a_h_t = 0
                else:
                    prob_a_h_t = human_action_to_prob_for_state[human_action]

                joint_action = {'robot': robot_action, 'human': human_action}

                next_state, (_, _, _), done = self.step_given_state(state, joint_action, self.ind_rew)
                if done:
                    break

                human_action_to_prob = {}
                for h_reward_idx in self.beliefs:
                    # print(f"h_reward_idx = {h_reward_idx} of total # beliefs {len(self.beliefs)}")
                    h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
                    probability_of_hyp = self.beliefs[h_reward_idx]['prob']

                    possible_human_action_to_prob = self.get_human_action_under_hypothesis(next_state,
                                                                                           h_reward_hypothesis)
                    # print("possible_human_action_to_prob", possible_human_action_to_prob)
                    # print("probability_of_hyp", probability_of_hyp)

                    for h_act in possible_human_action_to_prob:

                        # else:
                        h_prob = possible_human_action_to_prob[h_act]
                        if h_act is None:
                            # h_prob = 0
                            human_action_to_prob[h_act] = 0
                        else:
                            if h_act not in human_action_to_prob:
                                # h_prob = 0
                                human_action_to_prob[h_act] = 0

                            human_action_to_prob[h_act] += (probability_of_hyp * h_prob)

                # print("human_action_to_prob", human_action_to_prob)
                # pdb.set_trace()
                # updated_belief = self.hypothesize_updated_belief(state, human_action)

                best_info_gain = -10000000
                best_human_action = None
                best_belief = None
                for human_action_next_state in self.possible_actions:
                    if human_action_next_state is not None and next_state[human_action_next_state] > 0:
                        # prob_a_h_t1 = human_action_to_prob[human_action_next_state]
                        next_updated_belief = self.hypothesize_updated_belief(state, robot_action, human_action, next_state, human_action_next_state)

                        info_gain = np.linalg.norm(starting_belief - np.array(next_updated_belief))
                        if info_gain > best_info_gain:
                            best_info_gain = info_gain
                            best_human_action = human_action_next_state
                            best_belief = next_updated_belief

                # if human_action is None:
                #     prob_a_h_t = 0
                # print("\nupdated belief = ", best_belief)
                # print(f"robot: {robot_action}, human action {human_action}, prob= {prob_a_h_t}, best next h={best_human_action}: best info gain = {best_info_gain}\n init entropy = {entropy_of_current_state}, next entropy = {best_entropy_of_next_state}")
                robot_action_to_info_gain[robot_action] += prob_a_h_t * best_info_gain
                # pdb.set_trace()

        max_info_gain = -10000
        best_action = None
        best_actions_list = []
        for robot_action in robot_action_to_info_gain:
            # if robot_action is None:
            #     continue
            # robot_action_to_info_gain[robot_action] = entropy_of_current_state - robot_action_to_info_gain[robot_action]
            if robot_action_to_info_gain[robot_action] > max_info_gain:
                max_info_gain = robot_action_to_info_gain[robot_action]
                # best_action = robot_action
                best_actions_list = []
                best_actions_list.append(robot_action)

            elif robot_action_to_info_gain[robot_action] == max_info_gain:
                best_actions_list.append(robot_action)

        # print("robot_action_to_info_gain", robot_action_to_info_gain)
        # print("robot_action_to_info_gain", robot_action_to_info_gain)
        # best_action_idx = np.random.choice(np.arange(len(best_actions_list)))
        # best_action = best_actions_list[best_action_idx]
        # max_key = max(robot_action_to_info_gain, key=lambda k: robot_action_to_info_gain[k])
        return robot_action_to_info_gain

    def act_explore_1(self, state, is_start=False, round_no=0, use_exploration=False, boltzman=False):
        # max_key = max(self.beliefs, key=lambda k: self.beliefs[k]['prob'])
        # print("max prob belief", self.beliefs[max_key]['reward_dict'])

        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        action_distribution = self.policy[state_idx]

        # for a_idx in self.idx_to_action:
        #     print(f"action {self.idx_to_action[a_idx]} --> value = {action_distribution[a_idx]}")

        # print("action_distribution", action_distribution)
        # action = np.random.choice(np.flatnonzero(action_distribution == action_distribution.max()))
        action = np.argmax(action_distribution)
        # action = np.flatnonzero(action_distribution == action_distribution.max())[0]
        action = self.idx_to_action[action]

        # print("idx_to_action = ", self.idx_to_action)
        # print("action_distribution = ", action_distribution)
        # print("action", action)

        r_action = action[0]

        human_action_to_prob = {}
        for h_reward_idx in self.beliefs:
            # print(f"h_reward_idx = {h_reward_idx} of total # beliefs {len(self.beliefs)}")
            h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
            probability_of_hyp = self.beliefs[h_reward_idx]['prob']

            possible_human_action_to_prob = self.get_human_action_under_hypothesis(current_state, h_reward_hypothesis)
            # print("possible_human_action_to_prob", possible_human_action_to_prob)
            # print("probability_of_hyp", probability_of_hyp)

            for h_act in possible_human_action_to_prob:
                if h_act is None:
                    h_prob = 0
                else:
                    h_prob = possible_human_action_to_prob[h_act]
                if h_act not in human_action_to_prob:
                    human_action_to_prob[h_act] = 0

                human_action_to_prob[h_act] += (probability_of_hyp * h_prob)

        single_action_distribution = {}
        for i in range(len(action_distribution)):
            q_value_for_joint = action_distribution[i]
            j_action = self.idx_to_action[i]
            single_r_action = j_action[0]
            single_h_action = j_action[1]
            prob_human_act = human_action_to_prob[single_h_action]
            if single_r_action not in single_action_distribution:
                single_action_distribution[single_r_action] = 0
            single_action_distribution[single_r_action] += (q_value_for_joint * prob_human_act)

        if boltzman:
            own_beta = 2
            Z = 0
            for single_r_action in single_action_distribution:
                # print("single_action_distribution[single_r_action]", single_action_distribution[single_r_action])

                single_action_distribution[single_r_action] = np.round(
                    np.exp(single_action_distribution[single_r_action] / 100 * own_beta), 5)
                Z += single_action_distribution[single_r_action]

            if Z == 0:
                for single_r_action in single_action_distribution:
                    single_action_distribution[single_r_action] = 1 / len(single_action_distribution)
            else:
                is_nan_exists = False
                for single_r_action in single_action_distribution:
                    single_action_distribution[single_r_action] = single_action_distribution[single_r_action] / Z

                    if np.isnan(single_action_distribution[single_r_action]):
                        is_nan_exists = True

                    if is_nan_exists:
                        print("IS NAN EXISTS")
                        for single_r_action in single_action_distribution:
                            single_action_distribution[single_r_action] = 1 / len(single_action_distribution)

            # print("single_action_distribution", single_action_distribution)

            r_action_keys = list(single_action_distribution.keys())
            probs = list(single_action_distribution.values())
            r_action = r_action_keys[np.random.choice(np.arange(len(r_action_keys)), p=probs)]

        else:
            best_r_action = None
            max_prob = -100000
            # print("starting best action", best_r_action)
            for candidate_r_action in self.possible_actions:
                # print("candidate_r_action = ", candidate_r_action)
                if candidate_r_action not in single_action_distribution:
                    # print("continuing")
                    continue

                # print("single_action_distribution[candidate_r_action]", single_action_distribution[candidate_r_action])
                # print("max_prob", max_prob)
                candidate_prob = np.round(single_action_distribution[candidate_r_action], 3)
                if candidate_prob > max_prob:
                    max_prob = candidate_prob
                    best_r_action = candidate_r_action
                    # best_r_action = r_action
                # print("current best action", best_r_action)

            # if r_action != best_r_action:
            # print("best_r_action", best_r_action)
            # print("r_action", r_action)
            # print("single_action_distribution", single_action_distribution)
            r_action = best_r_action

        # r_action = max(single_action_distribution.items(), key=operator.itemgetter(1))[0]
        p_explore = np.random.uniform(0, 1)
        total_rounds = 4
        explore_alpha = max(0.0, -(1.0 / total_rounds) * round_no + 1.0)
        # print("originally proposed action = ", r_action)
        if use_exploration:
            if p_explore < explore_alpha:
                r_action = self.take_explore_action(state, human_action_to_prob)
        # if p_explore < explore_alpha:
        #     r_action = self.take_explore_action(state, human_action_to_prob)
        # print("Exploratory action = ", r_action)
        # self.take_explore_action_entropy_based(state)

        # print("single_action_distribution", single_action_distribution)
        # print("r_action", r_action)
        return r_action

    def act(self, state, is_start=False, round_no=0, use_exploration=False, boltzman=False):
        # max_key = max(self.beliefs, key=lambda k: self.beliefs[k]['prob'])
        # print("max prob belief", self.beliefs[max_key]['reward_dict'])

        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        action_distribution = self.policy[state_idx]

        # for a_idx in self.idx_to_action:
        #     print(f"action {self.idx_to_action[a_idx]} --> value = {action_distribution[a_idx]}")

        # print("action_distribution", action_distribution)
        # action = np.random.choice(np.flatnonzero(action_distribution == action_distribution.max()))
        action = np.argmax(action_distribution)
        # action = np.flatnonzero(action_distribution == action_distribution.max())[0]
        # print("idx_to_action", len(self.idx_to_action))
        action = self.idx_to_action[action]

        # print("idx_to_action = ", self.idx_to_action)
        # print("action_distribution = ", action_distribution)
        # print("action", action)

        r_action = action[0]

        human_action_to_prob = {}
        probability_of_hyp = 1

        possible_human_action_to_prob = self.get_human_action_under_hypothesis(current_state, self.believed_human_reward_dict)
        # print("possible_human_action_to_prob", possible_human_action_to_prob)
        # print("probability_of_hyp", probability_of_hyp)

        for h_act in possible_human_action_to_prob:
            if h_act is None:
                h_prob = 0
            else:
                h_prob = possible_human_action_to_prob[h_act]
            if h_act not in human_action_to_prob:
                human_action_to_prob[h_act] = 0

            human_action_to_prob[h_act] += (probability_of_hyp * h_prob)

        single_action_distribution = {}
        # print("action_distribution", len(action_distribution))
        # print("self.idx_to_state", len(self.idx_to_state))
        for i in range(len(action_distribution)):
            q_value_for_joint = action_distribution[i]
            j_action = self.idx_to_action[i]

            single_r_action = j_action[0]
            single_h_action = j_action[1]
            # print(f"Robot {single_r_action}, Human {single_h_action}")
            # print("q_value_for_joint", q_value_for_joint)
            prob_human_act = human_action_to_prob[single_h_action]
            if single_r_action not in single_action_distribution:
                single_action_distribution[single_r_action] = 0

            single_action_distribution[single_r_action] += (q_value_for_joint * prob_human_act)

        if use_exploration:
            robot_action_to_info_gain = self.get_info_gain(state, human_action_to_prob)
            total_rounds = 4
            explore_phi = max(0.0, -(10.0 / total_rounds) * round_no + 10.0)
            # explore_phi = 10.0
            # print("single_action_distribution", single_action_distribution)
            # print("robot_action_to_info_gain", robot_action_to_info_gain)
            for single_r_action in single_action_distribution:
                if single_r_action not in robot_action_to_info_gain:
                    continue
                potential_info_gain = robot_action_to_info_gain[single_r_action]
                single_action_distribution[single_r_action] += explore_phi * potential_info_gain

        if boltzman:
            own_beta = 2
            Z = 0
            for single_r_action in single_action_distribution:
                # print("single_action_distribution[single_r_action]", single_action_distribution[single_r_action])

                single_action_distribution[single_r_action] = np.round(
                    np.exp(single_action_distribution[single_r_action] / 100 * own_beta), 5)
                Z += single_action_distribution[single_r_action]

            if Z == 0:
                for single_r_action in single_action_distribution:
                    single_action_distribution[single_r_action] = 1 / len(single_action_distribution)
            else:
                is_nan_exists = False
                for single_r_action in single_action_distribution:
                    single_action_distribution[single_r_action] = single_action_distribution[single_r_action] / Z

                    if np.isnan(single_action_distribution[single_r_action]):
                        is_nan_exists = True

                    if is_nan_exists:
                        print("IS NAN EXISTS")
                        for single_r_action in single_action_distribution:
                            single_action_distribution[single_r_action] = 1 / len(single_action_distribution)

            # print("single_action_distribution", single_action_distribution)

            r_action_keys = list(single_action_distribution.keys())
            probs = list(single_action_distribution.values())
            r_action = r_action_keys[np.random.choice(np.arange(len(r_action_keys)), p=probs)]

        else:
            best_r_action = None
            max_prob = -100000
            # print("starting best action", best_r_action)
            for candidate_r_action in self.possible_actions:
                # print("candidate_r_action = ", candidate_r_action)
                if candidate_r_action not in single_action_distribution:
                    # print("continuing")
                    continue
                if candidate_r_action is not None and current_state[candidate_r_action] == 0:
                    continue

                # print("single_action_distribution[candidate_r_action]", single_action_distribution[candidate_r_action])
                # print("max_prob", max_prob)
                candidate_prob = np.round(single_action_distribution[candidate_r_action], 3)
                if candidate_prob > max_prob:
                    max_prob = candidate_prob
                    best_r_action = candidate_r_action
                    # best_r_action = r_action
                # print("current best action", best_r_action)

            # if r_action != best_r_action:
            # print("best_r_action", best_r_action)
            # print("r_action", r_action)
            # print("single_action_distribution", single_action_distribution)
            # print("single_action_distribution", single_action_distribution)
            r_action = best_r_action

        # r_action = max(single_action_distribution.items(), key=operator.itemgetter(1))[0]
        # p_explore = np.random.uniform(0,1)
        # total_rounds = 4
        # # explore_alpha = max(0.0, -(1.0/total_rounds) * round_no + 1.0)
        # explore_alpha = 1.0
        # print("originally proposed action = ", r_action)
        # if use_exploration:
        #     if p_explore < explore_alpha:
        #         r_action = self.take_explore_action(state, human_action_to_prob)
        #         robot_action_to_info_gain = self.get_info_gain(state, human_action_to_prob)
        # # if p_explore < explore_alpha:
        # #     r_action = self.take_explore_action(state, human_action_to_prob)
        #         print("Exploratory action = ", r_action)
        # self.take_explore_action_entropy_based(state)

        # print("single_action_distribution", single_action_distribution)
        # print("r_action", r_action)
        return r_action
