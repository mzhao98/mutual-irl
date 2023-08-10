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

ACT_AS_BAIT = 0
POSITION_TO_SHOOT = 1
SHOOT = 2
POSITION_TO_BAIT = 3
WAIT = 4

ACTION_LIST = [ACT_AS_BAIT, POSITION_TO_SHOOT, SHOOT, WAIT, POSITION_TO_BAIT]
ACTION_TO_TEXT = {ACT_AS_BAIT: 'act as bait',
                  POSITION_TO_SHOOT: 'position to shoot',
                  SHOOT: 'shoot',
                  POSITION_TO_BAIT: 'position to bait',
                    WAIT: 'wait',
                  None: 'None'}
ACTION_TO_PRECONDITION = {ACT_AS_BAIT: [POSITION_TO_BAIT],
                          POSITION_TO_SHOOT: [],
                          SHOOT: [POSITION_TO_SHOOT],
                          POSITION_TO_BAIT: [],
                          WAIT: [],
                          None: []}


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
        self.state_actions_completed = {}
        self.possible_actions = [None]

        self.starting_actions_to_perform = starting_state
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
        self.believed_human_reward_dict = {elem: 0 for elem in self.starting_actions_to_perform}
        self.feature_matrix = None
        self.believed_human_reward = [0,0,0,0,0]


    def reset(self):
        self.state_actions_completed = [[], []]
        self.possible_actions = []
        for action in self.starting_actions_to_perform:
            self.possible_actions.append(action)

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
                state_as_list = list(state)

                featurized_state = []
                for action_type in self.starting_actions_to_perform:
                    if action_type in state_as_list:
                        featurized_state.append(1)
                    else:
                        featurized_state.append(0)
                    # featurized_state.append(self.start_remaining_objects[object_type] - state_as_dict[object_type])
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
        if len(self.episode_history) > 50:
            self.episode_history.pop(0)


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
                state_as_list = list(np.concatenate(state))

                featurized_state = []
                for action_type in self.starting_actions_to_perform:
                    if action_type in state_as_list:
                        featurized_state.append(1)
                    else:
                        featurized_state.append(0)
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
            if min(self.believed_human_reward) < 0:
                self.believed_human_reward_dict[self.starting_actions_to_perform[i]] = (self.believed_human_reward[i] - min(self.believed_human_reward)) * 100

            else:
                self.believed_human_reward_dict[self.starting_actions_to_perform[i]] = self.believed_human_reward[i] * 100

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
        if SHOOT in np.concatenate(self.state_actions_completed):
            return True
        return False

    def is_done_given_state(self, state_actions_completed):
        # salad_actions = [GET_PLATE, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = False
        if SHOOT in np.concatenate(state_actions_completed):
            return True
        return done

    def step_given_state(self, input_state, joint_action, human_reward):
        state_actions_completed = copy.deepcopy(input_state)
        # print("joint_action", joint_action)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew, robot_rew, human_rew = -10, 0, 0

        if robot_action is not None and robot_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]

            if set(preconditions_list).issubset(state_actions_completed[0]) and len(state_actions_completed[0]) < 3:
                if robot_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        robot_action_successful = True
                        state_actions_completed[0].append(robot_action)
                        robot_rew += self.ind_rew[robot_action]
                else:
                    robot_action_successful = True
                    state_actions_completed[0].append(robot_action)
                    robot_rew += self.ind_rew[robot_action]

                # robot_rew += self.human.ind_rew[robot_action]

        if human_action is not None and human_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed[1]) and len(state_actions_completed[1]) < 3:
                if human_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        human_action_successful = True
                        state_actions_completed[1].append(human_action)
                        human_rew += human_reward[human_action]
                else:
                    human_action_successful = True
                    state_actions_completed[1].append(human_action)
                    human_rew += human_reward[human_action]

        # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = self.is_done_given_state(state_actions_completed)

        if done:
            team_rew = 0
        return state_actions_completed, (team_rew, robot_rew, human_rew), done

    def step_given_state_w_human_hyp_for_all(self, input_state, joint_action, human_reward):
        state_actions_completed = copy.deepcopy(input_state)
        # print("joint_action", joint_action)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew, robot_rew, human_rew = -10, 0, 0

        if robot_action is not None and robot_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]

            if set(preconditions_list).issubset(state_actions_completed[0]) and len(state_actions_completed[0]) < 3:
                if robot_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        robot_action_successful = True
                        state_actions_completed[0].append(robot_action)
                        robot_rew += human_reward[robot_action]
                else:
                    robot_action_successful = True
                    state_actions_completed[0].append(robot_action)
                    robot_rew += human_reward[robot_action]

                # robot_rew += self.human.ind_rew[robot_action]

        if human_action is not None and human_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed[1]) and len(state_actions_completed[1]) < 3:
                if human_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        human_action_successful = True
                        state_actions_completed[1].append(human_action)
                        human_rew += human_reward[human_action]
                else:
                    human_action_successful = True
                    state_actions_completed[1].append(human_action)
                    human_rew += human_reward[human_action]

        # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = self.is_done_given_state(state_actions_completed)

        if done:
            team_rew = 0
        return state_actions_completed, (team_rew, robot_rew, human_rew), done


    def state_to_tuple(self, state_actions_completed):

        return tuple([tuple(sorted(state_actions_completed[0])), tuple(sorted(state_actions_completed[1]))])


    def enumerate_states(self):
        self.reset()

        actions = self.get_all_possible_joint_actions()
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state_actions_completed)]

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

    def get_human_action_under_collaborative_hypothesis(self, current_state_remaining_objects, human_reward):

        best_human_act = []
        max_reward = -100
        h_action = None
        human_r = 0
        robot_r = 0
        best_joint_action = None

        best_reward_for_human_action = {}

        for joint_act in self.possible_joint_actions:
            candidate_r_act = joint_act['robot']
            candidate_h_act = joint_act['human']
            human_r = 0
            robot_r = 0
            # joint_act = {'robot': candidate_r_act, 'human': candidate_h_act}
            # print("joint_act", joint_act)
            team_r = -10
            state_actions_completed = copy.deepcopy(current_state_remaining_objects)
            # print("candidate_h_act", candidate_h_act)
            # _, (team_r, robot_r, human_r), _ = self.human_step_given_state(current_state_remaining_objects, joint_act, human_reward)
            if candidate_r_act is not None and candidate_r_act not in np.concatenate(state_actions_completed):
                preconditions_list = ACTION_TO_PRECONDITION[candidate_r_act]
                if set(preconditions_list).issubset(state_actions_completed[0]):
                    if candidate_r_act == SHOOT:
                        if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                            state_actions_completed[0].append(candidate_r_act)
                            robot_r += self.ind_rew[candidate_r_act]
                    else:
                        state_actions_completed[0].append(candidate_r_act)
                        robot_r += self.ind_rew[candidate_r_act]

                    # robot_r += human_reward[candidate_r_act]

            if candidate_h_act is not None and candidate_h_act not in np.concatenate(state_actions_completed):
                preconditions_list = ACTION_TO_PRECONDITION[candidate_h_act]
                if set(preconditions_list).issubset(state_actions_completed[1]):
                    if candidate_h_act == SHOOT:
                        if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                            state_actions_completed[1].append(candidate_h_act)
                            human_r += human_reward[candidate_h_act]
                    else:
                        state_actions_completed[1].append(candidate_h_act)
                        human_r += human_reward[candidate_h_act]


            # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
            # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
            done = self.is_done_given_state(state_actions_completed)

            if done:
                team_r = 0

            candidate_rew = team_r + robot_r + human_r
            # print("candidate_rew", candidate_rew)

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
            else:
                best_reward_for_human_action[candidate_h_act] = -1

        # if human_reward == {(0, 0): 3, (2, 0): 3, (0, 1): 6, (2, 1): 6}:
        #     print(f"best_reward_for_human_action, {best_reward_for_human_action}")
        # pdb.set_trace()
        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]
        # print("best_reward_for_human_action", best_reward_for_human_action)
        # print("self.ind_rew", self.ind_rew)
        # print("human_reward", human_reward)
        # human_beta = 15
        human_beta = self.beta
        possible_h_action_to_prob = {}
        denom = 0
        for candidate_h_act in best_reward_for_human_action:
            boltz_prob = np.exp(human_beta * best_reward_for_human_action[candidate_h_act])
            possible_h_action_to_prob[candidate_h_act] = boltz_prob
            denom += boltz_prob

        if denom != 0:
            for candidate_h_act in possible_h_action_to_prob:
                possible_h_action_to_prob[candidate_h_act] = possible_h_action_to_prob[candidate_h_act] / denom

        # possible_h_action_to_prob = {}
        # for candidate_h_act in self.possible_actions:
        #     if candidate_h_act == h_action:
        #         possible_h_action_to_prob[candidate_h_act] = 1
        #     elif candidate_h_act is None:
        #         possible_h_action_to_prob[candidate_h_act] = 0
        #     elif best_reward_for_human_action[candidate_h_act] == max_reward:
        #         possible_h_action_to_prob[candidate_h_act] = 1
        #     else:
        #         possible_h_action_to_prob[candidate_h_act] = 0
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
        possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(
            current_state_remaining_objects, human_reward)
        # if self.is_collaborative_human is True:
        #     possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(
        #         current_state_remaining_objects, human_reward)
        # else:
        #     possible_h_action_to_prob = self.get_human_action_under_greedy_hypothesis(current_state_remaining_objects,
        #                                                                               human_reward)
        return possible_h_action_to_prob

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

                # current_state_remaining_objects = copy.deepcopy(list(self.idx_to_state[s]))
                current_state_remaining_objects = copy.deepcopy([list(self.idx_to_state[s][0]), list(self.idx_to_state[s][1])])

                # current_state_remaining_objects = {}
                # for obj_tuple in current_state:
                #     if obj_tuple not in current_state_remaining_objects:
                #         current_state_remaining_objects[obj_tuple] = 1
                #     else:
                #         current_state_remaining_objects[obj_tuple] += 1
                # for obj_tuple in self.possible_actions:
                #     if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
                #         current_state_remaining_objects[obj_tuple] = 0
                #
                # if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                #     for action_idx in range(n_actions):
                #         # action_idx = self.action_to_idx[(None, None)]
                #         Q[s, action_idx] = vf[s]

                # else:
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

            # current_state_remaining_objects = copy.deepcopy(list(self.idx_to_state[s]))
            current_state_remaining_objects = copy.deepcopy(
                [list(self.idx_to_state[s][0]), list(self.idx_to_state[s][1])])

            # current_state_remaining_objects = {}
            # for obj_tuple in current_state:
            #     if obj_tuple not in current_state_remaining_objects:
            #         current_state_remaining_objects[obj_tuple] = 1
            #     else:
            #         current_state_remaining_objects[obj_tuple] += 1
            # for obj_tuple in self.possible_actions:
            #     if obj_tuple is not None and obj_tuple not in current_state_remaining_objects:
            #         current_state_remaining_objects[obj_tuple] = 0
            #
            # if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
            #     for action_idx in range(n_actions):
            #         # action_idx = self.action_to_idx[(None, None)]
            #         Q[s, action_idx] = vf[s]

            # else:
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
        # else:
        #     self.greedy_value_iteration()
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



                if candidate_r_action is not None and candidate_r_action in np.concatenate(current_state):
                    continue

                if set(ACTION_TO_PRECONDITION[candidate_r_action]).issubset(current_state[0]) is False:
                    continue

                # print("single_action_distribution[candidate_r_action]", single_action_distribution[candidate_r_action])
                # print("max_prob", max_prob)
                candidate_prob = np.round(single_action_distribution[candidate_r_action], 3)
                if candidate_prob >= max_prob:
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
