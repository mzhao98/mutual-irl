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
    def __init__(self, team_rew, ind_rew, human_rew, starting_state, robot_knows_human_rew, permutes, vi_type, is_collaborative_human, update_threshold=0.9, beta=5):
        self.team_rew = team_rew
        self.ind_rew = ind_rew
        self.true_human_rew = human_rew
        self.permutes = permutes
        self.robot_knows_human_rew = robot_knows_human_rew
        self.beta = beta

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
        self.epsilson = 0.01
        self.gamma = 0.8
        self.greedy_gamma = 0.9
        self.human_gamma = 0.001
        self.maxiter = 5
        self.small_maxiter = 10
        # self.beta = 1
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

    def reset(self):
        self.state_actions_completed = [[], []]
        self.possible_actions = []
        # for action in self.starting_actions_to_perform:
        for action in ACTION_LIST:
            self.possible_actions.append(action)

    def set_beliefs_with_true_reward(self):
        # self.human_rew = copy.deepcopy(self.true_human_rew)
        self.enumerate_states()
        self.n_states = self.transitions.shape[0]
        self.n_actions = self.transitions.shape[2]

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.true_human_rew.values())))
            permutes = list(set(permutes))
        # print("permutes = ", permutes)
        # print("len(permutes", len(permutes))
        object_keys = list(self.true_human_rew.keys())
        # self.belief_idx_to_q_values = {}
        for idx in range(len(permutes)):
            human_rew_values = permutes[idx]
            human_rew_dict = {object_keys[i]: list(human_rew_values)[i] for i in range(len(object_keys))}
            self.beliefs[idx] = {}
            self.beliefs[idx]['reward_dict'] = human_rew_dict
            self.beliefs[idx]['prob'] = 0
            self.beliefs[idx]['q'] = np.zeros((self.n_states, self.n_actions))

            # self.belief_idx_to_q_values[idx] = {}
            # self.belief_idx_to_q_values[idx]['reward_dict'] = human_rew_dict
            # self.belief_idx_to_q_values[idx]['q'] = np.zeros((self.n_states, self.n_actions))

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
        self.n_states = self.transitions.shape[0]
        self.n_actions = self.transitions.shape[2]

        self.beliefs = {}
        if self.permutes is not None:
            permutes = self.permutes
        else:
            permutes = list(itertools.permutations(list(self.true_human_rew.values())))
            permutes = list(set(permutes))

        object_keys = list(self.true_human_rew.keys())
        # self.belief_idx_to_q_values = {}
        for idx in range(len(permutes)):
            # print("starting idx = ", idx)
            human_rew_values = permutes[idx]
            human_rew_dict = {object_keys[i]: list(human_rew_values)[i] for i in range(len(object_keys))}

            # q_values_table = self.human_candidate_value_iteration(human_rew_dict)
            # self.belief_idx_to_q_values[idx] = {}
            # self.belief_idx_to_q_values[idx]['reward_dict'] = human_rew_dict
            # self.belief_idx_to_q_values[idx]['q'] = np.zeros((self.n_states, self.n_actions))


            self.beliefs[idx] = {}
            self.beliefs[idx]['reward_dict'] = human_rew_dict
            self.beliefs[idx]['prob'] = 1/len(permutes)
            self.beliefs[idx]['q'] = np.zeros((self.n_states, self.n_actions))
            # print("done with idx = ", idx)

            if human_rew_dict == self.true_human_rew:
                self.true_human_rew_idx = idx

        # print("sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]) = ", sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()]))
        assert abs(sum([self.beliefs[idx]['prob'] for idx in self.beliefs.keys()])-1) < 0.01

        # with open('data/q_values.pkl', 'wb') as file:
        #     pickle.dump(self.belief_idx_to_q_values, file)

    def reset_belief_history(self):
        self.history_of_human_beliefs = []
        self.history_of_robot_beliefs_of_true_human_rew = []
        self.history_of_robot_beliefs_of_max_human_rew = []
        self.episode_history = []

    def update_based_on_h_action(self, current_state, robot_action, human_action):
        # print("current_state, robot_action, human_action", (current_state, robot_action, human_action))
        self.episode_history.append((current_state, robot_action, human_action))

        if self.robot_knows_human_rew is True:
            return

        self.history_of_human_beliefs.append(copy.deepcopy(self.beliefs))
        self.history_of_robot_beliefs_of_true_human_rew.append(self.beliefs[self.true_human_rew_idx]['prob'])

        max_key = max(self.beliefs, key=lambda keyname: self.beliefs[keyname]['prob'])
        self.history_of_robot_beliefs_of_max_human_rew.append(self.beliefs[max_key]['reward_dict'])

        joint_action = (robot_action, human_action)
        joint_action_idx = self.action_to_idx[joint_action]

        current_state_tup = self.state_to_tuple(current_state)
        state_idx = self.state_to_idx[current_state_tup]

        normalize_Z = 0


        for idx in self.beliefs:
            h_reward_hypothesis = self.beliefs[idx]['reward_dict']
            Q = self.beliefs[idx]['q']
            q_sa = Q[state_idx, joint_action_idx]
            all_valid_q_values = []
            for action_idx in self.idx_to_action:
                if human_action is not None:
                    if robot_action is not None:
                        if human_action not in current_state and robot_action not in current_state:
                            all_valid_q_values.append(Q[state_idx, action_idx])
                    else:
                        if human_action not in current_state:
                            all_valid_q_values.append(Q[state_idx, action_idx])
                else:
                    if robot_action is not None:
                        if robot_action not in current_state:
                            all_valid_q_values.append(Q[state_idx, action_idx])
                    else:
                        all_valid_q_values.append(Q[state_idx, action_idx])


            # all_valid_q_values = [Q[state_idx, action_idx] for action_idx in self.idx_to_action if (current_state[human_action] > 0) and current_state[robot_action] > 0]
            if len(all_valid_q_values) > 0:
                if q_sa == max(all_valid_q_values):
                    q_sa = self.update_threshold
                else:
                    q_sa = 1-self.update_threshold
            else:
                q_sa = 1-self.update_threshold


            boltz_prob = np.exp(self.beta * q_sa) * self.beliefs[idx]['prob']
            self.beliefs[idx]['prob'] = boltz_prob
            normalize_Z += boltz_prob

        # pdb.set_trace()
        if normalize_Z == 0:
            # print("PROBLEM WITH Z=0 at beliefs")
            normalize_Z = 1
        for idx in self.beliefs:
            self.beliefs[idx]['prob'] = self.beliefs[idx]['prob'] / normalize_Z
            # print(f"idx = {idx}, final prob = {np.round(self.beliefs[idx]['prob'],2)}")

        # pdb.set_trace()
        return

    def tuple_to_state_list(self, state_tuple):
        # state_list = list(state_tuple)
        state_list = [list(state_tuple[0]),
                                    list(state_tuple[1])]

        return state_list

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

    def human_step_given_state(self, input_state, joint_action, human_reward):
        state_actions_completed = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew, robot_rew, human_rew = -5, 0, 0

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

        done = self.is_done_given_state(state_actions_completed)

        if done:
            team_rew = 0
        return state_actions_completed, (team_rew, robot_rew, human_rew), done

    def state_to_tuple(self, state_actions_completed):
        return tuple([tuple(sorted(state_actions_completed[0])),
                      tuple(sorted(state_actions_completed[1]))])


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
        # print("NUMBER OF actions", len(actions))
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # assert 0 ==1
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


    def get_human_action_under_collaborative_hypothesis(self, current_state_remaining_objects, h_reward_idx, robot_action):
        state_remaining_objects = copy.deepcopy(current_state_remaining_objects)


        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']
        Q = self.beliefs[h_reward_idx]['q']

        state_idx = self.state_to_idx[self.state_to_tuple(state_remaining_objects)]



        human_beta = self.beta
        possible_h_action_to_prob = {}
        denom = 0
        for candidate_h_act in self.possible_actions:
            joint_action = (robot_action, candidate_h_act)
            joint_action_idx = self.action_to_idx[joint_action]
            # print("joint_action_idx", joint_action_idx)
            q_sa = Q[state_idx, joint_action_idx]
            # print("q_sa", q_sa)
            boltz_prob = np.exp(human_beta * q_sa)
            possible_h_action_to_prob[candidate_h_act] = boltz_prob
            denom += boltz_prob

        # print("denom", denom)
        if denom != 0:
            for candidate_h_act in possible_h_action_to_prob:
                possible_h_action_to_prob[candidate_h_act] = possible_h_action_to_prob[candidate_h_act] / denom



        return possible_h_action_to_prob

    def get_human_action_under_hypothesis(self, current_state_remaining_objects, h_reward_idx, robot_action):
        # print("human_reward", human_reward)
        # if self.is_collaborative_human is True:
        possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(current_state_remaining_objects, h_reward_idx, robot_action)
        # else:
        #     possible_h_action_to_prob = self.get_human_action_under_greedy_hypothesis(current_state_remaining_objects, human_reward)
        return possible_h_action_to_prob

    def collective_value_iteration(self, h_reward_idx):
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
        self.n_actions = self.transitions.shape[2]

        # initialize value function
        # pi = np.zeros((self.n_states, 1))
        vf = np.zeros((self.n_states, 1))
        Q = np.zeros((self.n_states, self.n_actions))
        policy = {}
        h_reward_hypothesis = self.beliefs[h_reward_idx]['reward_dict']

        for i in range(self.maxiter):
            print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(self.n_states):
                # store old value function
                # print("s=", s)
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
                #     for action_idx in range(self.n_actions):
                #         Q[s, action_idx] = Q[s, action_idx]

                # else:

                human_action_to_updated_beliefs = {}
                # robot_action_to_human_policy = {}
                for action_idx in self.idx_to_action:
                    # print("action_idx", action_idx)
                    # check joint action
                    joint_action = self.idx_to_action[action_idx]
                    r_act = joint_action[0]
                    h_act = joint_action[1]
                    joint_action = {'robot': r_act, 'human': h_act}

                    expected_reward_sa = 0



                    next_state, (team_rew, robot_rew, human_rew), done = \
                        self.step_given_state(current_state_remaining_objects, joint_action,
                                              h_reward_hypothesis)

                    expected_reward_sa += (team_rew + robot_rew + human_rew)
                    if h_act not in human_action_to_updated_beliefs:
                        updated_beliefs = self.hypothesize_updated_belief(self.beliefs,
                                                                          current_state_remaining_objects,
                                                                          h_act, r_act)
                        human_action_to_updated_beliefs[h_act] = {}
                        human_action_to_updated_beliefs[h_act]['updated_beliefs'] = updated_beliefs
                        robot_policy = self.setup_robot_policy(updated_beliefs)
                        human_action_to_updated_beliefs[h_act]['robot_policy'] = robot_policy
                        next_robot_action = robot_policy[s]
                        human_action_to_updated_beliefs[h_act]['next_robot_action'] = next_robot_action
                        human_action_to_prob = self.get_human_action_under_hypothesis(
                            current_state_remaining_objects,
                            h_reward_idx, next_robot_action)
                        human_action_to_updated_beliefs[h_act]['human_action_to_prob'] = human_action_to_prob
                        # robot_action_to_human_policy[robot_policy[s]] = human_action_to_prob

                    else:
                        next_robot_action = human_action_to_updated_beliefs[h_act]['next_robot_action']
                        human_action_to_prob = human_action_to_updated_beliefs[h_act]['human_action_to_prob']
                        # human_action_to_prob = robot_action_to_human_policy[robot_policy[s]]
                    # # updated_beliefs = self.hypothesize_updated_belief(self.beliefs, current_state_remaining_objects, h_act)
                    # # robot_policy = self.setup_robot_policy(updated_beliefs)
                    # next_robot_action = robot_policy[s]
                    # human_action_to_prob = self.get_human_action_under_hypothesis(current_state_remaining_objects,
                    #                                                               h_reward_idx, next_robot_action)


                    s11 = self.state_to_idx[self.state_to_tuple(next_state)]

                    for next_human_action in human_action_to_prob:
                        next_joint_action = (next_robot_action, next_human_action)
                        next_joint_action_idx = self.action_to_idx[next_joint_action]

                        expected_reward_sa += (self.gamma * Q[s11, next_joint_action_idx] * human_action_to_prob[next_human_action])
                    Q[s, action_idx] = expected_reward_sa

                vf[s] = np.max(Q[s, :], 0)

                delta = max(delta, abs(old_v - vf[s]))



            # check for convergence
            # print("delta", delta)
            if delta < self.epsilson:
                # print("CVI DONE at iteration ", i)
                break

        return Q

    def setup_robot_policy(self, beliefs):
        robot_policy = {}
        for idx in beliefs:
            h_reward_hypothesis = beliefs[idx]['reward_dict']
            Q = beliefs[idx]['q']
            hypothesis_prob = beliefs[idx]['prob']

            for s in range(self.n_states):
                current_state_tuple = list(self.idx_to_state[s])
                current_state_dict = self.tuple_to_state_list(current_state_tuple)
                # human_action_to_prob = self.get_human_action_under_hypothesis(current_state_dict, idx)

                for a in self.idx_to_action:
                    Q_sa = Q[s, a]
                    j_action = self.idx_to_action[a]
                    single_r_action = j_action[0]
                    single_h_action = j_action[1]
                    human_action_to_prob = self.get_human_action_under_hypothesis(current_state_dict, idx, single_r_action)
                    prob_human_act = human_action_to_prob[single_h_action]

                    if s not in robot_policy:
                        robot_policy[s] = {}

                    if single_r_action not in robot_policy[s]:
                        robot_policy[s][single_r_action] = 0

                    robot_policy[s][single_r_action] += (prob_human_act * hypothesis_prob * Q_sa)

        for s in robot_policy:
            max_action = max(robot_policy[s], key=robot_policy[s].get)
            robot_policy[s] = max_action
        return robot_policy

    def setup_value_iteration(self, h_alpha=0.0):
        self.enumerate_states()
        # print("setting up robot policy")
        self.robot_policy = self.setup_robot_policy(self.beliefs)

        if self.vi_type == 'cvi':
            if self.robot_knows_human_rew is False:
                for idx in self.beliefs:
                    self.beliefs[idx]['q'] = self.collective_value_iteration(idx)
                self.robot_policy = self.setup_robot_policy(self.beliefs)

            else:

                for idx in self.beliefs:
                    self.beliefs[idx]['q'] = self.collective_value_iteration(idx)
                self.robot_policy = self.setup_robot_policy(self.beliefs)

        return



    def hypothesize_updated_belief(self, beliefs, current_state, human_action, robot_action):
        updated_beliefs = copy.deepcopy(beliefs)
        joint_action = (robot_action, human_action)
        joint_action_idx = self.action_to_idx[joint_action]

        current_state_tup = self.state_to_tuple(current_state)
        state_idx = self.state_to_idx[current_state_tup]

        normalize_Z = 0

        for idx in updated_beliefs:
            Q = updated_beliefs[idx]['q']
            q_sa = Q[state_idx, joint_action_idx]
            boltz_prob = np.exp(self.beta * q_sa) * updated_beliefs[idx]['prob']
            updated_beliefs[idx]['prob'] = boltz_prob
            normalize_Z += boltz_prob

        # pdb.set_trace()
        if normalize_Z == 0:
            # print("PROBLEM WITH Z=0 at beliefs")
            normalize_Z = 1
        for idx in updated_beliefs:
            updated_beliefs[idx]['prob'] = updated_beliefs[idx]['prob'] / normalize_Z

        return updated_beliefs


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

    def act(self, state, is_start=False, round_no=0, use_exploration=False, boltzman=False, epsilon=0.1):
        # max_key = max(self.beliefs, key=lambda k: self.beliefs[k]['prob'])
        # print("max prob belief", self.beliefs[max_key]['reward_dict'])

        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        r_action = self.robot_policy[state_idx]

        # print("single_action_distribution", single_action_distribution)
        # if use_exploration:
        #     robot_action_to_info_gain = self.get_info_gain(state, human_action_to_prob)
        #     total_rounds = 4
        #     explore_phi = max(0.0, -(10.0 / total_rounds) * round_no + 10.0)
        #     # explore_phi = 10.0
        #     # print("single_action_distribution", single_action_distribution)
        #     # print("robot_action_to_info_gain", robot_action_to_info_gain)
        #     for single_r_action in single_action_distribution:
        #         if single_r_action not in robot_action_to_info_gain:
        #             continue
        #         potential_info_gain = robot_action_to_info_gain[single_r_action]
        #         single_action_distribution[single_r_action] += explore_phi * potential_info_gain
        #
        # best_r_action = None
        # max_prob = -100000
        # # print("starting best action", best_r_action)
        # for candidate_r_action in self.possible_actions:
        #     # print("candidate_r_action = ", candidate_r_action)
        #     if candidate_r_action not in single_action_distribution:
        #         # print("continuing")
        #         continue
        #
        #     # print("single_action_distribution[candidate_r_action]", single_action_distribution[candidate_r_action])
        #     # print("max_prob", max_prob)
        #     candidate_prob = np.round(single_action_distribution[candidate_r_action], 3)
        #     if candidate_prob > max_prob:
        #         max_prob = candidate_prob
        #         best_r_action = candidate_r_action
        #         # best_r_action = r_action
        #     # print("current best action", best_r_action)
        #
        # # if r_action != best_r_action:
        #     # print("best_r_action", best_r_action)
        #     # print("r_action", r_action)
        #     # print("single_action_distribution", single_action_distribution)
        # r_action = best_r_action

        # r_action = max(single_action_distribution.items(), key=operator.itemgetter(1))[0]
        # p_explore = np.random.uniform(0,1)
        # total_rounds = 4
        # explore_alpha = max(0.0, -(1.0/total_rounds) * round_no + 1.0)
        # # # print("originally proposed action = ", r_action)
        # if use_exploration:
        #     if p_explore < explore_alpha:
        #         r_action = self.take_explore_action(state, human_action_to_prob)
        # if p_explore < explore_alpha:
        #     r_action = self.take_explore_action(state, human_action_to_prob)
        #     print("Exploratory action = ", r_action)
            # self.take_explore_action_entropy_based(state)


        # print("single_action_distribution", single_action_distribution)
        # print("r_action", r_action)
        return r_action

    def act_human(self, state, predicted_robot_action, round_no=0):

        current_state = copy.deepcopy(state)
        # print(f"current_state = {current_state}")
        current_state_tup = self.state_to_tuple(current_state)

        state_idx = self.state_to_idx[current_state_tup]

        Q = self.beliefs[self.true_human_rew_idx]['q']
        best_action_idx = np.argmax(Q[state_idx, :])
        best_action = self.idx_to_action[best_action_idx]
        action_distribution = Q[state_idx, :]
        # action_distribution = []
        # # action_idx_list = []
        # for human_action in self.possible_actions:
        #     joint_action = (human_action, predicted_robot_action)
        #     joint_action_idx = self.action_to_idx[joint_action]
        #     # action_idx_list.append(joint_action_idx)
        #     action_distribution.append(Q[state_idx, joint_action_idx])

        boltz_action_distribution = np.array([np.exp(self.beta * q) for q in action_distribution])
        boltz_action_distribution = boltz_action_distribution / np.sum(boltz_action_distribution)
        # best_action_idx = np.argmax(action_distribution)
        best_action_idx = np.random.choice(np.arange(len(boltz_action_distribution)), p=boltz_action_distribution)
        best_action = self.idx_to_action[best_action_idx]
        h_action = best_action[1]
        # h_action = self.possible_actions[best_action_idx]
        # pdb.set_trace()


        return h_action
