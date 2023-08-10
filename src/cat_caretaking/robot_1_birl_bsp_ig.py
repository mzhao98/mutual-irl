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
# import sklearn
# import sklearn.metrics
MAX_LENGTH_TIME = 7
FEED_DRY_FOOD = 0
FEED_WET_FOOD = 1
REFILL_WATER = 2
WASH_BOWL = 3
SWEEP_FLOOR = 4
CLEAR_LITTER = 5
BATHE_CAT = 6
ACTION_LIST = [FEED_DRY_FOOD, FEED_WET_FOOD, REFILL_WATER, WASH_BOWL, SWEEP_FLOOR, CLEAR_LITTER, BATHE_CAT]
ACTION_TO_TEXT = {FEED_DRY_FOOD: 'feed dry food', FEED_WET_FOOD: 'feed wet food', REFILL_WATER: 'refill water',
                  WASH_BOWL: 'wash bowl', SWEEP_FLOOR: 'sweep floor', CLEAR_LITTER: 'clear litter',
                  BATHE_CAT: 'bathe cat', None: 'None'}
ACTION_TO_PRECONDITION = {FEED_DRY_FOOD: [WASH_BOWL], FEED_WET_FOOD: [WASH_BOWL], REFILL_WATER: [WASH_BOWL], WASH_BOWL: [],
                          SWEEP_FLOOR: [CLEAR_LITTER, WASH_BOWL],
                          CLEAR_LITTER: [WASH_BOWL], BATHE_CAT: []}


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
        self.possible_actions = []

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
        self.gamma = 0.99
        self.greedy_gamma = 0.9
        self.human_gamma = 0.0001
        self.maxiter = 100
        self.small_maxiter = 10
        # self.beta = 5
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
        self.state_actions_completed = []
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
            self.beliefs[idx]['prob'] = 1/len(permutes)
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

    def is_done_given_state(self, state_actions_completed):
        # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = False
        if len(state_actions_completed) >= MAX_LENGTH_TIME:
            done = True
        return done

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

        # print("current_state_tup", current_state_tup)

        normalize_Z = 0

        dict_prob_obs_given_theta = {}
        # print("true idex", self.true_human_rew_idx)
        # print("current_state", current_state)
        # print("human_action", human_action)
        # print("current_state[human_action]", current_state[human_action])
        for idx in self.beliefs:
            # human_rew_dict = self.beliefs[idx]['reward_dict']
            q_values_table = self.belief_idx_to_q_values[idx]
            # prob_theta = self.beliefs[idx]['prob']
            h_reward_hypothesis = self.beliefs[idx]['reward_dict']
            # if robot_action is None:
            #     human_only_rew_for_action = h_reward_hypothesis[human_action]
            # else:
            #     human_only_rew_for_action = h_reward_hypothesis[human_action] + self.ind_rew[robot_action]
            human_only_rew_for_action = 0
            max_composite_reward_for_human_action = -100
            for r_action in self.possible_actions:
                human_r, robot_r = 0, 0
                copy_current_state = copy.deepcopy(current_state)

                if r_action is not None and r_action not in copy_current_state:
                    preconditions_list = ACTION_TO_PRECONDITION[r_action]
                    if set(preconditions_list).issubset(copy_current_state):
                        copy_current_state.append(r_action)
                        robot_r += self.ind_rew[r_action]
                        # robot_r += h_reward_hypothesis[r_action]

                if human_action is not None and human_action not in copy_current_state:
                    preconditions_list = ACTION_TO_PRECONDITION[human_action]
                    if set(preconditions_list).issubset(copy_current_state):
                        copy_current_state.append(human_action)
                        human_r += h_reward_hypothesis[human_action]

                team_r = -10
                if self.is_done_given_state(copy_current_state):
                    team_r = 0
                candidate_rew = team_r + robot_r + human_r
                if candidate_rew > max_composite_reward_for_human_action:
                    max_composite_reward_for_human_action = candidate_rew

            # human_only_rew_for_action = sum([(h_reward_hypothesis[human_action] + self.ind_rew[r_action]) for r_action in self.possible_actions])
            human_only_rew_for_action = max_composite_reward_for_human_action
            # print("human_only_rew_for_action", human_only_rew_for_action)
            # pdb.set_trace()

            # if current_state[human_action] == 0:
            #     human_only_rew_for_action = -2

            # print("human action", human_action)
            # print("human_only_rew_for_action", human_only_rew_for_action)


            sum_Z = 0
            all_possible_rews = []
            for possible_action in h_reward_hypothesis:
                if possible_action not in current_state:

                    human_only_rew_for_possible_action = -100
                    for r_action in self.possible_actions:
                        human_r, robot_r = 0, 0
                        copy_current_state = copy.deepcopy(current_state)

                        if r_action is not None and r_action not in copy_current_state:
                            preconditions_list = ACTION_TO_PRECONDITION[r_action]
                            if set(preconditions_list).issubset(copy_current_state):
                                copy_current_state.append(r_action)
                                robot_r += self.ind_rew[r_action]
                                # robot_r += h_reward_hypothesis[r_action]

                        if possible_action is not None and possible_action not in copy_current_state:
                            preconditions_list = ACTION_TO_PRECONDITION[possible_action]
                            if set(preconditions_list).issubset(copy_current_state):
                                copy_current_state.append(possible_action)
                                human_r += h_reward_hypothesis[possible_action]

                        team_r = -10
                        if self.is_done_given_state(copy_current_state):
                            team_r = 0
                        candidate_rew = team_r + robot_r + human_r
                        if candidate_rew > human_only_rew_for_possible_action:
                            human_only_rew_for_possible_action = candidate_rew


                    sum_Z += human_only_rew_for_possible_action
                    all_possible_rews.append(human_only_rew_for_possible_action)

            # print("h_reward_hypothesis", h_reward_hypothesis)
            # print("self.possible_actions", self.possible_actions)
            # print("all_possible_rews", all_possible_rews)
            # print("current_state", current_state)
            # print("sum_Z", sum_Z)
            # human_only_rew_for_action /= sum_Z
            if human_only_rew_for_action == max(all_possible_rews):
                human_only_rew_for_action = self.update_threshold
            else:
                human_only_rew_for_action = 1-self.update_threshold


            # print(f"idx = {idx}: {h_reward_hypothesis}")
            # print("human_only_rew_for_action", human_only_rew_for_action)
            # q_value_for_obs = q_values_table[state_idx, joint_action_idx]
            exp_q_value_for_obs = np.exp(self.beta * human_only_rew_for_action)
            # print(f"idx = {idx}, human_only_rew_for_action = {human_only_rew_for_action}, exp_q_value_for_obs = {np.round(exp_q_value_for_obs, 2)}")
            # print("h_reward_hypothesis", h_reward_hypothesis)
            # exp_q_value_for_obs = q_value_for_obs

            # print()
            # print("belief", self.beliefs[idx]['reward_dict'])
            # print("q_value_for_obs", q_value_for_obs)
            # print("exp_q_value_for_obs", exp_q_value_for_obs)

            normalize_Z += exp_q_value_for_obs

            dict_prob_obs_given_theta[idx] = exp_q_value_for_obs
            # print("exp_q_value_for_obs", exp_q_value_for_obs)

        if normalize_Z == 0:
            # print("PROBLEM WITH Z=0 at dict_prob_obs_given_theta")
            normalize_Z = 0.01
        for idx in dict_prob_obs_given_theta:
            dict_prob_obs_given_theta[idx] = dict_prob_obs_given_theta[idx] / normalize_Z
            # print(f"idx = {idx}, likelihood value = {np.round(dict_prob_obs_given_theta[idx],2)}")

        normalization_denominator = 0
        for idx in self.beliefs:
            prob_theta = self.beliefs[idx]['prob']
            prob_obs_given_theta_normalized = dict_prob_obs_given_theta[idx]
            self.beliefs[idx]['prob'] = prob_theta * prob_obs_given_theta_normalized
            # print(f"idx = {idx}, prob before normalize = {np.round(self.beliefs[idx]['prob'], 2)}")
            normalization_denominator += prob_theta * prob_obs_given_theta_normalized

        # pdb.set_trace()
        if normalization_denominator == 0:
            # print("PROBLEM WITH Z=0 at beliefs")
            normalization_denominator = 0.01
        for idx in self.beliefs:
            self.beliefs[idx]['prob'] = self.beliefs[idx]['prob'] / normalization_denominator
            # print(f"idx = {idx}, final prob = {np.round(self.beliefs[idx]['prob'],2)}")
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
        # salad_actions = [GET_PLATE, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        if len(self.state_actions_completed) >= MAX_LENGTH_TIME:
            return True
        return False


    def step_given_state(self, input_state, joint_action, human_reward):
        state_actions_completed = copy.deepcopy(input_state)
        # print("joint_action", joint_action)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew, robot_rew, human_rew = -10, 0, 0

        if robot_action is not None and robot_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]
            if set(preconditions_list).issubset(state_actions_completed):
                robot_action_successful = True
                state_actions_completed.append(robot_action)
                robot_rew += self.ind_rew[robot_action]
                # robot_rew += human_reward[robot_action]

        if human_action is not None and human_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed):
                human_action_successful = True
                state_actions_completed.append(human_action)
                human_rew += human_reward[human_action]

        # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = False
        if len(state_actions_completed) >= MAX_LENGTH_TIME:
            done = True

        if done:
            team_rew = 0
        return state_actions_completed, (team_rew, robot_rew, human_rew), done

    def human_step_given_state(self, input_state, joint_action, human_reward):
        state_actions_completed = copy.deepcopy(input_state)
        robot_action = joint_action['robot']
        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew, robot_rew, human_rew = -10, 0, 0

        if robot_action is not None and robot_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]
            if set(preconditions_list).issubset(state_actions_completed):
                robot_action_successful = True
                state_actions_completed.append(robot_action)
                robot_rew += self.ind_rew[robot_action]
                # robot_rew += human_reward[robot_action]

        if human_action is not None and human_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed):
                human_action_successful = True
                state_actions_completed.append(human_action)
                human_rew += human_reward[human_action]

        # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = False
        if len(state_actions_completed) >= MAX_LENGTH_TIME:
            done = True

        if done:
            team_rew = 0
        return state_actions_completed, (team_rew, robot_rew, human_rew), done

    def state_to_tuple(self, state_actions_completed):
        return tuple(sorted(state_actions_completed))


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
            if candidate_r_act is not None and candidate_r_act not in state_actions_completed:
                preconditions_list = ACTION_TO_PRECONDITION[candidate_r_act]
                if set(preconditions_list).issubset(state_actions_completed):
                    state_actions_completed.append(candidate_r_act)
                    robot_r += self.ind_rew[candidate_r_act]
                    # robot_r += human_reward[candidate_r_act]

            if candidate_h_act is not None and candidate_h_act not in state_actions_completed:
                preconditions_list = ACTION_TO_PRECONDITION[candidate_h_act]
                if set(preconditions_list).issubset(state_actions_completed):
                    state_actions_completed.append(candidate_h_act)
                    human_r += human_reward[candidate_h_act]
                # else:
                #     continue

            # salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
            # soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
            done = False
            if len(state_actions_completed) >= MAX_LENGTH_TIME:
                done = True

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
                possible_h_action_to_prob[candidate_h_act] = possible_h_action_to_prob[candidate_h_act]/denom

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
        # if self.is_collaborative_human is True:
        possible_h_action_to_prob = self.get_human_action_under_collaborative_hypothesis(current_state_remaining_objects, human_reward)
        # else:
        #     possible_h_action_to_prob = self.get_human_action_under_greedy_hypothesis(current_state_remaining_objects, human_reward)
        return possible_h_action_to_prob

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
        self.n_actions = self.transitions.shape[2]

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
                #     # action_idx = self.action_to_idx[(None, None)]
                #         self.Q[s, action_idx] = self.vf[s]

                # else:
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
                            self.step_given_state(current_state, joint_action, h_reward_hypothesis)


                        expected_reward_sa += ((team_rew + robot_rew + human_rew) * probability_of_hyp)


                    next_state, (_, _, _), done = \
                        self.step_given_state(current_state, joint_action, self.ind_rew)
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
            #         # action_idx = self.action_to_idx[(None, None)]
            #         self.Q[s, action_idx] = self.vf[s]

            # else:
                # print("\ncurrent_state", current_state)
                # sum_of_probs = 0
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
                        self.step_given_state(current_state, joint_action, h_reward_hypothesis)


                    expected_reward_sa +=  (( team_rew + robot_rew + human_rew) * probability_of_hyp)

                    # if r_act is None:
                    #     expected_reward_sa -= 0
                    # elif current_state_remaining_objects[r_act] == 0:
                    #     expected_reward_sa -= 100


                next_state, (_, _, _), done = \
                    self.step_given_state(current_state, joint_action, self.ind_rew)
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


    def setup_value_iteration(self, h_alpha=0.0):
        self.enumerate_states()

        if self.vi_type == 'cvi':
            if self.robot_knows_human_rew is False:
                # print("Running collective_value_iteration")
                # self.collective_value_iteration()
                self.collective_policy_iteration()
                # self.collective_value_iteration_argmax()
                # self.collective_policy_iteration_w_h_alpha(h_alpha)
                # self.collective_value_iteration_argmax()
                # print("Done running collective_value_iteration")
            else:
                # print("Running collective_value_iteration_with_true_human_reward")
                # self.collective_value_iteration()
                # self.collective_policy_iteration()
                self.collective_policy_iteration()
                # self.collective_value_iteration_argmax()
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

    def hypothesize_updated_belief(self, beliefs, current_state, human_action):
        updated_beliefs = copy.deepcopy(beliefs)
        normalize_Z = 0

        dict_prob_obs_given_theta = {}
        for idx in updated_beliefs:
            h_reward_hypothesis = updated_beliefs[idx]['reward_dict']

            max_composite_reward_for_human_action = -100
            for r_action in self.possible_actions:
                human_r, robot_r = 0, 0
                copy_current_state = copy.deepcopy(current_state)

                if r_action is not None and r_action not in copy_current_state:
                    preconditions_list = ACTION_TO_PRECONDITION[r_action]
                    if set(preconditions_list).issubset(copy_current_state):
                        copy_current_state.append(r_action)
                        robot_r += self.ind_rew[r_action]
                        # robot_r += h_reward_hypothesis[r_action]

                if human_action is not None and human_action not in copy_current_state:
                    preconditions_list = ACTION_TO_PRECONDITION[human_action]
                    if set(preconditions_list).issubset(copy_current_state):
                        copy_current_state.append(human_action)
                        human_r += h_reward_hypothesis[human_action]

                team_r = -10
                if self.is_done_given_state(copy_current_state):
                    team_r = 0

                # team_r = -2
                candidate_rew = team_r + robot_r + human_r
                if candidate_rew > max_composite_reward_for_human_action:
                    max_composite_reward_for_human_action = candidate_rew

            human_only_rew_for_action = max_composite_reward_for_human_action



            sum_Z = 0
            all_possible_rews = []
            for possible_action in h_reward_hypothesis:
                if possible_action not in current_state:

                    human_only_rew_for_possible_action = -100
                    for r_action in self.possible_actions:
                        human_r, robot_r = 0, 0
                        copy_current_state = copy.deepcopy(current_state)

                        if r_action is not None and r_action not in copy_current_state:
                            preconditions_list = ACTION_TO_PRECONDITION[r_action]
                            if set(preconditions_list).issubset(copy_current_state):
                                copy_current_state.append(r_action)
                                robot_r += self.ind_rew[r_action]
                                # robot_r += h_reward_hypothesis[r_action]

                        if possible_action is not None and possible_action not in copy_current_state:
                            preconditions_list = ACTION_TO_PRECONDITION[possible_action]
                            if set(preconditions_list).issubset(copy_current_state):
                                copy_current_state.append(possible_action)
                                human_r += h_reward_hypothesis[possible_action]

                        team_r = -10
                        if self.is_done_given_state(copy_current_state):
                            team_r = 0

                        # team_r = -2
                        candidate_rew = team_r + robot_r + human_r
                        if candidate_rew > human_only_rew_for_possible_action:
                            human_only_rew_for_possible_action = candidate_rew

                    sum_Z += human_only_rew_for_possible_action
                    all_possible_rews.append(human_only_rew_for_possible_action)

            # print("sum_Z", sum_Z)
            # human_only_rew_for_action /= sum_Z

            if human_only_rew_for_action == max(all_possible_rews):
                human_only_rew_for_action = self.update_threshold
            else:
                human_only_rew_for_action = 1 - self.update_threshold

            # print(f"idx = {idx}: {h_reward_hypothesis}")
            # print("human_only_rew_for_action", human_only_rew_for_action)
            # q_value_for_obs = q_values_table[state_idx, joint_action_idx]
            exp_q_value_for_obs = np.exp(self.beta * human_only_rew_for_action)
            # print(f"idx = {idx}, human_only_rew_for_action = {human_only_rew_for_action}, exp_q_value_for_obs = {np.round(exp_q_value_for_obs, 2)}")
            # print("h_reward_hypothesis", h_reward_hypothesis)
            # exp_q_value_for_obs = q_value_for_obs

            # print()
            # print("belief", self.beliefs[idx]['reward_dict'])
            # print("q_value_for_obs", q_value_for_obs)
            # print("exp_q_value_for_obs", exp_q_value_for_obs)

            normalize_Z += exp_q_value_for_obs

            dict_prob_obs_given_theta[idx] = exp_q_value_for_obs
            # print("exp_q_value_for_obs", exp_q_value_for_obs)

        if normalize_Z == 0:
            # print("PROBLEM WITH Z=0 at dict_prob_obs_given_theta")
            normalize_Z = 0.01
        for idx in dict_prob_obs_given_theta:
            dict_prob_obs_given_theta[idx] = dict_prob_obs_given_theta[idx] / normalize_Z
            # print(f"idx = {idx}, likelihood value = {np.round(dict_prob_obs_given_theta[idx],2)}")

        normalization_denominator = 0
        for idx in updated_beliefs:
            prob_theta = updated_beliefs[idx]['prob']
            prob_obs_given_theta_normalized = dict_prob_obs_given_theta[idx]
            updated_beliefs[idx]['prob'] = prob_theta * prob_obs_given_theta_normalized
            # print(f"idx = {idx}, prob before normalize = {np.round(self.beliefs[idx]['prob'], 2)}")
            normalization_denominator += prob_theta * prob_obs_given_theta_normalized

        # pdb.set_trace()
        if normalization_denominator == 0:
            # print("PROBLEM WITH Z=0 at beliefs")
            normalization_denominator = 0.01
        for idx in updated_beliefs:
            updated_beliefs[idx]['prob'] = updated_beliefs[idx]['prob'] / normalization_denominator
            # print(f"idx = {idx}, final prob = {np.round(self.beliefs[idx]['prob'],2)}")
        # pdb.set_trace()

        return updated_beliefs

    def get_info_gain(self, state, human_action_to_prob_for_state):
        num_categories = len(self.beliefs)
        entropy_of_current_state = entropy([self.beliefs[i]['prob'] for i in self.beliefs], base=num_categories)
        # print("entropy_of_current_state", entropy_of_current_state)
        max_prob = max([self.beliefs[i]['prob'] for i in self.beliefs])
        # print("max prob", max_prob)
        equal_to_max = []
        for i in self.beliefs:
            if self.beliefs[i]['prob'] == max_prob:
                # print("top hyps", self.beliefs[i]['reward_dict'])
                equal_to_max.append(self.beliefs[i]['reward_dict'])


        robot_action_to_info_gain = {}
        for robot_action in self.possible_actions:
            if robot_action is not None and robot_action in state:
                continue
            robot_action_to_info_gain[robot_action] = 0
            for human_action in self.possible_actions:
                if human_action is not None and human_action in state:
                    prob_a_h_t = 0
                elif set(ACTION_TO_PRECONDITION[human_action]).issubset(state) is False:
                    prob_a_h_t = 0
                else:
                    prob_a_h_t = human_action_to_prob_for_state[human_action]

                joint_action = {'robot': robot_action, 'human': human_action}

                next_state, (_, _, _), done = self.step_given_state(state, joint_action, self.ind_rew)
                if done:
                    break

                kl_diverge = 1.0
                if len(equal_to_max) == 1:
                    kl_diverge = 0.0
                elif len(equal_to_max) == 2:
                    h_reward_hypothesis_1 = equal_to_max[0]
                    human_dist_1 = self.get_human_action_under_hypothesis(next_state, h_reward_hypothesis_1)
                    human_dist_1 = {k: v for k, v in human_dist_1.items() if k in next_state and k not in next_state}
                    human_dist_1 = [k for k, v in human_dist_1.items() if v == max(human_dist_1.values())]

                    # human_dist_1 /= sum(human_dist_1)
                    h_reward_hypothesis_2 = equal_to_max[1]
                    human_dist_2 = self.get_human_action_under_hypothesis(next_state, h_reward_hypothesis_2)
                    human_dist_2 = {k: v for k, v in human_dist_2.items() if k in next_state and k not in next_state}
                    human_dist_2 = [k for k, v in human_dist_2.items() if v == max(human_dist_2.values())]
                    # print("human_dist_1", human_dist_1)
                    # print("human_dist_2", human_dist_2)
                    overlapping_actions = list(set(human_dist_1) & set(human_dist_2))
                    overlapping_actions = set(overlapping_actions)
                    # kl_diverge = sklearn.metrics.mutual_info_score(human_dist_1, human_dist_2)
                    kl_diverge = 1-min(1, len(overlapping_actions))
                    # print("kl_diverge", kl_diverge)
                    # pdb.set_trace()
                else:
                    all_overlapping_kl = []
                    for i in range(len(equal_to_max)):

                        h_reward_hypothesis_1 = equal_to_max[i]
                        human_dist_1 = self.get_human_action_under_hypothesis(next_state, h_reward_hypothesis_1)
                        human_dist_1 = {k: v for k, v in human_dist_1.items() if k in next_state and k not in next_state}
                        human_dist_1 = [k for k, v in human_dist_1.items() if v == max(human_dist_1.values())]
                        overlapping_actions = human_dist_1

                        for j in range(len(equal_to_max)):

                            h_reward_hypothesis_2 = equal_to_max[j]
                            human_dist_2 = self.get_human_action_under_hypothesis(next_state, h_reward_hypothesis_2)
                            human_dist_2 = {k: v for k, v in human_dist_2.items() if k in next_state and k not in next_state}
                            human_dist_2 = [k for k, v in human_dist_2.items() if v == max(human_dist_2.values())]

                            overlapping_actions = list(set(overlapping_actions) & set(human_dist_2))

                            # overlapping_actions = list(set(overlapping_actions))
                        # num_overlap = 1-min(1, len(overlapping_actions))
                        num_overlap = len(overlapping_actions)
                        all_overlapping_kl.append(num_overlap)
                    # all_overlapping_kl = np.array(all_overlapping_kl)
                    # pdb.set_trace()
                    kl_diverge = np.mean(all_overlapping_kl)
                    kl_diverge = 1.0 - min(1, kl_diverge)
                    # print("kl_diverge", kl_diverge)
                # kl_diverge = 1.0

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
                        elif set(ACTION_TO_PRECONDITION[h_act]).issubset(next_state) is False:
                            human_action_to_prob[h_act] = 0

                        else:
                            if h_act not in human_action_to_prob:
                                # h_prob = 0
                                human_action_to_prob[h_act] = 0

                            human_action_to_prob[h_act] += (probability_of_hyp * h_prob)

                # print("human_action_to_prob", human_action_to_prob)
                # pdb.set_trace()
                # updated_belief = self.hypothesize_updated_belief(self.beliefs, state, human_action)
                updated_belief = copy.deepcopy(self.beliefs)
                # print("updated_belief", updated_belief)

                best_entropy_of_next_state = None
                best_info_gain = -10000000
                best_human_action = None
                best_belief = None
                # print("entropy_of_current_state", entropy_of_current_state)
                for human_action_next_state in self.possible_actions:
                    if human_action_next_state is not None and human_action_next_state not in next_state:
                        # prob_a_h_t1 = human_action_to_prob[human_action_next_state]
                        next_updated_belief = self.hypothesize_updated_belief(updated_belief, next_state,
                                                                              human_action_next_state)
                        # print("next_updated_belief", [next_updated_belief[i]['prob'] for i in next_updated_belief])
                        entropy_of_next_state = entropy([next_updated_belief[i]['prob'] for i in next_updated_belief],
                                                        base=num_categories)

                        # print(f"human : {human_action_next_state}, entropy_of_next_state", entropy_of_next_state)
                        if set(ACTION_TO_PRECONDITION[human_action_next_state]).issubset(next_state) is False:
                            info_gain = 0
                        else:
                            info_gain = entropy_of_current_state - entropy_of_next_state
                        # print("info_gain", info_gain)
                        # print("best_info_gain", best_info_gain)
                        # # print("entropy_of_current_state", entropy_of_current_state)
                        # print("entropy_of_next_state", entropy_of_next_state)
                        if info_gain > best_info_gain:
                            best_info_gain = info_gain
                            best_human_action = human_action_next_state
                            best_entropy_of_next_state = entropy_of_next_state
                            best_belief = next_updated_belief

                # if human_action is None:
                #     prob_a_h_t = 0
                # print("\nupdated belief = ", best_belief)
                # print(f"robot: {robot_action}, human action {human_action}, prob= {prob_a_h_t}, best next h={best_human_action}: best info gain = {best_info_gain}\n init entropy = {entropy_of_current_state}, next entropy = {best_entropy_of_next_state}")
                robot_action_to_info_gain[robot_action] += prob_a_h_t * best_info_gain * kl_diverge
                # pdb.set_trace()

        # max_info_gain = -10000
        # best_action = None
        # best_actions_list = []
        # for robot_action in robot_action_to_info_gain:
        #     # if robot_action is None:
        #     #     continue
        #     robot_action_to_info_gain[robot_action] = entropy_of_current_state - robot_action_to_info_gain[robot_action]
        #     if robot_action_to_info_gain[robot_action] > max_info_gain:
        #         max_info_gain = robot_action_to_info_gain[robot_action]
        #         # best_action = robot_action
        #         best_actions_list = []
        #         best_actions_list.append(robot_action)
        #
        #     elif robot_action_to_info_gain[robot_action] == max_info_gain:
        #         best_actions_list.append(robot_action)

        # print("robot_action_to_info_gain", robot_action_to_info_gain)
        # print("robot_action_to_info_gain", robot_action_to_info_gain)
        # best_action_idx = np.random.choice(np.arange(len(best_actions_list)))
        # best_action = best_actions_list[best_action_idx]
        # max_key = max(robot_action_to_info_gain, key=lambda k: robot_action_to_info_gain[k])
        return robot_action_to_info_gain

    def act(self, state, is_start=False, round_no=0, use_exploration=False, boltzman=False, epsilon=0.1):
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
            # print("h_reward_hypothesis", h_reward_hypothesis)

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

        # print("human_action_to_prob", human_action_to_prob)
        single_action_distribution = {}
        for i in range(len(action_distribution)):
            q_value_for_joint = action_distribution[i]
            j_action = self.idx_to_action[i]
            # print(f"j_action = {j_action}", q_value_for_joint)
            single_r_action = j_action[0]
            single_h_action = j_action[1]
            prob_human_act = human_action_to_prob[single_h_action]
            # print("prob_human_act", prob_human_act)
            if single_r_action not in single_action_distribution:
                single_action_distribution[single_r_action] = 0
            single_action_distribution[single_r_action] += (q_value_for_joint * prob_human_act)

        # print("human_action_to_prob", human_action_to_prob)
        # print("single_action_distribution", single_action_distribution)
        # if use_exploration:
        robot_action_to_info_gain = self.get_info_gain(state, human_action_to_prob)

        total_rounds = 4
        # explore_phi = max(0.0, -(10.0 / total_rounds) * round_no + 10.0)
        explore_phi = 25.0
        print("single_action_distribution", single_action_distribution)
        print("robot_action_to_info_gain", robot_action_to_info_gain)
        for single_r_action in single_action_distribution:
            if single_r_action not in robot_action_to_info_gain:
                continue
            potential_info_gain = robot_action_to_info_gain[single_r_action]
            single_action_distribution[single_r_action] += explore_phi * potential_info_gain

        print("AFTER EXPLORE: single_action_distribution", single_action_distribution)

        best_r_action = None
        max_prob = -100000
        # print("starting best action", best_r_action)
        for candidate_r_action in self.possible_actions:
            # print("candidate_r_action = ", candidate_r_action)
            if candidate_r_action not in single_action_distribution:
                # print("continuing")
                continue

            # if candidate_r_action is not None and current_state[candidate_r_action] == 0:
            #     continue

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
