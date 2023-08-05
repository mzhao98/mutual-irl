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

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

import pickle
import json
import sys
import os



class Greedy_Human:
    def __init__(self, ind_rew, true_robot_rew, starting_state, h_alpha=0.0, h_deg_collab=0.5):
        self.ind_rew = ind_rew
        self.true_robot_rew = true_robot_rew
        self.h_alpha = h_alpha
        if self.true_robot_rew is not None:
            self.robot_rew = copy.deepcopy(self.true_robot_rew)

        self.starting_objects = starting_state
        self.possible_actions = [None]
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.possible_actions:
                self.possible_actions.append(obj_tuple)

    def reset(self):
        self.starting_objects = starting_state
        self.possible_actions = [None]
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.possible_actions:
                self.possible_actions.append(obj_tuple)


    def act(self, state):
        # print("human state, ", state)

        # best_human_act = []
        other_actions = []
        max_reward = -100
        h_action = None
        # state = None
        for candidate_h_act in self.possible_actions:
            if candidate_h_act is not None:
                if state[candidate_h_act] > 0:
                    candidate_rew = self.ind_rew[candidate_h_act]
                else:
                    candidate_rew = -1000

                # if candidate_rew == max_reward:
                #     if candidate_h_act not in best_human_act:
                #         best_human_act.append(candidate_h_act)
                #         best_human_act =

                # elif candidate_rew > max_reward:
                #     max_reward = candidate_rew
                #     best_human_act = [candidate_h_act]
                #
                # else:
                #     other_actions.append(candidate_h_act)
                if candidate_rew > max_reward:
                    h_action = candidate_h_act
                    max_reward = candidate_rew

        #
        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]
        #
        # r = np.random.uniform(0, 1)
        # if r < self.h_alpha:
        #     if len(other_actions) > 0:
        #         h_action = other_actions[np.random.choice(np.arange(len(other_actions)))]
        # print("human_acting", state)
        return h_action

    def act_old(self, state):
        other_actions = []
        best_human_act = []
        max_reward = -100
        for candidate_h_act in self.possible_actions:
            if candidate_h_act is not None:
                if state[candidate_h_act] > 0:

                    candidate_rew = self.ind_rew[candidate_h_act]
                    if candidate_rew == max_reward:
                        if candidate_h_act not in best_human_act:
                            best_human_act.append(candidate_h_act)

                    elif candidate_rew > max_reward:
                        max_reward = candidate_rew
                        best_human_act = [candidate_h_act]
                    else:
                        other_actions.append(candidate_h_act)


        if len(best_human_act) == 0:
            h_action = None
        else:
            # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
            h_action = best_human_act[0]

        r = np.random.uniform(0,1)
        if r < self.h_alpha:
            if len(other_actions) > 0:
                h_action = other_actions[np.random.choice(np.arange(len(other_actions)))]


        return h_action


class Collaborative_Human:
    def __init__(self, ind_rew, true_robot_rew, starting_state, h_alpha=0.0):
        self.ind_rew = ind_rew
        self.true_robot_rew = true_robot_rew
        self.h_alpha = h_alpha
        if self.true_robot_rew is not None:
            self.robot_rew = copy.deepcopy(self.true_robot_rew)

        self.starting_objects = starting_state

        self.possible_actions = []
        for obj_tuple_r in self.starting_objects:
            for obj_tuple_h in self.starting_objects:
                if (obj_tuple_r, obj_tuple_h) not in self.possible_actions:
                    self.possible_actions.append((obj_tuple_r, obj_tuple_h))


    def get_remaining_objects_from_state(self, state):
        state_remaining_objects = {}
        for obj_tuple in state:
            if obj_tuple not in state_remaining_objects:
                state_remaining_objects[obj_tuple] = 1
            else:
                state_remaining_objects[obj_tuple] += 1
        return state_remaining_objects

    def step_given_state(self, input_state, joint_action):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action[0]
        human_action = joint_action[1]

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
                        robot_rew += self.true_robot_rew[robot_action]
                        human_rew += self.ind_rew[robot_action]
                        # team_rew += (self.true_robot_rew[robot_action] + self.ind_rew[robot_action])

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = -1, -1
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1

                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     # human_rew = 0
                        #     robot_rew += self.true_robot_rew[robot_action]
                        # else:
                        # robot_rew = -1
                        human_rew += self.ind_rew[human_action]

                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.true_robot_rew[robot_action]
                        human_rew += self.ind_rew[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.true_robot_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += self.ind_rew[human_action]

        # team_rew = robot_rew + human_rew
        return team_rew, robot_rew, human_rew


    def act(self, state):
        # best_human_act = []
        other_actions = []
        max_reward = -2
        h_action = None
        for (candidate_r_act, candidate_h_act) in self.possible_actions:
            team_rew, r_rew, h_rew = self.step_given_state(state, (candidate_r_act, candidate_h_act))
            # print("(candidate_r_act, candidate_h_act)", (candidate_r_act, candidate_h_act))
            candidate_rew = team_rew + r_rew + h_rew
            # print("candidate_rew", candidate_rew)
            if candidate_h_act is not None:
                # if state[candidate_h_act] > 0
                # if candidate_rew == max_reward:
                #     if candidate_h_act not in best_human_act:
                #         best_human_act.append(candidate_h_act)
                #         best_human_act =

                # elif candidate_rew > max_reward:
                #     max_reward = candidate_rew
                #     best_human_act = [candidate_h_act]
                #
                # else:
                #     other_actions.append(candidate_h_act)
                if candidate_rew > max_reward:
                    h_action = candidate_h_act
                    max_reward = candidate_rew

        #
        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]
        #
        # r = np.random.uniform(0, 1)
        # if r < self.h_alpha:
        #     if len(other_actions) > 0:
        #         h_action = other_actions[np.random.choice(np.arange(len(other_actions)))]

        return h_action

class Suboptimal_Collaborative_Human:
    def __init__(self, ind_rew, true_robot_rew, starting_state, h_alpha=0.0, h_deg_collab=0.5):
        self.ind_rew = ind_rew
        self.true_robot_rew = true_robot_rew
        self.h_alpha = h_alpha
        self.h_deg_collab = h_deg_collab
        if self.true_robot_rew is not None:
            self.robot_rew = copy.deepcopy(self.true_robot_rew)

        self.starting_objects = starting_state

        self.possible_actions = []
        for obj_tuple_r in self.starting_objects:
            for obj_tuple_h in self.starting_objects:
                if (obj_tuple_r, obj_tuple_h) not in self.possible_actions:
                    self.possible_actions.append((obj_tuple_r, obj_tuple_h))

    def reset(self):
        self.possible_actions = []
        for obj_tuple_r in self.starting_objects:
            for obj_tuple_h in self.starting_objects:
                if (obj_tuple_r, obj_tuple_h) not in self.possible_actions:
                    self.possible_actions.append((obj_tuple_r, obj_tuple_h))


    def get_remaining_objects_from_state(self, state):
        state_remaining_objects = {}
        for obj_tuple in state:
            if obj_tuple not in state_remaining_objects:
                state_remaining_objects[obj_tuple] = 1
            else:
                state_remaining_objects[obj_tuple] += 1
        return state_remaining_objects

    def step_given_state(self, input_state, joint_action):
        state_remaining_objects = copy.deepcopy(input_state)
        robot_action = joint_action[0]
        human_action = joint_action[1]

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
                        robot_rew += self.true_robot_rew[robot_action]
                        human_rew += self.ind_rew[robot_action]
                        # team_rew += (self.true_robot_rew[robot_action] + self.ind_rew[robot_action])

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = -1, -1
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1

                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     # human_rew = 0
                        #     robot_rew += self.true_robot_rew[robot_action]
                        # else:
                        # robot_rew = -1
                        robot_rew += self.true_robot_rew[human_action]

                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.true_robot_rew[robot_action]
                        human_rew += self.ind_rew[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.true_robot_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += self.ind_rew[human_action]

        # team_rew = robot_rew + human_rew
        return team_rew, robot_rew, human_rew


    def act(self, state, round_no=0):
        # best_human_act = []
        other_actions = []
        max_reward = -2
        h_action = None
        beta = 1
        h_action_to_boltz_prob = {}

        for (candidate_r_act, candidate_h_act) in self.possible_actions:
            team_rew, r_rew, h_rew = self.step_given_state(state, (candidate_r_act, candidate_h_act))
            # print("(candidate_r_act, candidate_h_act)", (candidate_r_act, candidate_h_act))
            candidate_rew = team_rew + ((self.h_deg_collab*100) * r_rew) + ((1-self.h_deg_collab) * 100 * h_rew)
            # candidate_rew = team_rew + ((1 - self.h_deg_collab) * 100 * h_rew)
            if self.h_deg_collab == 0.5:
                candidate_rew = team_rew + (r_rew) + (h_rew)
            # print("candidate_rew", candidate_rew)
            # if candidate_h_act is not None:
                # if state[candidate_h_act] > 0
                # if candidate_rew == max_reward:
                #     if candidate_h_act not in best_human_act:
                #         best_human_act.append(candidate_h_act)
                #         best_human_act =

                # elif candidate_rew > max_reward:
                #     max_reward = candidate_rew
                #     best_human_act = [candidate_h_act]
                #
                # else:
                #     other_actions.append(candidate_h_act)
            if candidate_rew >= max_reward:
                h_action = candidate_h_act
                max_reward = candidate_rew

            if candidate_h_act not in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = -10000
            if candidate_h_act is not None:
                if state[candidate_h_act] > 0 and candidate_rew > h_action_to_boltz_prob[candidate_h_act]:
                    h_action_to_boltz_prob[candidate_h_act] = candidate_rew
            else:
                if candidate_rew > h_action_to_boltz_prob[candidate_h_act]:
                    h_action_to_boltz_prob[candidate_h_act] = candidate_rew

            if h_action not in other_actions:
                other_actions.append(h_action)

        sum_Z = 0
        for candidate_h_act in h_action_to_boltz_prob:
            h_action_to_boltz_prob[candidate_h_act] = np.round(np.exp(beta * h_action_to_boltz_prob[candidate_h_act]),5)
            sum_Z += h_action_to_boltz_prob[candidate_h_act]

        if sum_Z == 0:
            for candidate_h_act in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = 1/len(h_action_to_boltz_prob)
        else:
            for candidate_h_act in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = h_action_to_boltz_prob[candidate_h_act]/sum_Z

        h_action_keys = list(h_action_to_boltz_prob.keys())
        probs = list(h_action_to_boltz_prob.values())
        # print()
        # print("h_action_to_boltz_prob", h_action_to_boltz_prob)
        # print("probs", probs)
        # if round_no < 5:
        # h_action = h_action_keys[np.random.choice(np.arange(len(h_action_keys)), p=probs)]
        # print("h_action", h_action)
        #
        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]
        #
        # print("best h_action", h_action)
        # print(f"Other {other_actions} removing {h_action}")
        # other_actions.remove(h_action)
        # r = np.random.uniform(0, 1)
        # if round_no < 5 and r < self.h_alpha:
        #     if len(other_actions) > 0:
        #         # print("Using random")
        #         h_action = other_actions[np.random.choice(np.arange(len(other_actions)))]

        return h_action

    def act_ood(self, state, round_no=0):
        # best_human_act = []
        other_actions = []
        max_reward = -2
        h_action = None
        beta = 1
        h_action_to_boltz_prob = {}

        for (candidate_r_act, candidate_h_act) in self.possible_actions:
            team_rew, r_rew, h_rew = self.step_given_state(state, (candidate_r_act, candidate_h_act))
            # print("(candidate_r_act, candidate_h_act)", (candidate_r_act, candidate_h_act))
            candidate_rew = team_rew + ((self.h_deg_collab*100) * r_rew) + ((1-self.h_deg_collab) * 100 * h_rew)
            # candidate_rew = team_rew + ((1 - self.h_deg_collab) * 100 * h_rew)
            if self.h_deg_collab == 0.5:
                candidate_rew = team_rew + (r_rew) + (h_rew)
            if candidate_h_act not in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = 0
            h_action_to_boltz_prob[candidate_h_act] += candidate_rew

        h_action_idx = np.argmax(list(h_action_to_boltz_prob.values()))
        h_action = list(h_action_to_boltz_prob.keys())[h_action_idx]

        sum_Z = 0
        for candidate_h_act in h_action_to_boltz_prob:
            h_action_to_boltz_prob[candidate_h_act] = np.round(np.exp(beta * h_action_to_boltz_prob[candidate_h_act]),5)
            sum_Z += h_action_to_boltz_prob[candidate_h_act]

        if sum_Z == 0:
            for candidate_h_act in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = 1/len(h_action_to_boltz_prob)
        else:
            for candidate_h_act in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = h_action_to_boltz_prob[candidate_h_act]/sum_Z

        h_action_keys = list(h_action_to_boltz_prob.keys())
        probs = list(h_action_to_boltz_prob.values())
        # print()
        # print("h_action_to_boltz_prob", h_action_to_boltz_prob)
        # print("probs", probs)
        # if round_no < 5:
        # h_action = h_action_keys[np.random.choice(np.arange(len(h_action_keys)), p=probs)]
        # print("h_action", h_action)
        #
        # if len(best_human_act) == 0:
        #     h_action = None
        # else:
        #     # h_action = best_human_act[np.random.choice(range(len(best_human_act)))]
        #     h_action = best_human_act[0]
        #
        # print("best h_action", h_action)
        # print(f"Other {other_actions} removing {h_action}")
        # other_actions.remove(h_action)
        # r = np.random.uniform(0, 1)
        # if round_no < 5 and r < self.h_alpha:
        #     if len(other_actions) > 0:
        #         # print("Using random")
        #         h_action = other_actions[np.random.choice(np.arange(len(other_actions)))]

        return h_action


