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

CHOP_CHICKEN = 0
POUR_BROTH = 1
GET_BOWL = 2
CHOP_LETTUCE = 3
POUR_DRESSING = 4
GET_CUTTING_BOARD = 5
GET_PLATE = 6
ACTION_LIST = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD, GET_PLATE]
ACTION_TO_TEXT = {CHOP_CHICKEN: 'chop chicken', POUR_BROTH: 'pour broth', GET_BOWL: 'get bowl',
                  CHOP_LETTUCE: 'chop lettuce', POUR_DRESSING: 'pour dressing',GET_PLATE: 'get plate',
                  GET_CUTTING_BOARD: 'get cutting board', None: 'None'}
ACTION_TO_PRECONDITION = {CHOP_CHICKEN: [GET_CUTTING_BOARD], POUR_BROTH: [GET_BOWL], GET_BOWL: [], GET_PLATE: [],
                            CHOP_LETTUCE: [GET_CUTTING_BOARD], POUR_DRESSING: [], GET_CUTTING_BOARD: []}

import pickle
import json
import sys
import os



class Suboptimal_Collaborative_Human:
    def __init__(self, ind_rew, true_robot_rew, starting_actions_to_perform, h_alpha=0.0, h_deg_collab=0.5):
        self.ind_rew = ind_rew
        self.true_robot_rew = true_robot_rew
        self.h_alpha = h_alpha
        self.h_deg_collab = h_deg_collab
        if self.true_robot_rew is not None:
            self.robot_rew = copy.deepcopy(self.true_robot_rew)

        self.starting_actions_to_perform = starting_actions_to_perform

        self.possible_actions = []
        for action_r in self.starting_actions_to_perform:
            for action_h in self.starting_actions_to_perform:
                if (action_r, action_h) not in self.possible_actions:
                    self.possible_actions.append((action_r, action_h))

    def reset(self):
        self.possible_actions = []
        for action_r in self.starting_actions_to_perform:
            for action_h in self.starting_actions_to_perform:
                if (action_r, action_h) not in self.possible_actions:
                    self.possible_actions.append((action_r, action_h))


    def step_given_state(self, input_state, joint_action):
        state_actions_completed = copy.deepcopy(input_state)
        # print("joint_action", joint_action)
        robot_action = joint_action[0]

        human_action = joint_action[1]

        robot_action_successful = False
        human_action_successful = False
        team_rew, robot_rew, human_rew = -10, 0, 0

        if robot_action is not None and robot_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]
            if set(preconditions_list).issubset(state_actions_completed):
                robot_action_successful = True
                state_actions_completed.append(robot_action)
                robot_rew += self.true_robot_rew[robot_action]

        if human_action is not None and human_action not in state_actions_completed:
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed):
                human_action_successful = True
                state_actions_completed.append(human_action)
                human_rew += self.ind_rew[human_action]

        salad_actions = [GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD]
        soup_actions = [CHOP_CHICKEN, POUR_BROTH, GET_BOWL, GET_CUTTING_BOARD]
        done = False
        if set(soup_actions).issubset(state_actions_completed) or set(salad_actions).issubset(state_actions_completed):
            done = True

        if done:
            team_rew = 0
        return team_rew, robot_rew, human_rew


    def act_indist(self, state, round_no=0):
        # best_human_act = []
        other_actions = []
        max_reward = -2
        h_action = None
        beta = 1
        h_action_to_boltz_prob = {}

        for (candidate_r_act, candidate_h_act) in self.possible_actions:
            team_rew, r_rew, h_rew = self.step_given_state(state, (candidate_r_act, candidate_h_act))
            # print("(candidate_r_act, candidate_h_act)", (candidate_r_act, candidate_h_act))
            candidate_rew = team_rew + ((self.h_deg_collab) * r_rew) + ((1-self.h_deg_collab) * h_rew)
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
            if candidate_rew > max_reward:
                h_action = candidate_h_act
                max_reward = candidate_rew

            if candidate_h_act not in h_action_to_boltz_prob:
                h_action_to_boltz_prob[candidate_h_act] = -10000
            if candidate_h_act is not None:
                if candidate_h_act not in state and candidate_rew > h_action_to_boltz_prob[candidate_h_act]:
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
            candidate_rew = team_rew + ((self.h_deg_collab) * r_rew) + ((1-self.h_deg_collab) * h_rew)
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


