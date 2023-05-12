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
from robot_model_birl_rew import Robot
# from robot_model_fixed_lstm import Robot
# from robot_model_birl_prob_plan_out import Robot
# from robot_model_lstm_rew import Robot
from human_model import Greedy_Human, Collaborative_Human, Suboptimal_Collaborative_Human
# import seaborn as sns
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.stats import sem
import pickle
import json
import sys
import os
import subprocess
import glob

ROBOT_TYPE = 'robot_model_birl_rew'

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow', None:'white'}


class Simultaneous_Cleanup():
    def __init__(self, robot, human, starting_objects):
        self.robot = robot
        self.human = human

        self.total_reward = {'team': 0, 'robot': 0, 'human': 0}
        self.state_remaining_objects = {}
        self.possible_actions = [None]

        self.starting_objects = starting_objects
        self.reset()

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
        self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.0001
        self.gamma = 1.0
        self.maxiter = 10000


    def reset(self):
        self.state_remaining_objects = {}
        self.possible_actions = [None]
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.state_remaining_objects:
                self.state_remaining_objects[obj_tuple] = 1
                self.possible_actions.append(obj_tuple)
            else:
                self.state_remaining_objects[obj_tuple] += 1

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

    def resolve_heavy_pickup(self, rh_action):
        robot_action_successful, human_action_successful = False, False
        team_rew, robot_rew, human_rew = 0, 0, 0
        if rh_action in self.state_remaining_objects and self.state_remaining_objects[rh_action] > 0:
            self.state_remaining_objects[rh_action] -= 1
            # team_rew += 0
            robot_rew += self.robot.ind_rew[rh_action]
            human_rew += self.human.ind_rew[rh_action]
            robot_action_successful, human_action_successful = True, True

        self.total_reward['team'] += team_rew
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew
        return team_rew, robot_rew, human_rew, robot_action_successful, human_action_successful


    def resolve_two_agents_same_item(self, robot_action, human_action):
        robot_action_successful, human_action_successful = False, False
        (robot_action_color, robot_action_weight) = robot_action
        (human_action_color, human_action_weight) = human_action
        robot_rew, human_rew = 0, 0
        if robot_action in self.state_remaining_objects:
            if self.state_remaining_objects[robot_action] == 0:
                robot_rew, human_rew = 0, 0
            elif self.state_remaining_objects[robot_action] == 1:
                self.state_remaining_objects[robot_action] -= 1

                human_rew = self.human.ind_rew[human_action]  # Human takes object
                human_action_successful = True

                # Uncomment below if stochastic pickup.
                # pickup_agent = np.random.choice(['r', 'h'])
                # if pickup_agent == 'r':
                #     robot_rew = self.robot.ind_rew[robot_action]
                # else:
                #     human_rew = self.human.ind_rew[human_action]
            else:
                self.state_remaining_objects[robot_action] -= 1
                self.state_remaining_objects[human_action] -= 1
                robot_rew += self.robot.ind_rew[robot_action]
                human_rew += self.human.ind_rew[human_action]
                robot_action_successful, human_action_successful = True, True

        self.total_reward['team'] += 0
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew

        return 0, robot_rew, human_rew, robot_action_successful, human_action_successful

    def resolve_two_agents_diff_item(self, robot_action, human_action):

        robot_action_successful, human_action_successful = False, False
        robot_rew, human_rew = 0, 0

        if robot_action is not None and robot_action in self.state_remaining_objects:
            (robot_action_color, robot_action_weight) = robot_action

            if robot_action_weight == 0:
                if self.state_remaining_objects[robot_action] > 0:
                    self.state_remaining_objects[robot_action] -= 1
                    robot_rew += self.robot.ind_rew[robot_action]
                    robot_action_successful = True

        if human_action is not None and human_action in self.state_remaining_objects:
            (human_action_color, human_action_weight) = human_action
            if human_action_weight == 0:
                if self.state_remaining_objects[human_action] > 0:
                    self.state_remaining_objects[human_action] -= 1
                    human_rew += self.human.ind_rew[human_action]
                    human_action_successful = True

        self.total_reward['team'] += 0
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew
        return 0, robot_rew, human_rew, robot_action_successful, human_action_successful


    def step(self, joint_action):

        robot_action = joint_action['robot']

        human_action = joint_action['human']

        robot_action_successful = False
        human_action_successful = False
        # (human_action_color, human_action_weight) = human_action
        # total_reward = 0
        # total_robot_only = 0
        # total_human_only = 0

        if robot_action == human_action and robot_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                team_rew, robot_rew, human_rew, robot_action_successful, human_action_successful = self.resolve_heavy_pickup(
                    robot_action)

            # single pick up object
            else:
                team_rew, robot_rew, human_rew, robot_action_successful, human_action_successful = self.resolve_two_agents_same_item(
                    robot_action, human_action)

        else:
            team_rew, robot_rew, human_rew, robot_action_successful, human_action_successful = self.resolve_two_agents_diff_item(
                robot_action, human_action)

        team_rew = team_rew - 2

        # if robot_action is None:
        #     team_rew -= 2
        # if human_action is None:
        #     team_rew -= 2

        done = self.is_done()
        # total_reward = team_rew + robot_rew + human_rew
        return (team_rew, robot_rew, human_rew), done, (robot_action_successful, human_action_successful)


    def step_given_state(self, input_state, joint_action):
        state_remaining_objects = copy.deepcopy(input_state)


        robot_action = joint_action['robot']

        human_action = joint_action['human']

        robot_rew = 0
        human_rew = 0
        team_rew = -2

        # if robot_action is None:
        #     team_rew -= 2
        # if human_action is None:
        #     team_rew -= 2


        if robot_action == human_action and human_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                if robot_action in state_remaining_objects:
                    if robot_action in self.state_remaining_objects and state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.robot.ind_rew[robot_action]
                        human_rew += self.human.ind_rew[robot_action]

            # single pick up object
            else:
                if robot_action in state_remaining_objects:
                    if state_remaining_objects[robot_action] == 0:
                        robot_rew, human_rew = 0, 0
                    elif state_remaining_objects[robot_action] == 1:
                        state_remaining_objects[robot_action] -= 1
                        human_rew += self.human.ind_rew[human_action]
                        # pickup_agent = np.random.choice(['r', 'h'])
                        # if pickup_agent == 'r':
                        #     robot_rew += self.robot.ind_rew[robot_action]
                        # else:
                        #     human_rew += self.human.ind_rew[human_action]
                    else:
                        state_remaining_objects[robot_action] -= 1
                        state_remaining_objects[human_action] -= 1
                        robot_rew += self.robot.ind_rew[robot_action]
                        human_rew += self.human.ind_rew[human_action]

        else:
            if robot_action is not None and robot_action in state_remaining_objects:
                (robot_action_color, robot_action_weight) = robot_action
                if robot_action_weight == 0:
                    if state_remaining_objects[robot_action] > 0:
                        state_remaining_objects[robot_action] -= 1
                        robot_rew += self.robot.ind_rew[robot_action]

            if human_action is not None and human_action in state_remaining_objects:
                (human_action_color, human_action_weight) = human_action
                if human_action_weight == 0:
                    if state_remaining_objects[human_action] > 0:
                        state_remaining_objects[human_action] -= 1
                        human_rew += self.human.ind_rew[human_action]

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

                next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(state, action)

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

    def value_iteration(self):
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
            print("i=", i)
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
                    if obj_tuple not in current_state_remaining_objects:
                        current_state_remaining_objects[obj_tuple] = 0

                if len(current_state_remaining_objects) == 0 or sum(current_state_remaining_objects.values()) == 0:
                    # for action_idx in range(n_actions):
                    action_idx = self.action_to_idx[(None, None)]
                    Q[s, action_idx] = vf[s]

                else:
                    # compute new Q values
                    for action_idx in range(n_actions):
                        # check joint action
                        joint_action = self.idx_to_action[action_idx]
                        joint_action = {'robot':joint_action[0], 'human':joint_action[1]}

                        # print("current_state_remaining_objects = ", current_state_remaining_objects)
                        # print("joint_action = ", joint_action)
                        next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(current_state_remaining_objects, joint_action)
                        # print(f"current_state = ", current_state_remaining_objects)
                        # print("action=  ", joint_action)
                        # print("r_sa = ", r_sa)
                        # print("next_state = ", next_state)
                        # print("done = ", done)
                        # print()

                        r_sa = team_rew + robot_rew + human_rew
                        s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                        Q[s, action_idx] = r_sa + (self.gamma * vf[s11])

                vf[s] = np.max(Q[s, :], 0)

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # check for convergence
            if delta < self.epsilson:
                print("DONE at iteration ", i)
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

            # compute new Q values
            for action_idx in range(n_actions):
                # check joint action
                joint_action = self.idx_to_action[action_idx]
                joint_action = {'robot': joint_action[0], 'human': joint_action[1]}

                next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(current_state_remaining_objects,
                                                                                       joint_action)

                r_sa = team_rew + robot_rew + human_rew
                s11 = self.state_to_idx[self.state_to_tuple(next_state)]
                Q[s, action_idx] = r_sa + (self.gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def rollout_full_game_joint_optimal(self):
        self.reset()
        done = False
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        human_trace = []
        robot_trace = []
        human_greedy_alt = []
        robot_greedy_alt = []
        iters = 0

        while not done:
            iters += 1
        # for i in range(10):
            current_state = copy.deepcopy(self.state_remaining_objects)
            # print(f"current_state = {current_state}")
            current_state_tup = self.state_to_tuple(current_state)

            # print("availabel actions", self.get_possible_actions(current_state))
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # print("action_distribution = ", action_distribution)
            # print("action", action)

            action = {'robot': action[0], 'human': action[1]}

            next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(current_state, action)
            self.state_remaining_objects = next_state

            human_only_reward += human_rew
            robot_only_reward += robot_rew
            # print(f"current_state = ", current_state)
            # print("action=  ", action)
            # print("team_rew = ", team_rew)
            # print("next_state = ", next_state)
            # print("done = ", done)
            # print()


            # print(
            # f"current_state= {current_state}, next_state={next_state}, rew={rew}, is done = {done}")
            print(f"team = {team_rew}, robot={robot_rew}, human = {human_rew}, total = {team_rew + robot_rew + human_rew}")
            total_reward += (team_rew + robot_rew + human_rew)

            if iters > 100:
                break

        return total_reward, human_only_reward, robot_only_reward

    def rollout_full_game_two_agents(self):
        self.reset()
        done = False
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        iters = 0
        is_start = True

        while not done:
            iters += 1
            current_state = copy.deepcopy(self.state_remaining_objects)

            robot_action = self.robot.act(current_state)
            # is_start = False
            human_action = self.human.act(current_state)

            action = {'robot': robot_action, 'human': human_action}

            (team_rew, robot_rew, human_rew), done, (_,_) = self.step(action)
            human_only_reward += human_rew
            robot_only_reward += robot_rew
            # print(f"current_state = ", current_state)
            # print("action=  ", action)
            # print("team_rew = ", team_rew)
            # print("next_state = ", self.state_remaining_objects)
            # print("done = ", done)
            # print()
            total_reward += (team_rew + robot_rew + human_rew)

            if iters > 100:
                # print("Cannot finish")
                break

        return total_reward, human_only_reward, robot_only_reward

    def compute_optimal_performance(self):
        print("start enumerating states")
        self.enumerate_states()
        print("done enumerating states")
        print("start vi")
        self.value_iteration()
        print("done vi")

        optimal_rew, human_only_reward, robot_only_reward = self.rollout_full_game_joint_optimal()
        return optimal_rew, human_only_reward, robot_only_reward

    def rollout_multiround_game_two_agents(self, num_rounds, plot=False):
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        multiround_belief_history = {}
        reward_for_all_rounds = []

        # print(f"robot rew = {self.robot.ind_rew}")
        # print(f"human rew = {self.human.ind_rew}")
        # print(f"robot rew = {self.robot.ind_rew}")

        for round_no in range(num_rounds):
            print(f"\n\nRound = {round_no}")
            if type(self.robot) == Robot:
                self.robot.setup_value_iteration()
            if type(self.human) == Robot:
                self.human.setup_value_iteration()

            self.robot.reset_belief_history()

            self.reset()
            total_reward = 0
            human_only_reward = 0
            robot_only_reward = 0

            done = False
            total_reward = 0
            human_only_reward = 0
            robot_only_reward = 0
            iters = 0

            if plot:
                plt.figure()
                plt.ylim(0, 3)
                plt.yticks([1,2], ['H', 'R'], color='black',fontweight='bold', fontsize='17')

            # plt.xlim(0, 1)
            # Human y val = 1
            # Robot y val = 2
            # Round No on X axis
            human_y = 1
            robot_y = 2

            human_rewards_over_round = []
            robot_rewards_over_round = []
            team_rewards_over_round = []
            total_rewards_over_round = []

            is_start = True

            while not done:
                iters += 1
                current_state = copy.deepcopy(self.state_remaining_objects)

                robot_action = self.robot.act(current_state, is_start=is_start)
                is_start = False
                # print("current_state for human acting", current_state)
                human_action = self.human.act(current_state)

                if hasattr(self.robot, 'human_lstm'):
                    print("LSTM ROBOT")
                    self.robot.add_to_lstm_training_data(current_state, robot_action, human_action)
                else:
                    print("NOT LSTM ROBOT")

                max_key = max(self.robot.beliefs, key=lambda k: self.robot.beliefs[k]['prob'])
                # print()
                # print("max prob belief", self.robot.beliefs[max_key]['reward_dict'])
                # print("true robot rew", self.robot.ind_rew)
                # print("true human rew", self.human.ind_rew)

                action = {'robot': robot_action, 'human': human_action}
                # print(f"iter {iters}: objects left {sum(self.state_remaining_objects.values())} --> action {action}")

                (team_rew, robot_rew, human_rew), done, (robot_action_successful, human_action_successful) = self.step(action)

                human_rewards_over_round.append(human_rew)
                robot_rewards_over_round.append(robot_rew)
                team_rewards_over_round.append(team_rew)
                total_rewards_over_round.append(team_rew + human_rew + robot_rew)

                # pdb.set_trace()
                if plot:
                    if human_action is not None:
                        if human_action_successful is False:
                            plt.scatter([iters - 1], [human_y], c=COLOR_TO_TEXT[human_action[0]],
                                        s=np.power(3,human_action[1] + 4), alpha=0.2)
                        else:
                            plt.scatter([iters - 1], [human_y], c=COLOR_TO_TEXT[human_action[0]],
                                        s=np.power(3,human_action[1] + 4))
                    else:
                        plt.scatter([iters - 1], [human_y], c='k', s=2)

                    if robot_action is not None:
                        if robot_action_successful is False:
                            plt.scatter([iters - 1], [robot_y], c=COLOR_TO_TEXT[robot_action[0]],
                                        s=np.power(3,robot_action[1] + 4), alpha=0.2)
                        else:
                            plt.scatter([iters - 1], [robot_y], c=COLOR_TO_TEXT[robot_action[0]], s=np.power(3,robot_action[1] + 4))
                    else:
                        plt.scatter([iters - 1], [robot_y], c='k', s=2)

                if type(self.robot) == Robot:
                    self.robot.update_based_on_h_action(current_state, robot_action, human_action)

                human_only_reward += human_rew
                robot_only_reward += robot_rew

                # print(f"current_state = ", current_state)
                # print("action=  ", action)
                # print("team_rew = ", team_rew)
                # print("next_state = ", self.state_remaining_objects)
                # print("done = ", done)
                # print()
                total_reward += (team_rew + human_rew + robot_rew)
                # print(f"team rew = {team_rew}, human rew = {human_rew}, robot rew = {robot_rew} --> total rew = {total_reward}\n")


                if iters > 10:
                    # print("Cannot finish")
                    break

            multiround_belief_history[round_no] = self.robot.history_of_human_beliefs
            reward_for_all_rounds.append(total_reward)

            if plot:
                plt.title(f"Game Replay Round {round_no}")
                plt.savefig(f"images/game_replay_r{round_no}.png")
                plt.show()

                plt.figure()
                plt.plot(range(len(human_rewards_over_round)), human_rewards_over_round, color='g', label='human', linewidth=7.0, alpha=0.5)
                plt.plot(range(len(robot_rewards_over_round)), robot_rewards_over_round, color='m', label='robot', linewidth=2.0, alpha=1)
                plt.plot(range(len(team_rewards_over_round)), team_rewards_over_round, color='k', label='team')
                plt.plot(range(len(total_rewards_over_round)), total_rewards_over_round, color='b', label='total')
                plt.legend()
                plt.xlabel("Timestep")
                plt.ylabel("Reward")
                plt.title(f"Reward Earned in Round {round_no}")
                plt.savefig(f"images/round_reward_r{round_no}.png")
                plt.show()

                true_rew_belief = self.robot.history_of_robot_beliefs_of_true_human_rew
                plt.plot(range(len(true_rew_belief)), true_rew_belief)
                plt.xlabel("Iteration")
                plt.ylabel("Weight")
                plt.title(f"Belief of True Reward in Round {round_no}")
                plt.savefig(f"images/belief_history_r{round_no}.png")
                plt.show()

                plt.figure()
                item_weights = {}
                for i in range(len(self.robot.history_of_robot_beliefs_of_max_human_rew)):
                    reward_dict = self.robot.history_of_robot_beliefs_of_max_human_rew[i]
                    for object in reward_dict:
                        if object not in item_weights:
                            item_weights[object] = []
                        item_weights[object].append(reward_dict[object])

                for object in item_weights:
                    weights = item_weights[object]
                    obj_color = object[0]
                    obj_size = object[1]

                    line = 3.0
                    alpha = 1.0

                    if obj_size == 1:
                        line = 7.0
                        alpha = 0.3

                    color = COLOR_TO_TEXT[obj_color]

                    plt.plot(range(len(weights)), weights, color=color, label=object,
                             linewidth=line, alpha=alpha)

                plt.xlabel("Iteration")
                plt.ylabel("Weight")
                plt.title(f"Belief of Max Reward in Round {round_no}")
                plt.savefig(f"images/max_reward_beliefs_r{round_no}.png")
                plt.show()


                plt.figure()
                for object in self.robot.ind_rew:
                    weights = [self.robot.ind_rew[object]]*num_rounds
                    obj_color = object[0]
                    obj_size = object[1]

                    line = 3.0
                    alpha = 0.6

                    if obj_size == 1:
                        line = 9.0
                        alpha = 0.2

                    color = COLOR_TO_TEXT[obj_color]

                    plt.plot(range(len(weights)), weights, color=color, label=object,
                             linewidth=line, alpha=alpha)
                plt.xlabel("Iteration")
                plt.ylabel("Weight")
                plt.title(f"Robot True Reward in Round {round_no}")
                plt.savefig(f"images/true_robot_reward_r{round_no}.png")
                plt.show()

                plt.figure()
                for object in self.human.ind_rew:
                    weights = [self.human.ind_rew[object]] * num_rounds
                    obj_color = object[0]
                    obj_size = object[1]

                    line = 3.0
                    alpha = 1.0

                    if obj_size == 1:
                        line = 7.0
                        alpha = 0.3

                    color = COLOR_TO_TEXT[obj_color]

                    plt.plot(range(len(weights)), weights, color=color, label=object,
                             linewidth=line, alpha=alpha)
                plt.xlabel("Iteration")
                plt.ylabel("Weight")
                plt.title(f"Human True Reward in Round {round_no}")
                plt.savefig(f"images/true_human_reward_r{round_no}.png")
                plt.show()


        max_prob_idx = 0
        second_prob_idx = 0
        second_largest_prob = -100
        max_prob = -100
        for idx in self.robot.beliefs:
            if self.robot.beliefs[idx]['prob'] > max_prob:
                max_prob = self.robot.beliefs[idx]['prob']
                max_prob_idx = idx
                second_largest_prob = max_prob
                second_prob_idx = max_prob_idx

            elif self.robot.beliefs[idx]['prob'] > second_largest_prob:
                second_largest_prob = self.robot.beliefs[idx]['prob']
                second_prob_idx = idx

        max_prob_is_correct = False
        if self.robot.beliefs[max_prob_idx]['prob'] == self.robot.beliefs[self.robot.true_human_rew_idx]['prob']:
            max_prob_is_correct = True

        max_prob_is_close = False
        if self.robot.beliefs[max_prob_idx]['prob'] == self.robot.beliefs[self.robot.true_human_rew_idx]['prob'] or second_prob_idx == self.robot.true_human_rew_idx or abs(
                max_prob - self.robot.beliefs[self.robot.true_human_rew_idx]['prob']) <= 0.2:
            max_prob_is_close = True

        num_equal_to_max = 0
        for idx in self.robot.beliefs:
            if self.robot.beliefs[idx]['prob'] == max_prob:
                num_equal_to_max += 1

        if hasattr(self.robot, 'lstm_accuracies'):
            lstm_accuracies_list = self.robot.lstm_accuracies
        else:
            lstm_accuracies_list = None

        return total_reward, human_only_reward, robot_only_reward, multiround_belief_history, reward_for_all_rounds, max_prob_is_correct, max_prob_is_close, num_equal_to_max, lstm_accuracies_list




def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': -4, 'left': 4}

    for rect in rects:
        height = rect.get_height()
        # print("height",height)
        # height = [np.round(x,2) for x in height]
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + 0.08),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=14)


def plot_multiround_belief_history(multiround_belief_history, all_objects):
    all_history = []
    for round_no in multiround_belief_history:
        for t in range(len(multiround_belief_history[round_no])):
            weighted_beliefs = {elem: 0 for elem in all_objects}
            beliefs = multiround_belief_history[round_no][t]
            for belief_idx in beliefs:
                h_reward_hypothesis = beliefs[belief_idx]['reward_dict']
                probability_of_hyp = beliefs[belief_idx]['prob']
                for obj_key in h_reward_hypothesis:
                    weighted_beliefs[obj_key] += h_reward_hypothesis[obj_key] * probability_of_hyp
            all_history.append(weighted_beliefs)

    print(all_history)

    folder = 'imgs_for_video'

    for i in range(len(all_history)):
        belief_dict = all_history[i]
        keys = list(belief_dict.keys())
        values = list(belief_dict.values())
        # Data set
        height = values
        bars = keys
        y_pos = np.arange(len(bars))

        # Basic bar plot
        plt.bar(y_pos, height)

        plt.xticks(y_pos, labels=keys)

        # Custom Axis title
        plt.xlabel('Object Type')
        plt.ylabel('Reward')

        # Show the graph

        plt.savefig(folder + "/file%02d.png" % i)
        plt.close()

    # os.chdir("your_folder")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', folder + '/file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'history_of_beliefs_test_fail.mp4'
    ])
    for file_name in glob.glob(folder + "*.png"):
        os.remove(folder + file_name)


def run_k_rounds(exp_num, task_reward, h_alpha=0.0, update_threshold=0.9):
    print("exp_num = ", exp_num)
    # np.random.seed(1)

    # robot_rew = {
                # (BLUE, 0): np.random.randint(3,10),
                #  (RED, 0): np.random.randint(3,10),
                 # (GREEN, 0): np.random.randint(3,10),
                 # (YELLOW, 0): np.random.randint(3,10),
                 # (BLUE, 1): np.random.randint(3,10),
                 # (RED, 1): np.random.randint(3,10),
                 # (GREEN, 1): np.random.randint(3,10),
                 # (YELLOW, 1): np.random.randint(3,10)}
    # robot_rew = {(BLUE, 0): 0,
    #              (RED, 0): 0,
    #              (BLUE, 1): 0,
    #              (RED, 1): 0}

    robot_rew = {
                (BLUE, 0): np.random.randint(3, 10),
                 (RED, 0): np.random.randint(3, 10),
                 (BLUE, 1): np.random.randint(3, 10),
                 (RED, 1): np.random.randint(3, 10),
                 # (YELLOW, 1): np.random.randint(3,10)
                }

    team_rew = {
                (BLUE, 0): -1,
                 (RED, 0): -1,
                 (GREEN, 0): -1,
                 (YELLOW, 0): -1,
                 (BLUE, 1): -1,
                 (RED, 1): -1,
                 (GREEN, 1): -1,
                 (YELLOW, 1): -1,
                None: -1}


    # robot_rew_values =

    # human_rew = {(BLUE, 0): np.random.randint(1, 10),
    #              (RED, 0): np.random.randint(1, 10),
    #              (GREEN, 0): np.random.randint(1, 10),
    #              (YELLOW, 0): np.random.randint(1, 10),
    #              (BLUE, 1): np.random.randint(1, 10),
    #              (RED, 1): np.random.randint(1, 10),
    #              (GREEN, 1): np.random.randint(1, 10),
    #              (YELLOW, 1): np.random.randint(1, 10)}


    permutes = list(itertools.permutations(list(robot_rew.values())))
    permutes = list(set(permutes))
    human_rew_values = list(permutes[np.random.choice(np.arange(len(permutes)))])
    object_keys = list(robot_rew.keys())
    human_rew = {object_keys[i]: human_rew_values[i] for i in range(len(object_keys))}

    random_h_alpha = np.random.uniform(0.5, 1.0)
    random_h_deg_collab = np.random.uniform(0.1, 1.0)
    # random_h_alpha = h_alpha
    # random_h_deg_collab = 0.5

    # all_objects = [(BLUE, 0), (YELLOW, 0), (GREEN, 0), (RED, 0), (BLUE, 1),(GREEN, 1), (RED, 1), (YELLOW, 1)]
    # all_objects = [(BLUE, 0),  (RED, 0), (BLUE, 1), (RED, 1), (YELLOW, 1)]
    # all_objects = [(RED, 0), (BLUE, 1), (RED, 1), (YELLOW, 1)]
    # all_objects = [(BLUE, 1), (GREEN, 1), (RED, 1), (YELLOW, 1)]
    all_objects = [(BLUE, 0), (RED, 0), (BLUE, 1), (RED, 1)]

    n_objects = np.random.randint(4, 10)
    # starting_objects = [all_objects[i] for i in np.random.choice(np.arange(len(all_objects)), size=n_objects, replace=True)]
    starting_objects = []
    for object in all_objects:
        count = np.random.randint(1, 4)
        for c in range(count):
            starting_objects.append(object)
    # starting_objects = [(BLUE, 0), (RED, 0), (BLUE, 1), (RED, 1),(BLUE, 0), (RED, 0), (BLUE, 1), (RED, 1)]
    #
    # print("human_rew", human_rew)
    # print("robot_rew", robot_rew)
    # print("starting_objects", starting_objects)


    robot = Robot(team_rew, robot_rew, human_rew, starting_objects, robot_knows_human_rew=True, permutes=permutes, vi_type='cvi', is_collaborative_human=True)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha, h_deg_collab=random_h_deg_collab)
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    optimal_rew, best_human_rew, best_robot_rew = env.compute_optimal_performance()
    print("Optimal Reward = ", optimal_rew)

    # Test 2 greedy
    robot = Greedy_Human(robot_rew, human_rew, starting_objects, 0)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha, h_deg_collab=random_h_deg_collab)
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    greedy_team_rew, greedy_human_rew, greedy_robot_rew = env.rollout_full_game_two_agents()
    print("Fully 2 greedy final_team_rew = ", greedy_team_rew)

    robot = Robot(team_rew, robot_rew, human_rew, starting_objects, robot_knows_human_rew=False, permutes=permutes, vi_type='cvi', is_collaborative_human=True, update_threshold=update_threshold)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha, h_deg_collab=random_h_deg_collab)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    cvi_rew, cvi_human_rew, cvi_robot_rew, multiround_belief_history, reward_for_all_rounds, max_prob_is_correct, max_prob_is_close, num_equal_to_max, lstm_accuracies_list = env.rollout_multiround_game_two_agents(num_rounds=6, plot=False)
    print("CVI final_team_rew = ", cvi_rew)

    robot = Robot(team_rew, robot_rew, human_rew, starting_objects, robot_knows_human_rew=True, permutes=permutes, vi_type='stdvi', is_collaborative_human=True)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha, h_deg_collab=random_h_deg_collab)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    stdvi_rew, stdvi_human_rew, stdvi_robot_rew = env.rollout_full_game_two_agents()
    print("StdVI final_team_rew = ", stdvi_rew)


    # plot_multiround_belief_history(multiround_belief_history, all_objects)


    altruism_case = 'opt'
    if greedy_team_rew < optimal_rew:
        altruism_case = 'subopt'

    if cvi_rew > optimal_rew:
        print("PROBLEM CVI larger than OPT")
        print("optimal_rew", optimal_rew)
        print("cvi_rew = ", cvi_rew)
        print("starting_objects = ", starting_objects)
        print("human reward = ", human_rew)
        print("robot_rew = ", robot_rew)
        print()

    percent_opt_each_round = []
    for j in range(len(reward_for_all_rounds)):
        percent_opt_each_round.append(max(0,reward_for_all_rounds[j]) / optimal_rew)

    if cvi_rew > optimal_rew:
        print("OPTIMAL ERROR: CVI rew larger than Optimal")
        raise ArithmeticError

    if stdvi_rew > optimal_rew:
        print("OPTIMAL ERROR: STDVI rew larger than Optimal")
        raise ArithmeticError

    cvi_percent_of_opt_team = max(0,cvi_rew) / optimal_rew
    stdvi_percent_of_opt_team = max(0,stdvi_rew) / optimal_rew

    cvi_percent_of_opt_team = cvi_percent_of_opt_team
    stdvi_percent_of_opt_team = stdvi_percent_of_opt_team

    # cvi_percent_of_opt_human = max(0,cvi_human_rew) / best_human_rew
    # stdvi_percent_of_opt_human = max(0,stdvi_human_rew) / best_human_rew
    #
    # cvi_percent_of_opt_robot = max(0,cvi_robot_rew) / best_robot_rew
    # stdvi_percent_of_opt_robot = max(0,stdvi_robot_rew) / best_robot_rew
    cvi_percent_of_opt_human, stdvi_percent_of_opt_human, cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot = 0, 0, 0, 0

    print("done with exp_num = ", exp_num)
    return cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
           cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal_to_max, lstm_accuracies_list


def run_experiment():

    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 8

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    round_to_percent_rewards = {i:[] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    timestep_to_accuracy_list = {}

    with Pool(processes=num_exps) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal, lstm_accuracies_list = result

            for timestep in range(len(lstm_accuracies_list)):
                if timestep not in timestep_to_accuracy_list:
                    timestep_to_accuracy_list[timestep] = []
                timestep_to_accuracy_list[timestep].append(lstm_accuracies_list[timestep])

            num_equal_to_max.append(num_equal)

            if max_prob_is_correct is True:
                times_max_prob_is_correct += 1

            if max_prob_is_close is True:
                times_max_prob_is_close += 1


            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1

            for j in range(len(percent_opt_each_round)):
                round_to_percent_rewards[j].append(percent_opt_each_round[j])

            # if altruism_case == 'opt':
            #     continue

            cvi_percents.append(cvi_percent_of_opt_team)
            stdvi_percents.append(stdvi_percent_of_opt_team)

            cvi_humanrew_percents.append(cvi_percent_of_opt_human)
            stdvi_humanrew_percents.append(stdvi_percent_of_opt_human)

            cvi_robotrew_percents.append(cvi_percent_of_opt_robot)
            stdvi_robotrew_percents.append(stdvi_percent_of_opt_robot)

            diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
            diff = np.round(diff, 2)
            # print("percent_change = ", percent_change)
            if diff in percent_change:
                percent_change[diff] += 1

    teamrew_means = [np.round(np.mean(cvi_percents), 2), np.round(np.mean(stdvi_percents), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents), 2), np.round(np.std(stdvi_percents), 2)]

    # humanrew_means = [np.round(np.mean(cvi_humanrew_percents), 2), np.round(np.mean(stdvi_humanrew_percents), 2)]
    # humanrew_stds = [np.round(np.std(cvi_humanrew_percents), 2), np.round(np.std(stdvi_humanrew_percents), 2)]
    #
    # robotrew_means = [np.round(np.mean(cvi_robotrew_percents), 2), np.round(np.mean(stdvi_robotrew_percents), 2)]
    # robotrew_stds = [np.round(np.std(cvi_robotrew_percents), 2), np.round(np.std(stdvi_robotrew_percents), 2)]

    print("team rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_percents], [elem * 100 for elem in stdvi_percents]))
    # print("human rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents], [elem * 100 for elem in stdvi_humanrew_percents]))
    # print("robot rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents],
    #                       [elem * 100 for elem in cvi_robotrew_percents]))

    # print("n_altruism = ", n_altruism)
    # print("n_greedy = ", n_greedy)
    # print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d]/sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('CIRL', fontsize=16)
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.xlabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_cdf.png")
    # plt.show()
    plt.close()

    data = list([cvi_percents, stdvi_percents])
    fig, ax = plt.subplots(figsize=(5, 5))
    # build a violin plot
    ax.violinplot(data, showmeans=False, showmedians=True)
    # add title and axis labels
    ax.set_title('CIRL w RC', fontsize=16)
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # add x-tick labels
    xticklabels = ['CVI robot', 'StdVI robot']
    ax.set_xticks([1, 2])
    ax.set_xticklabels(xticklabels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add horizontal grid lines
    ax.yaxis.grid(True)
    # show the plot
    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_violin.png")
    # plt.show()
    plt.close()




    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    ind = np.arange(len(teamrew_means))  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 5))
    rects1 = ax.bar(ind - width, teamrew_means, width, yerr=teamrew_stds,
                    label='Team Reward', capsize=10)
    # rects2 = ax.bar(ind, humanrew_means, width, yerr=humanrew_stds,
    #                 label='Human Reward', capsize=10)
    # rects3 = ax.bar(ind + width, robotrew_means, width, yerr=robotrew_stds,
    #                 label='Robot Reward', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # ax.set_ylim(-0.00, 1.5)

    plt.yticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0], [0.0,0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # plt.xticks([])

    ax.set_title('CIRL', fontsize=16)
    ax.set_xticks(ind, fontsize=14)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'), fontsize=13)
    ax.legend(fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    autolabel(ax, rects1, "left")
    # autolabel(ax, rects2, "right")
    # autolabel(ax, rects3, "right")

    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_bar.png")
    # plt.show()
    plt.close()

    X = [i for i in np.arange(6)]
    Y = np.array([np.mean(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    Ystd = np.array([np.std(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    plt.title('Interactive IRL w RC', fontsize=16)
    plt.plot(X, Y, 'k-')
    # plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#1B2ACC', facecolor='#1B2ACC')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.ylabel("Percent of Optimal")
    plt.xlabel("Episode")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_by_round_Std.png")
    plt.close()

    X = [t for t in timestep_to_accuracy_list]
    Y = np.array([np.mean(timestep_to_accuracy_list[t]) for t in timestep_to_accuracy_list])
    Ystd = np.array([np.std(timestep_to_accuracy_list[t]) for t in timestep_to_accuracy_list])
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Timestep")
    plt.ylabel("Avg Prediction Accuracy")
    plt.title("LSTM Accuracy vs Timestep")
    plt.savefig(f"images/cirl_w_rc_{num_exps}_by_round_lstm_accuracy.png")
    plt.close()

    print()
    print("times_max_prob_is_correct = ", times_max_prob_is_correct)
    print("percent max_prob_is_correct = ", times_max_prob_is_correct/num_exps)

    print("times_max_prob_is_close = ", times_max_prob_is_close)
    print("percent max_prob_is_close = ", times_max_prob_is_close / num_exps)

    print(f"num_equal_to_max = {np.mean(num_equal_to_max)}, std: {np.std(num_equal_to_max)}")

    print("CVI Mean Percent of Opt reward = ", np.round(np.mean(cvi_percents), 3))
    print("CVI Std Percent of Opt reward = ", np.round(np.std(cvi_percents), 3))

def run_experiment_without_multiprocess():

    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 100

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    round_to_percent_rewards = {i:[] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    timestep_to_accuracy_list = {}

    for exp_num in range(num_exps):
        exp_num_results = run_k_rounds(exp_num, task_reward)
        cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
        cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal, lstm_accuracies_list = exp_num_results

        if lstm_accuracies_list is not None:
            for timestep in range(len(lstm_accuracies_list)):
                if timestep not in timestep_to_accuracy_list:
                    timestep_to_accuracy_list[timestep] = []
                timestep_to_accuracy_list[timestep].append(lstm_accuracies_list[timestep])

        num_equal_to_max.append(num_equal)

        if max_prob_is_correct is True:
            times_max_prob_is_correct += 1

        if max_prob_is_close is True:
            times_max_prob_is_close += 1


        if altruism_case == 'opt':
            n_greedy += 1
        if altruism_case == 'subopt':
            n_altruism += 1
        n_total += 1

        for j in range(len(percent_opt_each_round)):
            round_to_percent_rewards[j].append(percent_opt_each_round[j])

        # if altruism_case == 'opt':
        #     continue

        cvi_percents.append(cvi_percent_of_opt_team)
        stdvi_percents.append(stdvi_percent_of_opt_team)

        cvi_humanrew_percents.append(cvi_percent_of_opt_human)
        stdvi_humanrew_percents.append(stdvi_percent_of_opt_human)

        cvi_robotrew_percents.append(cvi_percent_of_opt_robot)
        stdvi_robotrew_percents.append(stdvi_percent_of_opt_robot)

        diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
        diff = np.round(diff, 2)
        # print("percent_change = ", percent_change)
        if diff in percent_change:
            percent_change[diff] += 1

    teamrew_means = [np.round(np.mean(cvi_percents), 2), np.round(np.mean(stdvi_percents), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents), 2), np.round(np.std(stdvi_percents), 2)]

    # humanrew_means = [np.round(np.mean(cvi_humanrew_percents), 2), np.round(np.mean(stdvi_humanrew_percents), 2)]
    # humanrew_stds = [np.round(np.std(cvi_humanrew_percents), 2), np.round(np.std(stdvi_humanrew_percents), 2)]
    #
    # robotrew_means = [np.round(np.mean(cvi_robotrew_percents), 2), np.round(np.mean(stdvi_robotrew_percents), 2)]
    # robotrew_stds = [np.round(np.std(cvi_robotrew_percents), 2), np.round(np.std(stdvi_robotrew_percents), 2)]
    print(f"ROBOT TYPE = {ROBOT_TYPE}")
    print("team rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_percents], [elem * 100 for elem in stdvi_percents]))
    # print("human rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents], [elem * 100 for elem in stdvi_humanrew_percents]))
    # print("robot rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents],
    #                       [elem * 100 for elem in cvi_robotrew_percents]))

    # print("n_altruism = ", n_altruism)
    # print("n_greedy = ", n_greedy)
    # print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d]/sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('CIRL', fontsize=16)
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.xlabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_{ROBOT_TYPE}_cdf.png")
    # plt.show()
    plt.close()

    data = list([cvi_percents, stdvi_percents])
    fig, ax = plt.subplots(figsize=(5, 5))
    # build a violin plot
    ax.violinplot(data, showmeans=False, showmedians=True)
    # add title and axis labels
    ax.set_title('CIRL w RC', fontsize=16)
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # add x-tick labels
    xticklabels = ['CVI robot', 'StdVI robot']
    ax.set_xticks([1, 2])
    ax.set_xticklabels(xticklabels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add horizontal grid lines
    ax.yaxis.grid(True)
    # show the plot
    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_{ROBOT_TYPE}_violin.png")
    # plt.show()
    plt.close()




    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    ind = np.arange(len(teamrew_means))  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 5))
    rects1 = ax.bar(ind - width, teamrew_means, width, yerr=teamrew_stds,
                    label='Team Reward', capsize=10)
    # rects2 = ax.bar(ind, humanrew_means, width, yerr=humanrew_stds,
    #                 label='Human Reward', capsize=10)
    # rects3 = ax.bar(ind + width, robotrew_means, width, yerr=robotrew_stds,
    #                 label='Robot Reward', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # ax.set_ylim(-0.00, 1.5)

    plt.yticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0], [0.0,0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # plt.xticks([])

    ax.set_title('CIRL', fontsize=16)
    ax.set_xticks(ind, fontsize=14)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'), fontsize=13)
    ax.legend(fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    autolabel(ax, rects1, "left")
    # autolabel(ax, rects2, "right")
    # autolabel(ax, rects3, "right")

    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_{ROBOT_TYPE}_bar.png")
    # plt.show()
    plt.close()

    X = [i for i in np.arange(6)]
    Y = np.array([np.mean(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    Ystd = np.array([np.std(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    plt.title('Interactive IRL w RC', fontsize=16)
    plt.plot(X, Y, 'k-')
    # plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#1B2ACC', facecolor='#1B2ACC')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.ylabel("Percent of Optimal")
    plt.xlabel("Episode")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_{ROBOT_TYPE}_by_round_Std.png")
    # plt.show()
    plt.close()

    X = [t for t in timestep_to_accuracy_list]
    Y = np.array([np.mean(timestep_to_accuracy_list[t]) for t in timestep_to_accuracy_list])
    Ystd = np.array([np.std(timestep_to_accuracy_list[t]) for t in timestep_to_accuracy_list])
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Timestep")
    plt.ylabel("Avg Prediction Accuracy")
    plt.title("LSTM Accuracy vs Timestep")
    plt.savefig(f"images/cirl_w_rc_{num_exps}_{ROBOT_TYPE}_by_round_lstm_accuracy.png")
    # plt.show()
    plt.close()

    print()
    print("times_max_prob_is_correct = ", times_max_prob_is_correct)
    print("percent max_prob_is_correct = ", times_max_prob_is_correct/num_exps)

    print("times_max_prob_is_close = ", times_max_prob_is_close)
    print("percent max_prob_is_close = ", times_max_prob_is_close / num_exps)

    print(f"num_equal_to_max = {np.mean(num_equal_to_max)}, std: {np.std(num_equal_to_max)}")

    print("CVI Mean Percent of Opt reward = ", np.round(np.mean(cvi_percents), 3))
    print("CVI Std Percent of Opt reward = ", np.round(np.std(cvi_percents), 3))


def run_experiment_for_update_threshold(update_threshold):

    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 5

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    round_to_percent_rewards = {i:[] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    with Pool(processes=5) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward, 0.0, update_threshold) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal = result
            num_equal_to_max.append(num_equal)
            if max_prob_is_correct is True:
                times_max_prob_is_correct += 1

            if max_prob_is_close is True:
                times_max_prob_is_close += 1


            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1

            for j in range(len(percent_opt_each_round)):
                round_to_percent_rewards[j].append(percent_opt_each_round[j])

            # if altruism_case == 'opt':
            #     continue

            cvi_percents.append(cvi_percent_of_opt_team)
            stdvi_percents.append(stdvi_percent_of_opt_team)

            cvi_humanrew_percents.append(cvi_percent_of_opt_human)
            stdvi_humanrew_percents.append(stdvi_percent_of_opt_human)

            cvi_robotrew_percents.append(cvi_percent_of_opt_robot)
            stdvi_robotrew_percents.append(stdvi_percent_of_opt_robot)

            diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
            diff = np.round(diff, 2)
            # print("percent_change = ", percent_change)
            if diff in percent_change:
                percent_change[diff] += 1


    return np.round(np.mean(cvi_percents), 3), np.round(np.std(cvi_percents), 3), times_max_prob_is_correct/num_exps, times_max_prob_is_close / num_exps, np.mean(num_equal_to_max), np.std(num_equal_to_max)

def run_experiment_for_human_alpha(human_alpha):

    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 5

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    round_to_percent_rewards = {i:[] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    with Pool(processes=5) as pool:
        update_threshold = 0.9
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward, human_alpha, update_threshold) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal = result
            num_equal_to_max.append(num_equal)
            if max_prob_is_correct is True:
                times_max_prob_is_correct += 1

            if max_prob_is_close is True:
                times_max_prob_is_close += 1


            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1

            for j in range(len(percent_opt_each_round)):
                round_to_percent_rewards[j].append(percent_opt_each_round[j])

            # if altruism_case == 'opt':
            #     continue

            cvi_percents.append(cvi_percent_of_opt_team)
            stdvi_percents.append(stdvi_percent_of_opt_team)

            cvi_humanrew_percents.append(cvi_percent_of_opt_human)
            stdvi_humanrew_percents.append(stdvi_percent_of_opt_human)

            cvi_robotrew_percents.append(cvi_percent_of_opt_robot)
            stdvi_robotrew_percents.append(stdvi_percent_of_opt_robot)

            diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
            diff = np.round(diff, 2)
            # print("percent_change = ", percent_change)
            if diff in percent_change:
                percent_change[diff] += 1


    return np.round(np.mean(cvi_percents), 3), np.round(np.std(cvi_percents), 3), times_max_prob_is_correct/num_exps, times_max_prob_is_close / num_exps, np.mean(num_equal_to_max), np.std(num_equal_to_max)


def evaluate_thresholds():
    np.random.seed(0)
    percent_opt_means = []
    percent_opt_stds = []
    num_times_correct = []
    num_times_close = []
    num_equal_to_max_means = []
    num_equal_to_max_stds = []

    updates = [0.6, 0.7, 0.8, 0.9, 1.0]

    for update_threshold in updates:
        mean_percent_rew_of_opt, std_percent_rew_of_opt, percent_correct, percent_close, num_equal_to_max_mean, num_equal_to_max_std = run_experiment_for_update_threshold(
            update_threshold)

        percent_opt_means.append(mean_percent_rew_of_opt)
        percent_opt_stds.append(std_percent_rew_of_opt)
        num_times_correct.append(percent_correct)
        num_times_close.append(percent_close)
        num_equal_to_max_means.append(num_equal_to_max_mean)
        num_equal_to_max_stds.append(num_equal_to_max_std)
    # run_experiment_h_alpha()

    X = [i for i in updates]
    Y = np.array(num_equal_to_max_means)
    Ystd = np.array(num_equal_to_max_stds)
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Threshold for Probabilities")
    plt.ylabel("Num Equal to Max")
    plt.title("Num Equal to Max by Threshold")
    plt.savefig("images/cirl_w_rc_num_max_hpmf.png")
    plt.show()

    X = [i for i in updates]
    Y = np.array(percent_opt_means)
    Ystd = np.array(percent_opt_stds)
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Threshold for Probabilities")
    plt.ylabel("Avg Percent of Optimal Reward")
    plt.title("% Reward by Threshold")
    plt.savefig("images/cirl_w_rc_percent_opt_hpmf.png")
    plt.show()

    plt.plot(updates, num_times_correct)
    plt.xlabel("Threshold for Probabilities")
    plt.ylabel("Percent Correct")
    plt.title("Percent Correct by Threshold")
    plt.savefig("images/cirl_w_rc_correct_hpmf.png")
    plt.show()

    plt.plot(updates, num_times_close)
    plt.xlabel("Threshold for Probabilities")
    plt.ylabel("Percent Close")
    plt.title("Percent Close by Threshold")
    plt.savefig("images/cirl_w_rc_close_hpmf.png")
    plt.show()

def evaluate_human_alphas():
    np.random.seed(0)
    percent_opt_means = []
    percent_opt_stds = []
    num_times_correct = []
    num_times_close = []
    num_equal_to_max_means = []
    num_equal_to_max_stds = []

    h_alpha_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for h_alpha_val in h_alpha_list:
        mean_percent_rew_of_opt, std_percent_rew_of_opt, percent_correct, percent_close, num_equal_to_max_mean, num_equal_to_max_std = run_experiment_for_human_alpha(
            h_alpha_val)

        percent_opt_means.append(mean_percent_rew_of_opt)
        percent_opt_stds.append(std_percent_rew_of_opt)
        num_times_correct.append(percent_correct)
        num_times_close.append(percent_close)
        num_equal_to_max_means.append(num_equal_to_max_mean)
        num_equal_to_max_stds.append(num_equal_to_max_std)
    # run_experiment_h_alpha()

    X = [i for i in h_alpha_list]
    Y = np.array(num_equal_to_max_means)
    Ystd = np.array(num_equal_to_max_stds)
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Percent of Time Human Suboptimal (Alpha)")
    plt.ylabel("Num Equal to Max")
    plt.title("Num Equal to Max by Alpha")
    plt.savefig("images/cirl_w_rc_halpha_num_max_hpmfs.png")
    plt.show()

    X = [i for i in h_alpha_list]
    Y = np.array(percent_opt_means)
    Ystd = np.array(percent_opt_stds)
    plt.plot(X, Y, 'k-')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.xlabel("Percent of Time Human Suboptimal (Alpha)")
    plt.ylabel("Avg Percent of Optimal Reward")
    plt.title("% Reward by Alpha")
    plt.savefig("images/cirl_w_rc_halpha_percent_opt_hpmf.png")
    plt.show()

    plt.plot(h_alpha_list, num_times_correct)
    plt.xlabel("Percent of Time Human Suboptimal (Alpha)")
    plt.ylabel("Percent Correct")
    plt.title("Percent Correct by Alpha")
    plt.savefig("images/cirl_w_halpha_rc_correct_hpmf.png")
    plt.show()

    plt.plot(h_alpha_list, num_times_close)
    plt.xlabel("Percent of Time Human Suboptimal (Alpha)")
    plt.ylabel("Percent Close")
    plt.title("Percent Close by Alpha")
    plt.savefig("images/cirl_w_rc_halpha_close_hpmf.png")
    plt.show()


def run_experiment_random_human_without_multiprocess():

    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 1

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    round_to_percent_rewards = {i:[] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    for exp_num in range(num_exps):
        exp_num_results = run_k_rounds(exp_num, task_reward)

        cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
        cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal = exp_num_results
        num_equal_to_max.append(num_equal)

        if max_prob_is_correct is True:
            times_max_prob_is_correct += 1

        if max_prob_is_close is True:
            times_max_prob_is_close += 1


        if altruism_case == 'opt':
            n_greedy += 1
        if altruism_case == 'subopt':
            n_altruism += 1
        n_total += 1

        for j in range(len(percent_opt_each_round)):
            round_to_percent_rewards[j].append(percent_opt_each_round[j])

        # if altruism_case == 'opt':
        #     continue

        cvi_percents.append(cvi_percent_of_opt_team)
        stdvi_percents.append(stdvi_percent_of_opt_team)

        cvi_humanrew_percents.append(cvi_percent_of_opt_human)
        stdvi_humanrew_percents.append(stdvi_percent_of_opt_human)

        cvi_robotrew_percents.append(cvi_percent_of_opt_robot)
        stdvi_robotrew_percents.append(stdvi_percent_of_opt_robot)

        diff = cvi_percent_of_opt_team - stdvi_percent_of_opt_team
        diff = np.round(diff, 2)
        # print("percent_change = ", percent_change)
        if diff in percent_change:
            percent_change[diff] += 1

    teamrew_means = [np.round(np.mean(cvi_percents), 2), np.round(np.mean(stdvi_percents), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents), 2), np.round(np.std(stdvi_percents), 2)]

    # humanrew_means = [np.round(np.mean(cvi_humanrew_percents), 2), np.round(np.mean(stdvi_humanrew_percents), 2)]
    # humanrew_stds = [np.round(np.std(cvi_humanrew_percents), 2), np.round(np.std(stdvi_humanrew_percents), 2)]
    #
    # robotrew_means = [np.round(np.mean(cvi_robotrew_percents), 2), np.round(np.mean(stdvi_robotrew_percents), 2)]
    # robotrew_stds = [np.round(np.std(cvi_robotrew_percents), 2), np.round(np.std(stdvi_robotrew_percents), 2)]

    print("team rew stat results: ",
          stats.ttest_ind([elem * 100 for elem in cvi_percents], [elem * 100 for elem in stdvi_percents]))
    # print("human rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents], [elem * 100 for elem in stdvi_humanrew_percents]))
    # print("robot rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents],
    #                       [elem * 100 for elem in cvi_robotrew_percents]))

    # print("n_altruism = ", n_altruism)
    # print("n_greedy = ", n_greedy)
    # print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d]/sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('CIRL', fontsize=16)
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.xlabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_cdf.png")
    plt.show()
    plt.close()

    data = list([cvi_percents, stdvi_percents])
    fig, ax = plt.subplots(figsize=(5, 5))
    # build a violin plot
    ax.violinplot(data, showmeans=False, showmedians=True)
    # add title and axis labels
    ax.set_title('CIRL w RC', fontsize=16)
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # add x-tick labels
    xticklabels = ['CVI robot', 'StdVI robot']
    ax.set_xticks([1, 2])
    ax.set_xticklabels(xticklabels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add horizontal grid lines
    ax.yaxis.grid(True)
    # show the plot
    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_violin.png")
    plt.show()
    plt.close()




    # collab_means = [np.round(np.mean(cvi_percents[1]), 2), np.round(np.mean(stdvi_percents[1]), 2)]
    # collab_stds = [np.round(np.std(cvi_percents[1]), 2), np.round(np.std(stdvi_percents[1]), 2)]

    ind = np.arange(len(teamrew_means))  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 5))
    rects1 = ax.bar(ind - width, teamrew_means, width, yerr=teamrew_stds,
                    label='Team Reward', capsize=10)
    # rects2 = ax.bar(ind, humanrew_means, width, yerr=humanrew_stds,
    #                 label='Human Reward', capsize=10)
    # rects3 = ax.bar(ind + width, robotrew_means, width, yerr=robotrew_stds,
    #                 label='Robot Reward', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Robot Type', fontsize=14)
    ax.set_ylabel('Percent of Optimal Reward', fontsize=14)
    # ax.set_ylim(-0.00, 1.5)

    plt.yticks([0.0,0.2, 0.4, 0.6, 0.8, 1.0], [0.0,0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # plt.xticks([])

    ax.set_title('CIRL', fontsize=16)
    ax.set_xticks(ind, fontsize=14)
    ax.set_xticklabels(('CVI robot', 'StdVI robot'), fontsize=13)
    ax.legend(fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    autolabel(ax, rects1, "left")
    # autolabel(ax, rects2, "right")
    # autolabel(ax, rects3, "right")

    fig.tight_layout()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_bar.png")
    plt.show()
    plt.close()

    X = [i for i in np.arange(6)]
    Y = np.array([np.mean(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    Ystd = np.array([np.std(round_to_percent_rewards[i]) for i in round_to_percent_rewards])
    plt.title('Interactive IRL w RC', fontsize=16)
    plt.plot(X, Y, 'k-')
    # plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#1B2ACC', facecolor='#1B2ACC')
    plt.fill_between(X, Y - Ystd, Y + Ystd, alpha=0.5, edgecolor='#FFD580', facecolor='#FFD580')
    plt.ylabel("Percent of Optimal")
    plt.xlabel("Episode")

    plt.legend()
    plt.savefig(f"images/cirl_w_rc_{num_exps}_by_round_Std.png")
    plt.show()

    print()
    print("times_max_prob_is_correct = ", times_max_prob_is_correct)
    print("percent max_prob_is_correct = ", times_max_prob_is_correct/num_exps)

    print("times_max_prob_is_close = ", times_max_prob_is_close)
    print("percent max_prob_is_close = ", times_max_prob_is_close / num_exps)

    print("num_equal_to_max = ", np.mean(num_equal_to_max))

    print("CVI Mean Percent of Opt reward = ", np.round(np.mean(cvi_percents), 3))
    print("CVI Std Percent of Opt reward = ", np.round(np.std(cvi_percents), 3))


if __name__ == "__main__":
    np.random.seed(0)
    # run_experiment()
    run_experiment_without_multiprocess()
    # run_experiment_random_human_without_multiprocess()
    # evaluate_thresholds()
    # evaluate_human_alphas()