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

# from robot_1_birl_bsp_ig import Robot
# from robot_2_birl_bsp import Robot
# from robot_3_birl_maxplan import Robot
# from robot_4_birl_maxplan_ig import Robot
from robot_5_pedbirl_pragplan import Robot
# from robot_6_pedbirl_taskbsp import Robot
# from robot_7_taskbirl_pragplan import Robot
# from robot_8_birlq_bsp_ig import Robot
# from robot_9_birlq_bsp import Robot
# from robot_10_maxent_maxplan import Robot


from human_model import Suboptimal_Collaborative_Human
# import seaborn as sns
# import matplotlib.cm as cm
# import matplotlib.animation as animation
from scipy.stats import sem
import pickle
import json
import sys
import os
import subprocess
import glob

# ROBOT_TYPE = 'exp2_cirl_w_hard_rc_wo_expl_w_replan'
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

class Bait_Shooter():
    def __init__(self, robot, human, starting_actions_to_perform):
        self.robot = robot
        self.human = human

        self.total_reward = {'team': 0, 'robot': 0, 'human': 0}
        self.state_actions_completed = []
        self.possible_actions = []

        self.starting_actions_to_perform = starting_actions_to_perform
        self.reset()

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
        self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.0001
        self.gamma = 0.9
        self.maxiter = 10000

    def reset(self):
        self.state_actions_completed = [[], []]
        self.possible_actions = []
        for action in self.starting_actions_to_perform:
            self.possible_actions.append(action)

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

    def step(self, joint_action):

        robot_action = joint_action['robot']

        human_action = joint_action['human']

        robot_action_successful = False
        human_action_successful = False
        team_rew, robot_rew, human_rew = -2,0,0

        if robot_action is not None and robot_action not in np.concatenate(self.state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]
            if set(preconditions_list).issubset(self.state_actions_completed[0]) and len(self.state_actions_completed[0]) < 3:
                if robot_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(self.state_actions_completed):
                        robot_action_successful = True
                        self.state_actions_completed[0].append(robot_action)
                        robot_rew += self.robot.ind_rew[robot_action]
                else:
                    robot_action_successful = True
                    self.state_actions_completed[0].append(robot_action)
                    robot_rew += self.robot.ind_rew[robot_action]


        if human_action is not None and human_action not in np.concatenate(self.state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(self.state_actions_completed[1]) and len(self.state_actions_completed[1]) < 3:
                if human_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(self.state_actions_completed):
                        human_action_successful = True
                        self.state_actions_completed[1].append(human_action)
                        human_rew += self.human.ind_rew[human_action]
                else:
                    human_action_successful = True
                    self.state_actions_completed[1].append(human_action)
                    human_rew += self.human.ind_rew[human_action]


        done = self.is_done()
        if done:
            team_rew = 0
        # total_reward = team_rew + robot_rew + human_rew
        return (team_rew, robot_rew, human_rew), done, (robot_action_successful, human_action_successful)

    def step_given_state(self, input_state, joint_action):
        state_actions_completed = copy.deepcopy(input_state)

        robot_action = joint_action['robot']

        human_action = joint_action['human']

        robot_action_successful = False
        human_action_successful = False
        team_rew, robot_rew, human_rew = -2, 0, 0

        # print("state_actions_completed", state_actions_completed)
        # print("np.concatenate(state_actions_completed)", np.concatenate(state_actions_completed))
        # assert 0==1

        if robot_action is not None and robot_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[robot_action]

            if set(preconditions_list).issubset(state_actions_completed[0]) and len(state_actions_completed[0]) < 3:
                if robot_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        robot_action_successful = True
                        state_actions_completed[0].append(robot_action)
                        robot_rew += self.robot.ind_rew[robot_action]
                else:
                    robot_action_successful = True
                    state_actions_completed[0].append(robot_action)
                    robot_rew += self.robot.ind_rew[robot_action]

                # robot_rew += self.human.ind_rew[robot_action]

        if human_action is not None and human_action not in np.concatenate(state_actions_completed):
            preconditions_list = ACTION_TO_PRECONDITION[human_action]
            if set(preconditions_list).issubset(state_actions_completed[1]) and len(state_actions_completed[1]) < 3:
                if human_action == SHOOT:
                    if ACT_AS_BAIT in np.concatenate(state_actions_completed):
                        human_action_successful = True
                        state_actions_completed[1].append(human_action)
                        human_rew += self.human.ind_rew[human_action]
                else:
                    human_action_successful = True
                    state_actions_completed[1].append(human_action)
                    human_rew += self.human.ind_rew[human_action]

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

                next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(state, action)
                # if done == True:
                #     print("state", state)
                #     print("action", action)
                #     print("next_state", next_state)
                #     print("team_rew", team_rew)
                #     print("robot_rew", robot_rew)
                #     print("human_rew", human_rew)
                #     print("done", done)
                #     print()


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
            # print("i=", i)
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()

                current_state = copy.deepcopy([list(self.idx_to_state[s][0]), list(self.idx_to_state[s][1])])

                for action_idx in range(n_actions):
                    # check joint action
                    joint_action = self.idx_to_action[action_idx]
                    joint_action = {'robot': joint_action[0], 'human': joint_action[1]}

                    # print("current_state_remaining_objects = ", current_state_remaining_objects)
                    # print("joint_action = ", joint_action)
                    next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(current_state, joint_action)
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
                # print("DONE at iteration ", i)
                break

        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            current_state = copy.deepcopy([list(self.idx_to_state[s][0]), list(self.idx_to_state[s][1])])


            # compute new Q values
            for action_idx in range(n_actions):
                # check joint action
                joint_action = self.idx_to_action[action_idx]
                joint_action = {'robot': joint_action[0], 'human': joint_action[1]}

                next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(
                    current_state,
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
        game_results = []

        while not done:

            iters += 1
            # for i in range(10):
            current_state = copy.deepcopy(self.state_actions_completed)
            # print(f"current_state = {current_state}")
            current_state_tup = self.state_to_tuple(current_state)

            # print("availabel actions", self.get_possible_actions(current_state))
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # print("action_distribution = ", action_distribution)
            # print("action", action)
            game_results.append((current_state, action[0], action[1]))
            action = {'robot': action[0], 'human': action[1]}



            next_state, (team_rew, robot_rew, human_rew), done = self.step_given_state(current_state, action)
            # if team_rew != 0:
            #     team_rew = -6
            self.state_actions_completed = next_state

            human_only_reward += human_rew
            robot_only_reward += robot_rew
            print(f"current_state = ", current_state)
            print("robot action=  ", ACTION_TO_TEXT[action['robot']])
            print("human action=  ", ACTION_TO_TEXT[action['human']])
            print("team_rew = ", team_rew)
            print("robot_rew = ", robot_rew)
            print("human_rew = ", human_rew)
            print("next_state = ", next_state)
            print("done = ", done)
            print()

            # print(
            # f"current_state= {current_state}, next_state={next_state}, rew={rew}, is done = {done}")
            # print(
            #     f"team = {team_rew}, robot={robot_rew}, human = {human_rew}, total = {team_rew + robot_rew + human_rew}")
            total_reward += (team_rew + robot_rew + human_rew)

            if iters > 20:
                break

        return total_reward, human_only_reward, robot_only_reward, game_results

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
            current_state = copy.deepcopy(self.state_actions_completed)

            robot_action = self.robot.act_old(current_state)
            # is_start = False
            human_action = self.human.act(current_state)

            action = {'robot': robot_action, 'human': human_action}

            (team_rew, robot_rew, human_rew), done, (_, _) = self.step(action)
            human_only_reward += human_rew
            robot_only_reward += robot_rew
            # print(f"current_state = ", current_state)
            # print("action=  ", action)
            # print("team_rew = ", team_rew)
            # print("next_state = ", self.state_remaining_objects)
            # print("done = ", done)
            # print()
            total_reward += (team_rew + robot_rew + human_rew)

            if iters > 20:
                # print("Cannot finish")
                break

        return total_reward, human_only_reward, robot_only_reward

    def compute_optimal_performance(self):
        # print("start enumerating states")
        self.enumerate_states()
        # print("done enumerating states")
        # print("start vi")
        self.value_iteration()
        # print("done vi")

        optimal_rew, human_only_reward, robot_only_reward, game_results = self.rollout_full_game_joint_optimal()
        return optimal_rew, human_only_reward, robot_only_reward, game_results

    def rollout_multiround_game_two_agents(self, replan_online, use_exploration, num_rounds, plot=False):
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        multiround_belief_history = {}
        reward_for_all_rounds = []

        # print(f"robot rew = {self.robot.ind_rew}")
        # print(f"human rew = {self.human.ind_rew}")
        # print(f"robot rew = {self.robot.ind_rew}")
        game_results = {}

        round_no = -1
        game_results[round_no] = {'traj': [], 'rewards': [], 'beliefs': [], 'final_reward': 0}
        # print(f"\n\nRound = {round_no}")
        if type(self.robot) == Robot:
            self.robot.setup_value_iteration()
        if type(self.human) == Robot:
            self.human.setup_value_iteration()

        self.robot.reset_belief_history()

        # print("self.robot.beliefs", self.robot.beliefs)
        # print(f"true = {self.robot.true_human_rew_idx}, beliefs = {self.robot.beliefs[self.robot.true_human_rew_idx]}")

        self.reset()
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        done = False
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0
        iters = 0

        human_y = 1
        robot_y = 2

        human_rewards_over_round = []
        robot_rewards_over_round = []
        team_rewards_over_round = []
        total_rewards_over_round = []

        is_start = True
        prev_robot_action = None
        prev_data_pt = (None, None, None)
        while not done:
            iters += 1
            current_state = copy.deepcopy(self.state_actions_completed)

            robot_action = self.robot.act(current_state, is_start=is_start, round_no=round_no,
                                          use_exploration=use_exploration, boltzman=False)
            # robot_action = self.robot.act_old(current_state)
            is_start = False
            # print("current_state for human acting", current_state)
            # human_action = self.human.act(current_state, round_no)
            # human_action = self.human.act_human(current_state, round_no)
            human_action = self.robot.act_human(current_state, robot_action, round_no)

            game_results[round_no]['traj'].append((current_state, robot_action, human_action))

            action = {'robot': robot_action, 'human': human_action}
            # print(f"iter {iters}: objects left {sum(self.state_remaining_objects.values())} --> action {action}")

            (team_rew, robot_rew, human_rew), done, (robot_action_successful, human_action_successful) = self.step(
                action)
            # if team_rew != 0:
            #     team_rew = -6

            game_results[round_no]['rewards'].append((team_rew, robot_rew, human_rew))
            game_results[round_no]['beliefs'].append(self.robot.beliefs)

            human_rewards_over_round.append(human_rew)
            robot_rewards_over_round.append(robot_rew)
            team_rewards_over_round.append(team_rew)
            total_rewards_over_round.append(team_rew + human_rew + robot_rew)

            prev_robot_action = robot_action

            human_only_reward += human_rew
            robot_only_reward += robot_rew
            total_reward += (team_rew + human_rew + robot_rew)

            if iters > 20:
                # print("Cannot finish")
                break
        reward_for_all_rounds.append(total_reward)
        game_results[round_no]['final_reward'] = total_reward
        # print(f"Round {round_no}: total reward = {total_reward}")

        for round_no in range(num_rounds):
            game_results[round_no] = {'traj': [], 'rewards': [], 'beliefs': [], 'final_reward': 0}
            print(f"\n\nRound = {round_no}")
            self.robot.setup_value_iteration()
            # self.human.setup_value_iteration()

            self.robot.reset_belief_history()
            # self.human.reset_belief_history()

            # self.robot.reset_belief_history()

            # print("self.robot.beliefs", self.robot.beliefs)
            # print(f"true = {self.robot.true_human_rew_idx}, beliefs = {self.robot.beliefs[self.robot.true_human_rew_idx]}")

            self.reset()
            total_reward = 0
            human_only_reward = 0
            robot_only_reward = 0

            done = False
            total_reward = 0
            human_only_reward = 0
            robot_only_reward = 0
            iters = 0

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
            prev_robot_action = None
            prev_data_pt = (None, None, None)
            while not done:
                # if len(self.state_actions_completed) >= MAX_LENGTH_TIME:
                #     break
                iters += 1
                current_state = copy.deepcopy(self.state_actions_completed)

                robot_action = self.robot.act(current_state, is_start=is_start, round_no=round_no,
                                              use_exploration=use_exploration, boltzman=False)
                # robot_action = self.robot.act_old(current_state)
                is_start = False
                # print("current_state for human acting", current_state)
                # human_action = self.human.act(current_state, round_no)
                # human_action = self.human.act_human(current_state, round_no)
                human_action = self.robot.act_human(current_state, robot_action, round_no)

                # max_key = max(self.robot.beliefs, key=lambda k: self.robot.beliefs[k]['prob'])
                # print()
                # print("max prob belief", (self.robot.beliefs[max_key]['reward_dict'], self.robot.beliefs[max_key]['prob']))
                # print("true robot rew", self.robot.ind_rew)
                # print("true human rew", self.human.ind_rew)

                game_results[round_no]['traj'].append((current_state, robot_action, human_action))

                action = {'robot': robot_action, 'human': human_action}
                # print(f"iter {iters}: objects left {sum(self.state_remaining_objects.values())} --> action {action}")

                (team_rew, robot_rew, human_rew), done, (robot_action_successful, human_action_successful) = self.step(
                    action)
                # if team_rew != 0:
                #     team_rew = -6

                game_results[round_no]['rewards'].append((team_rew, robot_rew, human_rew))
                game_results[round_no]['beliefs'].append(self.robot.beliefs)

                human_rewards_over_round.append(human_rew)
                robot_rewards_over_round.append(robot_rew)
                team_rewards_over_round.append(team_rew)
                total_rewards_over_round.append(team_rew + human_rew + robot_rew)

                if type(self.robot) == Robot:
                    # print("human_action_successful", human_action_successful)
                    if human_action_successful is True:
                        # print("updating based on h action")
                        # print("current_state", current_state)
                        # print("human_action", human_action)
                        # max_key = max(self.robot.beliefs, key=lambda k: self.robot.beliefs[k]['prob'])
                        # print("old prob belief", self.robot.beliefs[max_key]['reward_dict'])

                        self.robot.update_based_on_h_action(current_state, robot_action, human_action)
                        max_key = max(self.robot.beliefs, key=lambda k: self.robot.beliefs[k]['prob'])
                        print("new prob belief", {ACTION_TO_TEXT[x]: self.robot.beliefs[max_key]['reward_dict'][x]
                                                  for x in self.robot.beliefs[max_key]['reward_dict']})
                        print("prob belief = ", self.robot.beliefs[max_key]['prob'])
                    # if robot_action_successful is True:
                    #     self.human.update_based_on_h_action(current_state, robot_action, human_action)

                if replan_online:
                    self.robot.setup_value_iteration()
                    # self.human.setup_value_iteration()

                prev_robot_action = robot_action

                human_only_reward += human_rew
                robot_only_reward += robot_rew
                total_reward += (team_rew + human_rew + robot_rew)

                print(f"current_state = ", current_state)
                print("action=  ", action)
                print("robot action", ACTION_TO_TEXT[action['robot']])
                print("human action", ACTION_TO_TEXT[action['human']])
                print("team_rew = ", team_rew)
                print("human_rew", human_rew)
                print("robot_rew", robot_rew)
                print("cumulative reward", total_reward)
                print("next_state = ", self.state_actions_completed)
                max_key = max(self.robot.beliefs, key=lambda keyname: self.robot.beliefs[keyname]['prob'])
                max_prob_hyp = self.robot.beliefs[max_key]['reward_dict']
                max_prob = self.robot.beliefs[max_key]['prob']
                # print("True human reward, ", self.human.ind_rew)
                # print("True robot reward, ", self.robot.ind_rew)
                # print("new robot beliefs = ", max_prob_hyp)
                # print("new robot beliefs prob = ", max_prob)
                # print("done = ", done)
                # print()

                # print(f"team rew = {team_rew}, human rew = {human_rew}, robot rew = {robot_rew} --> total rew = {total_reward}\n")

                if iters > 20:
                    # print("Cannot finish")
                    break

            # print("total_reward", total_reward)

            multiround_belief_history[round_no] = self.robot.history_of_human_beliefs
            reward_for_all_rounds.append(total_reward)
            game_results[round_no]['final_reward'] = total_reward

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
        if self.robot.beliefs[max_prob_idx]['prob'] == self.robot.beliefs[self.robot.true_human_rew_idx][
            'prob'] or second_prob_idx == self.robot.true_human_rew_idx or abs(
            max_prob - self.robot.beliefs[self.robot.true_human_rew_idx]['prob']) <= 0.2:
            max_prob_is_close = True

        num_equal_to_max = 0
        for idx in self.robot.beliefs:
            if self.robot.beliefs[idx]['prob'] == max_prob:
                num_equal_to_max += 1

        lstm_accuracies_list = []
        return total_reward, human_only_reward, robot_only_reward, multiround_belief_history, reward_for_all_rounds, max_prob_is_correct, max_prob_is_close, num_equal_to_max, lstm_accuracies_list, game_results


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


def run_k_rounds(exp_num, task_reward, seed, h_alpha, update_threshold, random_human, replan_online, use_exploration, task_type):
    print("exp_num = ", exp_num)
    cvi_percent_opt_list = []

    experiment_config = {}
    experiment_config['seed'] = seed

    np.random.seed(seed)
    # for round_number in range(1):
    pref = np.random.choice([0, 1])
    pref = 0
    # CHOP_CHICKEN, POUR_BROTH, GET_BOWL, CHOP_LETTUCE, POUR_DRESSING, GET_CUTTING_BOARD
    # FEED_DRY_FOOD, FEED_WET_FOOD, REFILL_WATER, WASH_BOWL, SWEEP_FLOOR, CLEAR_LITTER, BATHE_CAT
    # [POSITION_TO_BAIT, ACT_AS_BAIT, POSITION_TO_SHOOT, SHOOT]
    if pref == 0:
        human_rew = {
            POSITION_TO_BAIT: 6,
            ACT_AS_BAIT: 6,
            POSITION_TO_SHOOT: 2,
            SHOOT: 2,
            WAIT: 0,
        }
    else:
        human_rew = {
            POSITION_TO_BAIT: 2,
            ACT_AS_BAIT: 2,
            POSITION_TO_SHOOT: 6,
            SHOOT: 6,
            WAIT: 0,
        }


    experiment_config['pref'] = pref

    team_rew = {
        POSITION_TO_BAIT: 0,
        ACT_AS_BAIT: 0,
        POSITION_TO_SHOOT: 0,
        SHOOT: 0,
        WAIT: 0,
        }



    # permutes = list(itertools.permutations(list(human_rew.values())))
    # permutes = list(set(permutes))
    permutes = [[6,6,2,2,0], [2,2,6,6,0]]


    robot_rew = {
        POSITION_TO_BAIT: 4.4,
        ACT_AS_BAIT: 2,
        POSITION_TO_SHOOT: 2,
        SHOOT: 2,
        WAIT: 2,
    }

    experiment_config['robot_rew'] = robot_rew
    experiment_config['human_rew'] = human_rew

    # random_h_alpha = np.random.uniform(0.5, 1.0)
    # random_h_deg_collab = np.random.uniform(0.1, 1.0)
    if random_human is False:
        random_h_alpha = h_alpha
        random_h_deg_collab = 0.5
    else:
        # random_h_alpha = np.random.uniform(0.1, 1.0)
        # random_h_deg_collab = np.random.uniform(0.1, 1.0)
        random_h_alpha = np.random.uniform(0.1, 1.0)
        random_h_deg_collab = 0.5

    experiment_config['random_h_alpha'] = random_h_alpha
    experiment_config['random_h_deg_collab'] = random_h_deg_collab

    starting_objects = copy.deepcopy(ACTION_LIST)
    experiment_config['starting_objects'] = starting_objects
    #
    print()
    print("seed", seed)
    print("human_rew", human_rew)
    print("robot_rew", robot_rew)
    print("starting_objects", starting_objects)
    print()
    exp_results = {}

    robot = Robot(team_rew, robot_rew, human_rew, starting_objects, robot_knows_human_rew=True, permutes=permutes,
                  vi_type='cvi', is_collaborative_human=True)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha,
                                           h_deg_collab=random_h_deg_collab)
    env = Bait_Shooter(robot, human, starting_objects)
    optimal_rew, best_human_rew, best_robot_rew, game_results = env.compute_optimal_performance()
    print("Optimal Reward = ", optimal_rew)
    exp_results['optimal_rew'] = optimal_rew
    exp_results['optimal_game_results'] = game_results


    robot = Robot(team_rew, robot_rew, human_rew, starting_objects, robot_knows_human_rew=False, permutes=permutes,
                  vi_type='cvi', is_collaborative_human=True, update_threshold=update_threshold)
    human = Suboptimal_Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha=random_h_alpha,
                                           h_deg_collab=random_h_deg_collab)
    robot.setup_value_iteration()
    env = Bait_Shooter(robot, human, starting_objects)
    cvi_rew, cvi_human_rew, cvi_robot_rew, multiround_belief_history, \
    reward_for_all_rounds, max_prob_is_correct, max_prob_is_close, num_equal_to_max, \
    lstm_accuracies_list, game_results = env.rollout_multiround_game_two_agents(replan_online, use_exploration,
        num_rounds=1, plot=False)
    cvi_percent_opt = cvi_rew/optimal_rew
    cvi_percent_opt_list.append(cvi_percent_opt)
    print("CVI final_team_rew = ", max(0.0, cvi_rew/optimal_rew))
    exp_results['cvi_rew'] = cvi_rew
    exp_results['cvi_multiround_belief_history'] = multiround_belief_history
    exp_results['cvi_reward_for_all_rounds'] = reward_for_all_rounds
    exp_results['max_prob_is_correct'] = max_prob_is_correct
    exp_results['max_prob_is_close'] = max_prob_is_close
    exp_results['num_equal_to_max'] = num_equal_to_max
    exp_results['lstm_accuracies_list'] = lstm_accuracies_list
    exp_results['cvi_game_results'] = game_results


    altruism_case = 'not_checked'
    exp_results['altruism_case'] = altruism_case

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
        percent_opt_each_round.append(max(0, reward_for_all_rounds[j]) / optimal_rew)

    if cvi_rew > optimal_rew:
        print("OPTIMAL ERROR: CVI rew larger than Optimal")
        raise ArithmeticError

    cvi_percent_of_opt_team = max(0, cvi_rew) / optimal_rew

    exp_results['cvi_percent_of_opt_team'] = cvi_percent_of_opt_team

    cvi_percent_of_opt_team = cvi_percent_of_opt_team
    stdvi_percent_of_opt_team = 0

    cvi_percent_of_opt_human, stdvi_percent_of_opt_human, cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot = 0, 0, 0, 0

    print("done with exp_num = ", exp_num)
    experiment_results = {exp_num: {'config': experiment_config, 'results':exp_results}}
    return cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
           cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close, num_equal_to_max, lstm_accuracies_list, experiment_results, cvi_percent_opt_list
    # return cvi_percent_opt_list

def run_experiment(global_seed, experiment_number, task_type, exploration_type, replan_type, random_human, num_exps, belief_threshold=0.9):
    np.random.seed(global_seed)
    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    r_h_str = 'random_human'
    if random_human is False:
        r_h_str = 'deter_human'


    replan_online = True
    if replan_type == 'wo_replan':
        replan_online = False

    use_exploration = True
    if exploration_type == 'wo_expl':
        use_exploration = False

    experiment_name = f'exp-{experiment_number}_nexps-{num_exps}_globalseed-{global_seed}_task-{task_type}_explore-{exploration_type}_replan-{replan_type}_h-{r_h_str}_thresh-{belief_threshold}'
    path = f"results/{experiment_name}"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        os.makedirs(path + "/images")
        os.makedirs(path + "/exps")
        print(f"The new directory is created @ {path} ")

    configs = {}
    configs['global_seed'] = global_seed
    configs['experiment_number'] = experiment_number
    configs['task_type'] = task_type
    configs['exploration_type'] = exploration_type
    configs['replan_type'] = replan_type
    configs['random_human'] = random_human
    configs['num_exps'] = num_exps
    configs['r_h_str'] = r_h_str



    h_alpha = 0
    update_threshold = belief_threshold

    configs['h_alpha'] = h_alpha
    configs['update_threshold'] = update_threshold



    round_to_percent_rewards = {i: [] for i in range(6)}

    times_max_prob_is_correct = 0
    times_max_prob_is_close = 0
    num_equal_to_max = []

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    timestep_to_accuracy_list = {}
    random_seeds = np.random.randint(0,100000000, num_exps)
    configs['random_seeds'] = random_seeds
    # print("random_seeds", random_seeds)
    # return

    with open(path + '/' + 'config.pkl', 'wb') as fp:
        pickle.dump(configs, fp)

    experiment_num_to_results = {exp_num: {} for exp_num in range(num_exps)}

    with Pool(processes=num_exps) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward, random_seeds[exp_num], h_alpha, update_threshold, random_human, replan_online, use_exploration, task_type) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case, percent_opt_each_round, max_prob_is_correct, max_prob_is_close,\
            num_equal, lstm_accuracies_list, exp_results_dict, cvi_percent_opt_list = result
            print("cvi_percent_opt_list", cvi_percent_opt_list)

            for key_number in exp_results_dict:
                experiment_num_to_results[key_number] = exp_results_dict[key_number]
            # if lstm_accuracies_list is not None:
            #     for timestep in range(len(lstm_accuracies_list)):
            #         if timestep not in timestep_to_accuracy_list:
            #             timestep_to_accuracy_list[timestep] = []
            #         timestep_to_accuracy_list[timestep].append(lstm_accuracies_list[timestep])

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

    with open(path + '/exps/' + 'experiment_num_to_results.pkl', 'wb') as fp:
        pickle.dump(experiment_num_to_results, fp)

    aggregate_results = {}
    aggregate_results['cvi_percents'] = cvi_percents
    aggregate_results['stdvi_percents'] = stdvi_percents
    aggregate_results['percent_change'] = percent_change
    aggregate_results['round_to_percent_rewards'] = round_to_percent_rewards

    # with open(path + '/' + 'aggregate_results.pkl', 'wb') as fp:
    #     pickle.dump(aggregate_results, fp)

    # humanrew_means = [np.round(np.mean(cvi_humanrew_percents), 2), np.round(np.mean(stdvi_humanrew_percents), 2)]
    # humanrew_stds = [np.round(np.std(cvi_humanrew_percents), 2), np.round(np.std(stdvi_humanrew_percents), 2)]
    #
    # robotrew_means = [np.round(np.mean(cvi_robotrew_percents), 2), np.round(np.mean(stdvi_robotrew_percents), 2)]
    # robotrew_stds = [np.round(np.std(cvi_robotrew_percents), 2), np.round(np.std(stdvi_robotrew_percents), 2)]
    print(f"ROBOT TYPE = {experiment_name}, HUMAN TYPE = {r_h_str}")
    # print("team rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_percents], [elem * 100 for elem in stdvi_percents]))
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
    Y = [percent_change[d] / sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)


    print()
    print("times_max_prob_is_correct = ", times_max_prob_is_correct)
    print("percent max_prob_is_correct = ", times_max_prob_is_correct / num_exps)

    print("times_max_prob_is_close = ", times_max_prob_is_close)
    print("percent max_prob_is_close = ", times_max_prob_is_close / num_exps)

    print(f"num_equal_to_max = {np.mean(num_equal_to_max)}, std: {np.std(num_equal_to_max)}")

    print("cvi_percents", cvi_percents)

    print("CVI Mean Percent of Opt reward = ", np.round(np.mean(cvi_percents), 5))
    print("CVI Std Percent of Opt reward = ", np.std(cvi_percents))

    aggregate_results['times_max_prob_is_correct'] = times_max_prob_is_correct
    aggregate_results['times_max_prob_is_close'] = times_max_prob_is_close
    aggregate_results['num_equal_to_max'] = num_equal_to_max
    aggregate_results['mean_num_equal_to_max'] = np.mean(num_equal_to_max)


    with open(path + '/' + 'aggregate_results.pkl', 'wb') as fp:
        pickle.dump(aggregate_results, fp)


if __name__ == "__main__":
    # eval_threshold()
    robot_type = 'robot_1_birl_bsp_ig'

    # human_types = ['noiseless', 'boltz']
    task_types = ['cirl_w_hard_rc']
    exploration_types = ['wo_expl']
    human_type = 'boltz_b1'

    global_seed = 0
    num_exps = 1
    replan_type = 'w_replan'
    random_human = False

    # for human_type in human_types:
    # experiment_number = f'domain2_approp_diff_specified_3objs5_{robot_type}_{human_type}_human'
    experiment_number = f'domain3_soupsalad_{robot_type}_{human_type}_human'
    # f'2_1_3objs1_{robot_type}_{human_type}_human'
    # experiment_number = 'testing'
    # experiment_number = '7_baseline-cirl_boltz_human'
    # experiment_number = '7_coirl_birl-cirl_boltz_human'
    # task_type = 'cirl_w_hard_rc' # ['cirl', 'cirl_w_easy_rc', 'cirl_w_hard_rc']
    # exploration_type = 'wo_expl'
    # replan_type = 'wo_replan' # ['wo_replan', 'w_replan']
    # random_human = False
    for exploration_type in exploration_types:
    # exploration_type = 'wo_expl'
        for task_type in task_types:
            run_experiment(global_seed, experiment_number, task_type, exploration_type, replan_type, random_human, num_exps)

