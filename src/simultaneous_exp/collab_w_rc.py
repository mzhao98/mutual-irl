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
from collab_robot import Robot
from human_model import Greedy_Human, Collaborative_Human

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

import pickle
import json
import sys
import os

COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow', None:'none'}

class Toy():
    def __init__(self, color, weight):
        self.color = color
        self.weight = weight


class Player:
    def __init__(self, rew):
        self.ind_rew = rew



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
        robot_rew, human_rew = -1, -1
        if rh_action in self.state_remaining_objects and self.state_remaining_objects[rh_action] > 0:
            self.state_remaining_objects[rh_action] -= 1
            robot_rew += self.robot.ind_rew[rh_action]
            human_rew += self.human.ind_rew[rh_action]

        self.total_reward['team'] += (robot_rew + human_rew)
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew
        return (robot_rew + human_rew), robot_rew, human_rew


    def resolve_two_agents_same_item(self, robot_action, human_action):
        (robot_action_color, robot_action_weight) = robot_action
        (human_action_color, human_action_weight) = human_action
        robot_rew, human_rew = -1, -1
        if robot_action in self.state_remaining_objects:
            if self.state_remaining_objects[robot_action] == 0:
                robot_rew, human_rew = -1, -1
            elif self.state_remaining_objects[robot_action] == 1:
                self.state_remaining_objects[robot_action] -= 1
                robot_rew = self.robot.ind_rew[robot_action]
                human_rew = self.human.ind_rew[human_action]
                pickup_agent = np.random.choice(['r', 'h'])
                if pickup_agent == 'r':
                    human_rew = -1
                else:
                    robot_rew = -1
            else:
                self.state_remaining_objects[robot_action] -= 1
                self.state_remaining_objects[human_action] -= 1
                robot_rew += self.robot.ind_rew[robot_action]
                human_rew += self.human.ind_rew[human_action]

        self.total_reward['team'] += (robot_rew + human_rew)
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew

        return (robot_rew + human_rew), robot_rew, human_rew

    def resolve_two_agents_diff_item(self, robot_action, human_action):

        robot_rew, human_rew = -1, -1

        if robot_action is not None and robot_action in self.state_remaining_objects:
            (robot_action_color, robot_action_weight) = robot_action

            if robot_action_weight == 0:
                if self.state_remaining_objects[robot_action] > 0:
                    self.state_remaining_objects[robot_action] -= 1
                    robot_rew += self.robot.ind_rew[robot_action]

        if human_action is not None and human_action in self.state_remaining_objects:
            (human_action_color, human_action_weight) = human_action
            if human_action_weight == 0:
                if self.state_remaining_objects[human_action] > 0:
                    self.state_remaining_objects[human_action] -= 1
                    human_rew += self.human.ind_rew[human_action]

        self.total_reward['team'] += (robot_rew + human_rew)
        self.total_reward['robot'] += robot_rew
        self.total_reward['human'] += human_rew
        return (robot_rew + human_rew), robot_rew, human_rew


    def step(self, joint_action):

        robot_action = joint_action['robot']

        human_action = joint_action['human']
        # (human_action_color, human_action_weight) = human_action

        if robot_action == human_action and robot_action is not None:
            # collaborative pick up object
            (robot_action_color, robot_action_weight) = robot_action
            if robot_action_weight == 1:
                team_rew, robot_rew, human_rew = self.resolve_heavy_pickup(robot_action)

            # single pick up object
            else:
                team_rew, robot_rew, human_rew = self.resolve_two_agents_same_item(robot_action, human_action)

        else:
            team_rew, robot_rew, human_rew = self.resolve_two_agents_diff_item(robot_action, human_action)

        done = self.is_done()
        return (team_rew, robot_rew, human_rew), done


    def step_given_state(self, input_state, joint_action):
        state_remaining_objects = copy.deepcopy(input_state)


        robot_action = joint_action['robot']

        human_action = joint_action['human']

        robot_rew, human_rew = -1, -1
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
                        robot_rew += self.robot.ind_rew[robot_action]
                        human_rew += self.human.ind_rew[human_action]
                        pickup_agent = np.random.choice(['r', 'h'])
                        if pickup_agent == 'r':
                            human_rew = -1
                        else:
                            robot_rew = -1
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
        team_rew = robot_rew + human_rew
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
                        next_state, (r_sa, robot_rew, human_rew), done = self.step_given_state(current_state_remaining_objects, joint_action)
                        # print(f"current_state = ", current_state_remaining_objects)
                        # print("action=  ", joint_action)
                        # print("r_sa = ", r_sa)
                        # print("next_state = ", next_state)
                        # print("done = ", done)
                        # print()

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

                next_state, (r_sa, robot_rew, human_rew), done = self.step_given_state(current_state_remaining_objects,
                                                                                       joint_action)

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

            total_reward += team_rew

            if iters > 10:
                break

        return total_reward, human_only_reward, robot_only_reward

    def rollout_full_game_two_agents(self):
        self.reset()
        done = False
        total_reward = 0
        human_only_reward = 0
        robot_only_reward = 0

        iters = 0
        while not done:
            iters += 1
            current_state = copy.deepcopy(self.state_remaining_objects)

            robot_action = self.robot.act(current_state)
            human_action = self.human.act(current_state)

            action = {'robot': robot_action, 'human': human_action}

            (team_rew, robot_rew, human_rew), done = self.step(action)
            human_only_reward += human_rew
            robot_only_reward += robot_rew
            # print(f"current_state = ", current_state)
            # print("action=  ", action)
            # print("team_rew = ", team_rew)
            # print("next_state = ", self.state_remaining_objects)
            # print("done = ", done)
            # print()
            total_reward += team_rew

            if iters > 10:
                print("Cannot finish")
                break

        return total_reward, human_only_reward, robot_only_reward

    def compute_optimal_performance(self):
        self.enumerate_states()
        self.value_iteration()

        optimal_rew, human_only_reward, robot_only_reward = self.rollout_full_game_joint_optimal()
        return optimal_rew, human_only_reward, robot_only_reward




def test_optimal():
    robot = Player({(BLUE, 0): 2, (RED,0): 3, (GREEN, 0):1, (YELLOW, 0):1,
                    (BLUE, 1): 4, (RED,1): 6, (GREEN, 1):4, (YELLOW, 1):5})
    human = Player({(BLUE, 0): 2, (RED,0): 1, (GREEN, 0):2, (YELLOW, 0):3,
                    (BLUE, 1): 4, (RED,1): 3, (GREEN, 1):6, (YELLOW, 1):3})

    starting_objects = [(BLUE, 0),(YELLOW, 0),(BLUE, 0),(RED, 0),
                        (BLUE, 1),(BLUE, 1),(GREEN, 1),(RED, 1),
                        (YELLOW, 0),(GREEN, 0),(BLUE, 0),(RED, 0)]
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    opt = env.compute_optimal_performance()
    print("opt = ", opt)


def test_one_exp():
    robot_rew = {(BLUE, 0): 2, (RED, 0): 3, (GREEN, 0): 1, (YELLOW, 0): 1,
                (BLUE, 1): 4, (RED, 1): 6, (GREEN, 1): -1, (YELLOW, 1): -1}

    human_rew = {(BLUE, 0): 2, (RED, 0): 1, (GREEN, 0): 2, (YELLOW, 0): 3,
                (BLUE, 1): 4, (RED, 1): 3, (GREEN, 1): 6, (YELLOW, 1): 3}

    starting_objects = [(BLUE, 0), (YELLOW, 0), (BLUE, 0), (RED, 0),
                        (BLUE, 1), (BLUE, 1), (GREEN, 1), (RED, 1),
                        (YELLOW, 0), (GREEN, 0), (BLUE, 0), (YELLOW, 1)]

    # starting_objects = [(BLUE, 0), (YELLOW, 0),  (GREEN, 1), (RED, 1), (YELLOW, 1)]

    robot = Robot(robot_rew, human_rew, starting_objects, vi_type='cvi')
    human = Human(human_rew, robot_rew, starting_objects)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    final_team_rew = env.rollout_full_game_two_agents()
    print("CVI final_team_rew = ", final_team_rew)

    robot = Robot(robot_rew, human_rew, starting_objects, vi_type='stdvi')
    human = Human(human_rew, robot_rew, starting_objects)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    final_team_rew = env.rollout_full_game_two_agents()
    print("StdVI final_team_rew = ", final_team_rew)

    env = Simultaneous_Cleanup(robot, human, starting_objects)
    opt = env.compute_optimal_performance()
    print("opt = ", opt)


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

def run_k_rounds(exp_num, task_reward, h_alpha=0.0):
    print("exp_num = ", exp_num)

    robot_rew = {(BLUE, 0): np.random.randint(1,10),
                 (RED, 0): np.random.randint(1,10),
                 (GREEN, 0): np.random.randint(1,10),
                 (YELLOW, 0): np.random.randint(1,10),
                 (BLUE, 1): np.random.randint(1,10),
                 (RED, 1): np.random.randint(1,10),
                 (GREEN, 1): np.random.randint(1,10),
                 (YELLOW, 1): np.random.randint(1,10)}

    # human_rew = {(BLUE, 0): np.random.randint(1, 10),
    #              (RED, 0): np.random.randint(1, 10),
    #              (GREEN, 0): np.random.randint(1, 10),
    #              (YELLOW, 0): np.random.randint(1, 10),
    #              (BLUE, 1): np.random.randint(1, 10),
    #              (RED, 1): np.random.randint(1, 10),
    #              (GREEN, 1): np.random.randint(1, 10),
    #              (YELLOW, 1): np.random.randint(1, 10)}
    permutes = list(itertools.permutations(robot_rew.values()))
    # print("permutes",permutes)
    human_rew_values = list(permutes[np.random.choice(np.arange(len(permutes)))])
    object_keys = list(robot_rew.keys())
    human_rew = {object_keys[i]: human_rew_values[i] for i in range(len(object_keys))}

    all_objects = [(BLUE, 0), (YELLOW, 0), (GREEN, 0), (RED, 0), (BLUE, 1),(GREEN, 1), (RED, 1), (YELLOW, 1)]
    n_objects = np.random.randint(4,10)
    starting_objects = [all_objects[i] for i in np.random.choice(np.arange(len(all_objects)), size=n_objects, replace=True)]

    robot = Robot(robot_rew, human_rew, starting_objects, vi_type='cvi')
    human = Collaborative_Human(human_rew, robot_rew, starting_objects, 0)
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    optimal_rew, best_human_rew, best_robot_rew = env.compute_optimal_performance()
    # print("opt = ", optimal_rew)

    # Test 2 greedy
    robot = Greedy_Human(robot_rew, human_rew, starting_objects, 0)
    human = Greedy_Human(human_rew, robot_rew, starting_objects, 0)
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    greedy_team_rew, greedy_human_rew, greedy_robot_rew = env.rollout_full_game_two_agents()
    # print("Fully 2 greedy final_team_rew = ", greedy_team_rew)

    robot = Robot(robot_rew, human_rew, starting_objects, vi_type='cvi')
    human = Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    cvi_rew, cvi_human_rew, cvi_robot_rew = env.rollout_full_game_two_agents()
    # print("CVI final_team_rew = ", cvi_rew)

    robot = Robot(robot_rew, human_rew, starting_objects, vi_type='stdvi')
    human = Collaborative_Human(human_rew, robot_rew, starting_objects, h_alpha)
    robot.setup_value_iteration()
    env = Simultaneous_Cleanup(robot, human, starting_objects)
    stdvi_rew, stdvi_human_rew, stdvi_robot_rew = env.rollout_full_game_two_agents()
    # print("StdVI final_team_rew = ", stdvi_rew)



    altruism_case = 'opt'
    if greedy_team_rew < optimal_rew:
        altruism_case = 'subopt'

    if cvi_rew > optimal_rew:
        print("PROBLEM CVI larger than OPT")
        print("optimal_rew", optimal_rew)
        print("cvi_rew = ", cvi_rew)
        print()


    cvi_percent_of_opt_team = max(0,cvi_rew) / optimal_rew
    stdvi_percent_of_opt_team = max(0,stdvi_rew) / optimal_rew

    # cvi_percent_of_opt_human = max(0,cvi_human_rew) / best_human_rew
    # stdvi_percent_of_opt_human = max(0,stdvi_human_rew) / best_human_rew
    #
    # cvi_percent_of_opt_robot = max(0,cvi_robot_rew) / best_robot_rew
    # stdvi_percent_of_opt_robot = max(0,stdvi_robot_rew) / best_robot_rew
    cvi_percent_of_opt_human, stdvi_percent_of_opt_human, cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot = 0,0,0,0

    print("done with exp_num = ", exp_num)
    return cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
           cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case


def run_experiment():
    np.random.seed(0)
    task_reward = [1, 1, 1, 1]

    cvi_percents = []
    stdvi_percents = []

    cvi_humanrew_percents = []
    stdvi_humanrew_percents = []

    cvi_robotrew_percents = []
    stdvi_robotrew_percents = []

    num_exps = 10

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    percent_change = {}
    for percent in np.arange(-1.0, 1.01, step=0.01):
        percent_change[np.round(percent, 2)] = 0

    with Pool(processes=100) as pool:
        k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward) for exp_num in range(num_exps)])
        for result in k_round_results:
            cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
            cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case = result

            if altruism_case == 'opt':
                n_greedy += 1
            if altruism_case == 'subopt':
                n_altruism += 1
            n_total += 1

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
            print("percent_change = ", percent_change)
            percent_change[diff] += 1


    teamrew_means = [np.round(np.mean(cvi_percents), 2), np.round(np.mean(stdvi_percents), 2)]
    teamrew_stds = [np.round(np.std(cvi_percents), 2), np.round(np.std(stdvi_percents), 2)]

    # humanrew_means = [np.round(np.mean(cvi_humanrew_percents), 2), np.round(np.mean(stdvi_humanrew_percents), 2)]
    # humanrew_stds = [np.round(np.std(cvi_humanrew_percents), 2), np.round(np.std(stdvi_humanrew_percents), 2)]
    #
    # robotrew_means = [np.round(np.mean(cvi_robotrew_percents), 2), np.round(np.mean(stdvi_robotrew_percents), 2)]
    # robotrew_stds = [np.round(np.std(cvi_robotrew_percents), 2), np.round(np.std(stdvi_robotrew_percents), 2)]

    print("team rew stat results: ", stats.ttest_ind([elem*100 for elem in cvi_percents], [elem*100 for elem in stdvi_percents]))
    # print("human rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_humanrew_percents], [elem * 100 for elem in stdvi_humanrew_percents]))
    # print("robot rew stat results: ",
    #       stats.ttest_ind([elem * 100 for elem in cvi_robotrew_percents],
    #                       [elem * 100 for elem in cvi_robotrew_percents]))

    print("n_altruism = ", n_altruism)
    print("n_greedy = ", n_greedy)
    print("n_total = ", n_total)

    X = [d for d in percent_change]
    sum_Y = sum([percent_change[d] for d in percent_change])
    Y = [percent_change[d]/sum_Y for d in percent_change]

    # Compute the CDF
    CY = np.cumsum(Y)

    # Plot both
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('Collaboration w/ Hidden RC', fontsize=16)
    plt.plot(X, Y, label='Diff PDF')
    plt.plot(X, CY, 'r--', label='Diff CDF')
    plt.ylabel("% of Opt CVI - % of Opt StdVI")

    plt.legend()
    plt.savefig("simultaneous_collab_1_cdf.png")
    plt.show()






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

    ax.set_title('Collaboration w/ Hidden RC', fontsize=16)
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
    plt.savefig("simultaneous_collab_1.png")
    plt.show()


def run_experiment_h_alpha():

    task_reward = [1, 1, 1, 1]



    num_exps = 5

    n_altruism = 0
    n_total = 0
    n_greedy = 0

    h_alpha_to_avg_percent_cvi = {}
    for h_alpha in np.arange(0.0, 1.3, step=0.5):

        cvi_percents = []
        stdvi_percents = []

        cvi_humanrew_percents = []
        stdvi_humanrew_percents = []

        cvi_robotrew_percents = []
        stdvi_robotrew_percents = []

        with Pool(processes=10) as pool:
            k_round_results = pool.starmap(run_k_rounds, [(exp_num, task_reward, h_alpha) for exp_num in range(num_exps)])
            for result in k_round_results:
                cvi_percent_of_opt_team, stdvi_percent_of_opt_team, cvi_percent_of_opt_human, stdvi_percent_of_opt_human, \
                cvi_percent_of_opt_robot, stdvi_percent_of_opt_robot, altruism_case = result

                if altruism_case == 'opt':
                    n_greedy += 1
                if altruism_case == 'subopt':
                    n_altruism += 1
                n_total += 1

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

        h_alpha_to_avg_percent_cvi[h_alpha] = np.round(np.mean(cvi_percents), 2)


    X = [d for d in h_alpha_to_avg_percent_cvi]
    Y = [h_alpha_to_avg_percent_cvi[d] for d in h_alpha_to_avg_percent_cvi]

    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.title('Collaboration w/ Hidden RC', fontsize=16)
    plt.plot(X, Y, label='CVI')
    plt.xlabel("Percent of Optimal")

    plt.legend()
    plt.savefig("simultaneous_collab_1_h_alph.png")
    plt.show()



if __name__ == "__main__":
    run_experiment()
    # run_experiment_h_alpha()

