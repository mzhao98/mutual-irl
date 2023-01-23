import copy

import numpy as np
import operator
import random
import pdb

import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from human_agent import Human_Hypothesis
# from

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3

# COLOR_LIST = [BLUE, GREEN, RED, YELLOW]


class OptimalMDP:

    def __init__(self, first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho):
        # World parameters
        self.start_state = start_state
        # self.starting_player = 0
        self.start_state.append(0)
        self.all_colors_list = all_colors_list
        self.task_reward = task_reward
        self.NOP = len(all_colors_list)
        self.first_player = first_player
        self.turn_to_reward_vector = {}
        if self.first_player == 'r':
            self.turn_to_reward_vector[0] = robot_rew
            self.turn_to_reward_vector[1] = human_rew
        else:
            self.turn_to_reward_vector[1] = robot_rew
            self.turn_to_reward_vector[0] = human_rew


        # Human model parameters
        self.human_rew = human_rew
        self.h_rho = h_rho

        # Robot model parameters
        # self.vi_type = vi_type
        self.robot_rew = robot_rew
        self.r_rho = r_rho
        # self.human_model_of_robot = None

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
            self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.0001
        self.gamma = 1.0
        self.maxiter = 10000

        # reset environment
        self.state = None
        self.reset()

    def set_robot_rew(self, robot_rew):
        self.robot_rew = robot_rew

    def set_human_rew(self, human_rew):
        self.human_rew = human_rew

    # def create_human_model_from_scratch(self):
    #     self.human_model_of_robot = Human_Hypothesis(self.human_rew, self.h_rho)
    #
    # def set_human_model_from_existing(self, human_model_of_robot):
    #     self.human_model_of_robot = copy.deepcopy(human_model_of_robot)

    def reset(self):
        self.initialize_game()

    def initialize_game(self):
        self.state = copy.deepcopy(self.start_state)

    def state_to_tuple(self, state):
        # return tuple([item for sublist in state for item in sublist])
        return tuple(state)

    def set_to_state(self, state):
        self.state = list(state)

    def get_available_actions_from_state(self, state):
        available = [self.NOP]
        for color in self.all_colors_list:
            if state[color] > 0:
                available.append(color)
        return available

    def is_done(self):
        return sum(self.state[:-1]) == 0

    def is_done_given_state(self, state):
        return sum(state[:-1]) == 0

    def robot_step_given_state(self, current_state, robot_action):
        # robot_action is a color
        next_state = copy.deepcopy(list(current_state))


        if robot_action == self.NOP:
            next_state[-1] = 1-next_state[-1]
            return next_state, 0, self.is_done_given_state(next_state)

        # have the robot act
        # update state
        rew = 0
        if next_state[robot_action] > 0:
            next_state[robot_action] -= 1

            rew += self.turn_to_reward_vector[next_state[-1]][robot_action]

        next_state[-1] = 1 - next_state[-1]
        # print(f"current_state= {current_state}, next_state={next_state}, rew={rew}, is done = {self.is_done_given_state(next_state)}")
        return next_state, rew, self.is_done_given_state(next_state)

    def enumerate_states(self):
        self.reset()

        actions = copy.deepcopy(self.all_colors_list)
        actions.append(self.NOP)
        # actions = []
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state)]

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get available actions
            available_robot_actions = self.get_available_actions_from_state(state)

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_robot_actions):

                next_state, rew, done = self.robot_step_given_state(state, action)

                new_state_tup = self.state_to_tuple(next_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=rew, action=action)

        states = list(G.nodes)
        # print("NUMBER OF STATES", len(states))
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

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

                # compute new Q values

                # Consider partner rew
                initial_state = copy.deepcopy(list(self.idx_to_state[s]))
                for action_idx in range(n_actions):
                    # have the robot act

                    current_state = copy.deepcopy(list(self.idx_to_state[s]))
                    robot_action = self.idx_to_action[action_idx]

                    if robot_action == self.NOP:
                        if sum((list(self.idx_to_state[s]))) == 0:
                            r_sa = 0
                        else:
                            r_sa = -10
                    else:
                        # get robot reward
                        r_sa = (self.turn_to_reward_vector[current_state[-1]][action_idx] if current_state[robot_action] > 0 else -10)

                        # update state
                        if current_state[robot_action] > 0:
                            current_state[robot_action] -= 1

                        current_state[-1] = 1-current_state[-1]

                    s11 = self.state_to_idx[tuple(current_state)]
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

            # compute new Q values

            # Consider partner rew
            initial_state = copy.deepcopy(list(self.idx_to_state[s]))
            for action_idx in range(n_actions):
                # have the robot act

                current_state = copy.deepcopy(list(self.idx_to_state[s]))
                robot_action = self.idx_to_action[action_idx]

                if robot_action == self.NOP:
                    if sum((list(self.idx_to_state[s]))) == 0:
                        r_sa = 0
                    else:
                        r_sa = -10
                else:
                    # get robot reward
                    r_sa = (self.turn_to_reward_vector[current_state[-1]][action_idx] if current_state[
                                                                                             robot_action] > 0 else -10)

                    # update state
                    if current_state[robot_action] > 0:
                        current_state[robot_action] -= 1

                    current_state[-1] = 1 - current_state[-1]

                s11 = self.state_to_idx[tuple(current_state)]
                Q[s, action_idx] = r_sa + (self.gamma * vf[s11])

            pi[s] = np.argmax(Q[s, :], 0)
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi

    def rollout_full_game_vi_policy(self):
        # print("policy", policy)
        self.reset()
        done = False
        total_reward = 0

        human_trace = []
        robot_trace = []
        human_greedy_alt = []
        robot_greedy_alt = []


        while not done:
        # for i in range(10):
            current_state = copy.deepcopy(self.state)
            # print(f"current_state = {current_state}")
            current_state_tup = self.state_to_tuple(current_state)

            # print("availabel actions", self.get_possible_actions(current_state))
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            next_state, rew, done = self.robot_step_given_state(current_state, action)
            self.state = next_state
            # print(
            # f"current_state= {current_state}, next_state={next_state}, rew={rew}, is done = {done}")

            total_reward += rew


        return total_reward


if __name__ == "__main__":
    start_state = [2, 2, 2, 2]
    all_colors_list = [BLUE, GREEN, RED, YELLOW]
    task_reward = [1, 1, 1]
    human_rew = [0.5, 0.1, 0.5, 0.1]
    h_rho = 0
    robot_rew = [0.5, 0.5, 0.1, 0.1]
    r_rho = 1
    first_player = 'r'
    # vi_type = 'stdvi'

    himdp = OptimalMDP(first_player, start_state, all_colors_list, task_reward, human_rew, h_rho, robot_rew, r_rho)
    himdp.enumerate_states()
    himdp.value_iteration()
    print('done with vi')
    rew = himdp.rollout_full_game_vi_policy()
    print("rew = ", rew)
