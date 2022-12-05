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
from human_hypothesis import Human_Hypothesis

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

ENV_IDX = 0
ROBOT_IDX = 1
HUMAN_IDX = 2

START_STATE = [2,5,1,2]


class HiMDP():
    def __init__(self, human_rew, human_depth, vi_type):
        self.human_rew = human_rew
        self.human_depth = human_depth
        self.vi_type = vi_type

        self.create_human_model()
        self.reset()


    def set_robot_rew(self, robot_rew):
        self.robot_rew = robot_rew


    def create_human_model(self):
        self.human_model = Human_Hypothesis(self.human_rew, self.human_depth)

    def set_human_model(self, robot_model):
        self.human_model = copy.deepcopy(robot_model)


    def reset(self):
        self.initialize_game()
        self.set_robot_rew((0.9, -0.9, 0.1, 0.3))


    def initialize_game(self):
        self.state = [START_STATE, [], []]

    def flatten_to_tuple(self, state):
        # return tuple([item for sublist in state for item in sublist])
        return tuple(state[ENV_IDX])

    def set_to_state(self, state):
        self.state = state


    def get_available_actions_from_state(self, state):
        available = []
        for color in COLOR_LIST:
            if state[ENV_IDX][color] > 0:
                available.append(color)
        # if state[ENV_IDX] == [0,4,0,0]:
        #     print("\n\navailable", available)

        return available

    def is_done(self):
        if sum(self.state[ENV_IDX]) == 0:
            return True
        return False

    def is_done_given_state(self, state):
        if sum(state[ENV_IDX]) == 0:
            return True
        return False

    def step(self, robot_action):
        # have the robot act
        # update state
        robot_state = copy.deepcopy(self.state[ENV_IDX])
        rew = 0
        if self.state[ENV_IDX][robot_action] > 0:
            self.state[ENV_IDX][robot_action] -= 1
            rew += self.robot_rew[robot_action]
            rew -= self.human_rew[robot_action]

        # Update robot history in state
        self.state[ROBOT_IDX].append(robot_action)

        # update human model's model of robot
        self.human_model.update_with_partner_action(robot_state, robot_action)

        # have the human act
        human_action = self.human_model.act(self.state, [], [])

        # Update human history in state
        self.state[HUMAN_IDX].append(robot_action)

        # update state and human's model of robot
        if self.state[ENV_IDX][human_action] > 0:
            self.state[ENV_IDX][human_action] -= 1
            rew += self.human_rew[human_action]

        # We're not going to update the robot beliefs. This is bc we just need to solve these mdps
        # self.robot.update_with_partner_action(human_state, human_action)

        return self.state, rew, self.is_done()

    def step_given_state(self, input_state, robot_action):
        state = copy.deepcopy(input_state)
        # have the robot act
        # update state
        rew = 0
        if state[ENV_IDX][robot_action] > 0:
            state[ENV_IDX][robot_action] -= 1
            rew += self.robot_rew[robot_action]
            rew -= self.human_rew[robot_action]

        # Update robot history in state
        state[ROBOT_IDX].append(robot_action)

        # update human model's model of robot FROM SCRATCH
        self.create_human_model()
        sim_state = START_STATE
        for i in range(len(state[ROBOT_IDX])):
            robot_action = state[ROBOT_IDX][i]
            human_action = state[HUMAN_IDX][i]
            self.human_model.update_with_partner_action(sim_state, robot_action)
            if sim_state[robot_action] > 0:
                sim_state[robot_action] -= 1
            if sim_state[human_action] > 0:
                sim_state[human_action] -= 1

        # have the human act
        human_action = self.human_model.act(state, [], [])

        # Update human history in state
        state[HUMAN_IDX].append(robot_action)

        # update state and human's model of robot
        if state[ENV_IDX][human_action] > 0:
            state[ENV_IDX][human_action] -= 1
            rew += self.human_rew[human_action]

        # We're not going to update the robot beliefs. This is bc we just need to solve these mdps
        # self.robot.update_with_partner_action(human_state, human_action)

        return state, rew, self.is_done_given_state(state)

    def robot_step_given_state(self, input_state, robot_action):
        state = copy.deepcopy(input_state)
        # have the robot act
        # update state
        rew = 0
        if state[ENV_IDX][robot_action] > 0:
            state[ENV_IDX][robot_action] -= 1
            rew += self.robot_rew[robot_action]


        return state, rew, self.is_done_given_state(state)
    
    def human_step_given_state(self, input_state, human_action):
        state = copy.deepcopy(input_state)
        # have the robot act
        # update state
        rew = 0
        if state[ENV_IDX][human_action] > 0:
            state[ENV_IDX][human_action] -= 1
            rew += self.human_rew[human_action]

        return state, rew, self.is_done_given_state(state)

    def enumerate_states(self):
        self.reset()

        actions = COLOR_LIST
        # actions = []
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state)]
        # prev_state = self.state

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.flatten_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get available
            available_robot_actions = self.get_available_actions_from_state(state)

            # print(f"\n\nstate = {state}, available = {available_robot_actions}")

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_robot_actions):

                next_state, rew, done = self.robot_step_given_state(state, action)

                new_state_tup = self.flatten_to_tuple(next_state)

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
        
        
        self.enumerate_states_for_partner()
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def enumerate_states_for_partner(self):
        self.reset()

        actions = COLOR_LIST
        # actions = []
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.state)]
        # prev_state = self.state

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.flatten_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get available
            available_robot_actions = self.get_available_actions_from_state(state)

            # print(f"\n\nstate = {state}, available = {available_robot_actions}")

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_robot_actions):

                next_state, rew, done = self.human_step_given_state(state, action)

                new_state_tup = self.flatten_to_tuple(next_state)

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

        self.partner_transitions, self.partner_rewards, self.partner_state_to_idx, \
        self.partner_idx_to_action, self.partner_idx_to_state, self.partner_action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx
        return
        
    def partner_value_iteration(self):
        self.partner_epsilson = 0.0001
        self.partner_gamma = 0.999
        self.partner_maxiter = 100000
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
        n_states = self.partner_transitions.shape[0]
        n_actions = self.partner_transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        partner_fixed_rew_in_state = np.zeros((n_states, 1))

        for i in range(self.partner_maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                # vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s,:,:],0))
                # inner_vf = np.max(np.sum((self.partner_rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                vf[s] = np.max(np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0))
                # vf[s] = np.max(np.sum((rewards[s]) * transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

                # if s == 148:
                #     action = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
                #     print("action = ", action)
                # pdb.set_trace()
            # check for convergence
            if delta < self.partner_epsilson:
                break
        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # pdb.set_trace()
            # pi[s] = np.argmax(np.sum(vf * transitions[s,:,:],0))
            pi[s] = np.argmax(np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0))
            policy[s] = np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0)
            partner_fixed_rew_in_state[s] = self.partner_rewards[s, int(pi[s][0])]
            # pi[s] = np.sum(vf * transitions[s,:,:],0)

        self.partner_vf = vf
        self.partner_pi = pi
        self.partner_pi = [int(x[0]) for x in self.partner_pi]
        # print("self.partner_pi", self.partner_pi)

        self.partner_policy = policy
        self.partner_fixed_rew_in_state = partner_fixed_rew_in_state
        # print("self.pi", self.pi)
        return vf, pi


    def partner_value_iteration_no_save(self):
        self.partner_epsilson = 0.0001
        self.partner_gamma = 0.999
        self.partner_maxiter = 100000
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
        n_states = self.partner_transitions.shape[0]
        n_actions = self.partner_transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        partner_fixed_rew_in_state = np.zeros((n_states, 1))

        for i in range(self.partner_maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                # vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s,:,:],0))
                # inner_vf = np.max(np.sum((self.partner_rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                vf[s] = np.max(np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0))
                # vf[s] = np.max(np.sum((rewards[s]) * transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

                # if s == 148:
                #     action = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
                #     print("action = ", action)
                # pdb.set_trace()
            # check for convergence
            if delta < self.partner_epsilson:
                break
        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # pdb.set_trace()
            # pi[s] = np.argmax(np.sum(vf * transitions[s,:,:],0))
            pi[s] = np.argmax(np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0))
            policy[s] = np.sum((self.partner_rewards[s] + self.partner_gamma * vf) * self.partner_transitions[s, :, :], 0)
            partner_fixed_rew_in_state[s] = self.partner_rewards[s, int(pi[s][0])]
            # pi[s] = np.sum(vf * transitions[s,:,:],0)

        # self.partner_vf = vf
        # self.partner_pi = pi
        # self.partner_pi = [int(x[0]) for x in self.partner_pi]
        # # print("self.partner_pi", self.partner_pi)
        #
        # self.partner_policy = policy
        # self.partner_fixed_rew_in_state = partner_fixed_rew_in_state
        # print("self.pi", self.pi)
        return pi, partner_fixed_rew_in_state

    # implementation of tabular value iteration
    def value_iteration(self):
        self.partner_value_iteration()
        self.epsilson = 0.0001
        self.gamma = 0.999
        self.maxiter = 1000
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

        for i in range(self.maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                # vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s,:,:],0))
                inner_component = np.zeros((n_states, n_actions))

                
                for a in range(n_actions):
                    for s1 in range(n_states):
                        if self.vi_type == 'mmvi':
                            # update human model's model of robot based on last robot action
                            initial_state = (list(self.idx_to_state[s]), [], [])
                            robot_action = self.idx_to_action[a]
                            copy_human_model = copy.deepcopy(self.human_model)
                            copy_human_model.update_with_partner_action(initial_state, robot_action)
                            new_pi, new_partner_fixed_rew_in_state = self.partner_value_iteration_no_save()
                            s2 = np.where(self.transitions[s1, :, int(new_pi[s1][0])] == 1)[0]
                            # print("s2=", s2)
                            if len(s2) == 0:
                                curr_vf = 0
                            else:
                                curr_vf = vf[s2[0]]

                            inner_component[s1, a] = (self.rewards[s, a] + new_partner_fixed_rew_in_state[s1] + self.gamma * curr_vf)[0]


                        elif self.vi_type == 'mmvi-nh':
                            s2 = np.where(self.transitions[s1, :, int(self.partner_pi[s1][0])] == 1)[0]
                            if len(s2) == 0:
                                curr_vf = 0
                            else:
                                curr_vf = vf[s2[0]]
                            inner_component[s1, a] = (self.rewards[s,a] + self.partner_fixed_rew_in_state[s1] + self.gamma * curr_vf)[0]

                        else:
                            inner_component[s1, a] = (self.rewards[s, a] + self.gamma * vf[s1])[0]

                # vf[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)

                vf[s] = np.max(np.sum((inner_component) * self.transitions[s, :, :], 0))

                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))

                # if s == 148:
                #     action = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
                #     print("action = ", action)
                # pdb.set_trace()
            # check for convergence
            if delta < self.epsilson:
                break
        # compute optimal policy
        policy = {}
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()
            # compute new value function
            inner_component = np.zeros((n_states, n_actions))

            for a in range(n_actions):
                for s1 in range(n_states):
                    if self.vi_type == 'mmvi':
                        # update human model's model of robot based on last robot action
                        initial_state = (list(self.idx_to_state[s]), [], [])
                        robot_action = self.idx_to_action[a]
                        copy_human_model = copy.deepcopy(self.human_model)
                        copy_human_model.update_with_partner_action(initial_state, robot_action)
                        new_pi, new_partner_fixed_rew_in_state = self.partner_value_iteration_no_save()
                        s2 = np.where(self.transitions[s1, :, int(new_pi[s1][0])] == 1)[0]
                        if len(s2) == 0:
                            curr_vf = 0
                        else:
                            curr_vf = vf[s2[0]]
                        inner_component[s1, a] = \
                        (self.rewards[s, a] + new_partner_fixed_rew_in_state[s1] + self.gamma * curr_vf)[0]


                    elif self.vi_type == 'mmvi-nh':
                        s2 = np.where(self.transitions[s1, :, int(self.partner_pi[s1][0])] == 1)[0]
                        if len(s2) == 0:
                            curr_vf = 0
                        else:
                            curr_vf = vf[s2[0]]
                        inner_component[s1, a] = \
                        (self.rewards[s, a] + self.partner_fixed_rew_in_state[s1] + self.gamma * curr_vf)[0]

                    else:
                        inner_component[s1, a] = (self.rewards[s, a] + self.gamma * vf[s1])[0]

            pi[s] = np.argmax(np.sum((inner_component) * self.transitions[s, :, :], 0))
            policy[s] = np.sum((inner_component) * self.transitions[s, :, :], 0)


        self.vf = vf
        self.pi = pi
        self.policy = policy
        return vf, pi






