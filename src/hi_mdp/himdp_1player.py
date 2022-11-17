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
    def __init__(self, human_rew, human_depth):
        self.human_rew = human_rew
        self.human_depth = human_depth

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
        initial_state = copy.deepcopy(input_state)
        # have the robot act
        # update state
        rew = 0
        if state[ENV_IDX][robot_action] > 0:
            state[ENV_IDX][robot_action] -= 1
            rew += self.robot_rew[robot_action]
            # rew -= self.human_rew[robot_action]

        # Update robot history in state
        state[ROBOT_IDX].append(robot_action)

        # update human model's model of robot FROM SCRATCH
        # self.create_human_model()
        copy_human_model = copy.deepcopy(self.human_model)
        # sim_state = copy.deepcopy(START_STATE)
        # for i in range(len(state[ROBOT_IDX])):
        #     robot_action = state[ROBOT_IDX][i]
            # human_action = state[HUMAN_IDX][i]
            # print("sim_state, robot_action", sim_state, robot_action)
            # if sim_state == [0,4,0,0]:
                # print("available",self.get_available_actions_from_state([sim_state, [], []]))
        copy_human_model.update_with_partner_action(initial_state[ENV_IDX], robot_action)

        # if sim_state[robot_action] > 0:
        #     sim_state[robot_action] -= 1
            # if sim_state[human_action] > 0:
            #     sim_state[human_action] -= 1

        # have the human act
        # print("state", state)
        human_action = copy_human_model.act(state[ENV_IDX], [], [])
        #
        # # Update human history in state
        # state[HUMAN_IDX].append(robot_action)

        max_belief = max(list(copy_human_model.beliefs.values()))
        sig_max_belief = 1 / (1 + np.exp(-max_belief))
        sig_max_belief = 1
        # update state and human's model of robot
        if human_action is not None and state[ENV_IDX][human_action] > 0:
            # state[ENV_IDX][human_action] -= 1
            rew += (sig_max_belief * self.human_rew[human_action])

        # We're not going to update the robot beliefs. This is bc we just need to solve these mdps
        # self.robot.update_with_partner_action(human_state, human_action)

        return state, rew, self.is_done_given_state(state)

    def enumerate_states_joint(self):
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

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_robot_actions):
                # set the environment to the current state
                # self.set_to_state(copy.deepcopy(state))

                # Have the robot take the action
                current_state = copy.deepcopy(state)
                # have the robot act
                # update state
                current_rew = 0
                if current_state[ENV_IDX][action] > 0:
                    current_state[ENV_IDX][action] -= 1
                    current_rew += self.robot_rew[action]
                    # current_rew -= self.human_rew[action]

                # convert old state to tuple
                current_state_tup = self.flatten_to_tuple(current_state)
                if current_state_tup not in visited_states:
                    stack.append(copy.deepcopy(current_state))

                # Update robot history in state
                current_state[ROBOT_IDX].append(action)

                available_human_actions = self.get_available_actions_from_state(current_state)

                for h_idx, h_action in enumerate(available_human_actions):
                    current_human_state = copy.deepcopy(current_state)
                    current_human_rew = current_rew
                    # have the robot act
                    # update state
                    if current_human_state[ENV_IDX][h_action] > 0:
                        current_human_state[ENV_IDX][h_action] -= 1
                        # current_human_rew += self.human_rew[h_action]

                    # Update human history in state
                    current_human_state[HUMAN_IDX].append(h_action)

                    done = self.is_done_given_state(current_human_state)


                    new_state_tup = self.flatten_to_tuple(current_human_state)

                    if new_state_tup not in visited_states:
                        stack.append(copy.deepcopy(current_human_state))

                    # add edge to graph from current state to new state with weight equal to reward
                    G.add_edge(state_tup, new_state_tup, weight=current_human_rew, action=action)


        states = list(G.nodes)
        print("NUMBER OF STATES", len(states))
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
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx


    # implementation of tabular value iteration
    def value_iteration(self):
        self.epsilson = 0.0001
        self.gamma = 0.999
        self.maxiter = 100000
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
                vf[s] = np.max(np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                # vf[s] = np.max(np.sum((rewards[s]) * transitions[s, :, :], 0))
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
            # pdb.set_trace()
            # pi[s] = np.argmax(np.sum(vf * transitions[s,:,:],0))
            pi[s] = np.argmax(np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
            policy[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
            # pi[s] = np.sum(vf * transitions[s,:,:],0)

        self.vf = vf
        self.pi = pi
        self.policy = policy
        # print("self.pi", self.pi)
        return vf, pi






if __name__ == "__main__":
    test = HiMDP((0.9, 0.1, -0.9, 0.2), 1)
    test.enumerate_states()
    test.value_iteration()



