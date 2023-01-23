import pdb

import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
NO_MOVE = 4
ACTION_LIST = [BLUE, GREEN, RED, YELLOW]
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
NUM_PLAYERS = 2
COLOR_TO_TEXT = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}
TURN_IDX = 4

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class Joint_MDP:
    def __init__(self, index_of_human, index_of_robot, players_to_reward, initial_state):
        self.initial_state_input = initial_state

        self.initial_state = [[0, initial_state[0]], [0, initial_state[1]], [0, initial_state[2]],
                              [0, initial_state[3]], [0]]

        self.index_of_human = index_of_human
        self.index_of_robot = index_of_robot

        self.player_to_reward_function = players_to_reward

        self.actions = ACTION_LIST
        self.state = copy.deepcopy(self.initial_state)
        self.task_rew = 1


    def reset(self):
        self.state = copy.deepcopy(self.initial_state)

    def get_num_remaining_objects(self):
        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (self.state[color][1] - self.state[color][0])

        return num_remaining

    def get_num_remaining_objects_in_state(self, state):
        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (state[color][1] - state[color][0])

        return num_remaining


    def is_game_over(self):
        if self.get_num_remaining_objects() == 0:
            return True
        return False

    def is_game_over_given_state(self, state):
        if self.get_num_remaining_objects_in_state(state) == 0:
            return True
        return False


    def get_possible_actions(self, state):
        # num_remaining = 0
        # for color in COLOR_LIST:
        #     num_remaining += (state[color][1] - state[color][0])
        #
        # if num_remaining == 0:
        #     return []

        valid_joint_actions = []
        for action in COLOR_LIST:
            num_remain = state[action][1] - state[action][0]
            if num_remain > 0:
                valid_joint_actions.append(action)

        # return valid_joint_actions
        return ACTION_LIST


    def set_to_state(self, state):
        self.state = state


    def step_given_state(self, current_state, action):
        # joint_action = (p1 action color, p2 action color)

        state = copy.deepcopy(current_state)
        player_idx = state[TURN_IDX][0]

        total_reward = 0
        # Resolve player action
        action_color = action

        if action_color == NO_MOVE:
            rew = -100
            total_reward += rew
        else:
            num_remain = state[action_color][1] - state[action_color][0]
            # more colors remain
            # pdb.set_trace()
            if num_remain > 0:
                state[action_color][0] = state[action_color][0] + 1
                rew = self.player_to_reward_function[player_idx][action_color]
                total_reward += (rew + self.task_rew)
            else:
                rew = -100
                total_reward += rew

        state[TURN_IDX][0] = 1-state[TURN_IDX][0]
        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (state[color][1] - state[color][0])
        done = (True if num_remaining == 0 else False)

        return copy.deepcopy(state), total_reward, done


    def step(self, action):
        # joint_action = (p1 action color, p2 action color)
        # state = copy.deepcopy(current_state)
        # print("Init self.state = ", self.state)
        player_idx = self.state[TURN_IDX][0]

        total_reward = 0
        # Resolve player action
        action_color = copy.deepcopy(action)
        # print("action_color = ", action_color)

        if action_color == NO_MOVE:
            rew = -100
            total_reward += rew
        else:
            num_remain = self.state[action_color][1] - self.state[action_color][0]
            # print("num_remain = ", num_remain)
            # more colors remain
            # pdb.set_trace()
            if num_remain > 0:
                self.state[action_color][0] = self.state[action_color][0] + 1
                rew = self.player_to_reward_function[player_idx][action_color]
                total_reward += (rew + self.task_rew)
                # print("New self.state = ", self.state)
            else:
                rew = -100
                total_reward += rew

        self.state[TURN_IDX][0] = 1 - self.state[TURN_IDX][0]
        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (self.state[color][1] - self.state[color][0])

        done = self.is_game_over()
        # if done:
        #     total_reward *= 10
        # print("individual rews = ", individual_rewards)
        individual_rewards = None
        return self.state, total_reward, done, individual_rewards

    def select_greedy_action_pair(self, input_state):
        current_state = copy.deepcopy(input_state)

        player_idx = input_state[TURN_IDX][0]
        selected_action = None
        valid_actions = self.get_possible_actions(current_state)

        max_rew = -100
        best_action_color = NO_MOVE
        for action_color in valid_actions:
            if action_color == NO_MOVE:
                cand_rew = -100
            else:
                if current_state[action_color][1] - current_state[action_color][0] == 0:
                    cand_rew = -100
                else:
                    cand_rew = self.player_to_reward_function[player_idx][action_color]
            if cand_rew > max_rew:
                max_rew = cand_rew
                best_action_color = action_color

        selected_action = best_action_color

        return selected_action



    def flatten_to_tuple(self, state):
        # return tuple(list(sum(state, ())))
        return tuple([item for sublist in state for item in sublist])

    def enumerate_states(self):
        self.reset()
        actions = self.actions
        # actions = [0,1,2,3,4]
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
            if self.is_game_over_given_state(state):
                available_actions = []
            else:

                available_actions = self.get_possible_actions(state)
            # available_actions = self.actions
            # print(f"state = {state}, available_actions = {available_actions}")

            # if state_tup == (1, 3, 0, 0, 9, 9, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0):
            #     pdb.set_trace()
            # if state_tup ==  (0, 1, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0):
            #     pdb.set_trace()
            # if state_tup == (2, 3, 0, 0, 2, 19, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0):
            #     pdb.set_trace()


            # get the neighbors of this state by looping through possible actions
            # available_actions = [0,1,2,3,4]
            for idx, action in enumerate(available_actions):
                # set the environment to the current state
                # self.set_to_state(copy.deepcopy(state))

                # take the action
                new_state, rew, done  = self.step_given_state(state, action)

                # print(f"s {state} action={action} --> snext {new_state}, rew={rew}")

                # if state_tup == (0, 3, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0):
                #     pdb.set_trace()

                # if done:
                #     pdb.set_trace()
                # convert new state to tuple
                new_state_tup = self.flatten_to_tuple(new_state)
                # assert new_state[-1][0] != state[-1][0]

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(new_state))

                # pdb.set_trace()
                # add edge to graph from current state to new state with weight equal to reward
                # pdb.set_trace()
                # if not self.is_game_over_given_state(state):
                #     G.add_edge(state_tup, new_state_tup, weight=rew, action=action)

                # prev_state_tup = self.flatten_to_tuple(new_state)
                # G.add_edge(prev_state_tup, state_tup, weight=rew, action=action)
                # if rew < 0:
                #     rew = rew * 10
                G.add_edge(state_tup, new_state_tup, weight=rew, action=action)


        states = list(G.nodes)
        # print("NUMBER OF STATES", len(states))
        # print("NUMBER OF ACTIONS", len(actions))
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}
        # print("action_to_idx", action_to_idx)

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
        self.transition_mat, self.reward_mat, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx



    def rollout_full_game_vi_policy(self, policy, savename='example_rollout.png'):
        # print("policy", policy)
        self.reset()
        done = False
        total_reward = 0

        human_altruism_n_instances = 0
        robot_altruism_n_instances = 0

        human_altruism_percent = 0
        robot_altruism_percent = 0

        human_trace = []
        robot_trace = []
        human_greedy_alt = []
        robot_greedy_alt = []


        while not done:
        # for i in range(10):
            current_state = copy.deepcopy(self.state)
            # print(f"current_state = {current_state}")
            current_state_tup = self.flatten_to_tuple(current_state)

            # print("availabel actions", self.get_possible_actions(current_state))
            state_idx = self.state_to_idx[current_state_tup]

            action_selected = self.idx_to_action[int(policy[state_idx, 0])]

            greedy_action = self.select_greedy_action_pair(current_state)
            # if action_selected != greedy_action:
            if current_state[-1][0]==self.index_of_robot:
                # robot_altruism_n_instances += 1
                robot_trace.append(action_selected)
                robot_greedy_alt.append(greedy_action)

            else:
                # human_altruism_n_instances += 1
                human_trace.append(action_selected)
                human_greedy_alt.append(greedy_action)

            next_state, rew, done, _ = self.step(action_selected)
            # print(f"curr state = {current_state}, action = {action_selected}, reward = {rew}, done = {done}, next={next_state}")
            # print(f"joint_action_selected = {action_selected}, rews  = {rew}")
            total_reward += rew

            # pdb.set_trace()

        # print(f"FINAL REWARD for optimal team = {total_reward}")
        # print()
        if len(human_trace) != len(robot_trace):
            enablePrint()
            print("ISSUE with OPTIMAL Length")
            print("init state", self.initial_state_input)
            print("player_to_reward_function", self.player_to_reward_function)
            print("human_trace = ", human_trace)
            print("robot_trace = ", robot_trace)
        assert len(human_trace) == len(robot_trace)
        return total_reward, human_trace, robot_trace, human_greedy_alt, robot_greedy_alt


    def rollout_full_game_greedy_pair(self, savename='example_rollout.png'):
        self.reset()
        done = False
        total_reward = 0
        # while not done:
        self.reset()
        done = False
        total_reward = 0
        human_trace = []
        robot_trace = []

        while not done:
            # for i in range(10):
            current_state = copy.deepcopy(self.state)

            action_selected = self.select_greedy_action_pair(current_state)

            if current_state[-1][0] == self.index_of_robot:
                robot_trace.append(action_selected)
            else:
                human_trace.append(action_selected)

            next_state, rew, done, _ = self.step(action_selected)
            # print(f"curr state = {current_state[-1][0]}, action = {action_selected}, reward = {rew}, done = {done}")
            # print(f"joint_action_selected = {action_selected}, rews  = {rew}")
            total_reward += rew

        # print(f"FINAL REWARD for team = {total_reward}")
        # print()
        if len(human_trace) != len(robot_trace):
            # enablePrint()
            print("ISSUE with GREEDY Length")
            print("init state", self.initial_state_input)
            print("player_to_reward_function", self.player_to_reward_function)
            print("human_trace = ", human_trace)
            print("robot_trace = ", robot_trace)
        assert len(human_trace) == len(robot_trace)
        return total_reward, human_trace, robot_trace


def value_iteration(transitions, players_to_reward, idx_to_state, state_to_idx, epsilson=0.0001, gamma=1.0, maxiter=100000):
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
    # players_to_reward = [(0.9, -0.9, 0.1, 0.3), (0.9, 0.1, -0.9, 0.2)]
    n_states = transitions.shape[0]
    # n_actions = transitions.shape[2]
    n_actions = 5

    # initialize value function
    pi = np.zeros((n_states, 1))
    vf = np.zeros((n_states, 1))
    Q = np.zeros((n_states, n_actions))

    # pi.fill(-20)
    # vf.fill(-20)
    # Q.fill(-20)

    # print("starting vi")
    for i in range(maxiter):
        # initalize delta
        delta = 0
        # perform Bellman update
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()

            # compute new Q values

            initial_state = copy.deepcopy(list(idx_to_state[s]))
            # print("initial_state", initial_state)
            initial_state = [list(initial_state[0:2]), list(initial_state[2:4]), list(initial_state[4:6]),
                         list(initial_state[6:8]), [initial_state[-1]]]
            # print("curr_init", initial_state)
            possible_actions = []
            impossible_actions = []
            for color in [0,1,2,3]:
                if initial_state[color][1]- initial_state[color][0] > 0:
                    possible_actions.append(color)
                else:
                    impossible_actions.append(color)
            # if initial_state == [[2, 2], [4, 4], [3, 3], [0, 1], [1]]:
            #     print("possible_actions", possible_actions)
            # print(f"initial_state= {initial_state}, possible actions = {possible_actions}")

            # print("possible_actions", possible_actions)
            # possible_actions = [0,1,2,3]
            if len(possible_actions) == 0:
                V_s11 = vf[s]
                Q[s, 4] = 0 + (gamma * V_s11)
            else:
                for action_idx in possible_actions:
                    initial_state = copy.deepcopy(list(idx_to_state[s]))
                    # print("initial_state", initial_state)
                    initial_state = [list(initial_state[0:2]), list(initial_state[2:4]), list(initial_state[4:6]),
                             list(initial_state[6:8]), [initial_state[-1]]]
                    # print("curr_init", initial_state)
                    possible_actions = []
                    for color in [0, 1, 2, 3]:
                        if initial_state[color][1] - initial_state[color][0] > 0:
                            possible_actions.append(color)

                    r_sa = players_to_reward[initial_state[-1][0]][action_idx]
                    # if action_idx in possible_actions:
                    #     r_sa = players_to_reward[initial_state[-1][0]][action_idx]
                    # else:
                    #     r_sa = -1000

                    # print("r_sa", r_sa)
                    # print("n_actions", n_actions)
                    if action_idx != 4:
                        if initial_state[action_idx][1] - initial_state[action_idx][0]  > 0:
                            initial_state[action_idx][0] = initial_state[action_idx][0] + 1

                    initial_state[4][0] = 1-initial_state[4][0]
                    current_state = copy.deepcopy(initial_state)

                    current_state = tuple([item for sublist in current_state for item in sublist])
                    # print("state_to_idx", state_to_idx)
                    s11 = state_to_idx[tuple(current_state)]
                    V_s11 = vf[s11]
                    Q[s, action_idx] = r_sa + (gamma * V_s11)

                # for action_idx in impossible_actions:
                #     initial_state = copy.deepcopy(list(idx_to_state[s]))
                #     # print("initial_state", initial_state)
                #     initial_state = [list(initial_state[0:2]), list(initial_state[2:4]), list(initial_state[4:6]),
                #              list(initial_state[6:8]), [initial_state[-1]]]
                #
                #     current_state = copy.deepcopy(initial_state)
                #
                #     current_state = tuple([item for sublist in current_state for item in sublist])
                #     r_sa = -1
                #     s11 = state_to_idx[tuple(current_state)]
                #     V_s11 = vf[s11]
                #     Q[s, action_idx] = -10

            vf[s] = np.max(Q[s, :], 0)
            delta = np.max((delta, np.abs(old_v - vf[s])[0]))
        # check for convergence

        print(f"i={i}, delta= {delta}")
        if delta < epsilson:
            print("finished at i=", i)
            break
    # print("finished at max iters=", maxiter)
    # compute optimal policy
    for s in range(n_states):
        # store old value function
        old_v = vf[s].copy()

        # compute new Q values
        initial_state = copy.deepcopy(list(idx_to_state[s]))
        # print("initial_state", initial_state)
        initial_state = [list(initial_state[0:2]), list(initial_state[2:4]), list(initial_state[4:6]),
                         list(initial_state[6:8]), [initial_state[-1]]]
        # print("curr_init", initial_state)
        possible_actions = []
        impossible_actions = []
        for color in [0, 1, 2, 3]:
            if initial_state[color][1] - initial_state[color][0] > 0:
                possible_actions.append(color)
            else:
                impossible_actions.append(color)

        # possible_actions = [0, 1, 2, 3]
        # if len(possible_actions) == 0:
        # possible_actions = [0,1,2,3,4]
        for action_idx in possible_actions:
            initial_state = copy.deepcopy(list(idx_to_state[s]))
            # print("initial_state", initial_state)
            initial_state = [list(initial_state[0:2]), list(initial_state[2:4]), list(initial_state[4:6]),
                         list(initial_state[6:8]), [initial_state[-1]]]
            # print("curr_init", initial_state)
            possible_actions = []
            for color in [0, 1, 2, 3]:
                if initial_state[color][1] - initial_state[color][0] > 0:
                    possible_actions.append(color)

            r_sa = players_to_reward[initial_state[-1][0]][action_idx]
            # if action_idx in possible_actions:
            #     r_sa = players_to_reward[initial_state[-1][0]][action_idx]
            # else:
            #     r_sa = -1000
            #     if sum(initial_state[color][1]-initial_state[color][0] for color in [0,1,2,3])==0:
            #         r_sa = 0
            # print("n_actions", n_actions)
            if action_idx != 4:
                # print('initial_state[action_idx]', initial_state[action_idx])
                if initial_state[action_idx][1] - initial_state[action_idx][0] > 0:
                    initial_state[action_idx][0] = initial_state[action_idx][0] + 1


            current_state = copy.deepcopy(initial_state)
            current_state[4][0] = 1 - current_state[4][0]
            assert current_state[-1][0] != initial_state[-1][0]

            current_state = tuple([item for sublist in current_state for item in sublist])
            # print("state_to_idx", state_to_idx)
            s11 = state_to_idx[tuple(current_state)]
            V_s11 = vf[s11]
            Q[s, action_idx] = r_sa + (gamma * V_s11)

        # pdb.set_trace()
        # pi[s] = np.argmax(Q[s, :], 0)
        max_q = -1000000
        best_a = None
        for cand_a in possible_actions:
            if Q[s, cand_a] > max_q:
                max_q = Q[s, cand_a]
                best_a = cand_a
        if best_a is None:
            # raise ArithmeticError
            # enablePrint()
            # print("PROBLEM")
            # print("s", idx_to_state[s])
            # print("possible_actions", possible_actions)
            pi[s] = NO_MOVE
        else:
            pi[s] = best_a
        # policy[s] = Q[s, :]

    return vf, pi


def compare_optimal_to_greedy(players_to_reward, initial_state, h_player_idx, r_player_idx):
    blockPrint()
    env = Joint_MDP(index_of_human=h_player_idx, index_of_robot=r_player_idx, players_to_reward=players_to_reward, initial_state=initial_state)
    transitions, rewards, state_to_idx, idx_to_action, idx_to_state, action_to_idx = env.enumerate_states()

    # compute optimal policy with value iteration
    # print("running value iteration...")
    values, policy = value_iteration(transitions, rewards, idx_to_state, state_to_idx)
    # policy = find_policy(n_states=len(state_to_idx), n_actions=len(idx_to_action), transition_probabilities=transitions,
    #                              reward=rewards, discount=0.99,
    #             threshold=1e-2, v=None, stochastic=True, max_iters=10)
    # print("...completed value iteration")
    # print("policy", policy)
    # return

    optimal_rew, human_trace_opt, robot_trace_opt, human_greedy_alt, robot_greedy_alt = env.rollout_full_game_vi_policy(policy)

    greedy_rew, human_trace_greedy, robot_trace_greedy = env.rollout_full_game_greedy_pair()

    human_altruism_n_instances, robot_altruism_n_instances = 0,0
    non_overlap_list = []
    if sorted(robot_trace_greedy) == sorted(robot_trace_opt):
        # if optimal_rew == greedy_rew is False:
        #     print()
        #     print("robot_trace_greedy", robot_trace_greedy)
        #     print("robot_trace_opt", robot_trace_opt)
        #     print("optimal_rew == greedy_rew = ", optimal_rew == greedy_rew)
        assert abs(optimal_rew-greedy_rew) < 0.01

    else:
        assert sorted(human_trace_greedy) != sorted(human_trace_opt)

        print()
        human_in_both = []
        only_in_human_opt = []
        only_in_human_greedy = []
        copy_of_human_trace_greedy = copy.deepcopy(human_trace_greedy)

        print("human_trace_opt", human_trace_opt)
        print("human_trace_greedy", human_trace_greedy)



        for i in range(len(human_trace_opt)):
            # print("copy_of_human_trace_greedy = ", copy_of_human_trace_greedy)
            if human_trace_opt[i] in copy_of_human_trace_greedy:
                human_in_both.append(human_trace_opt[i])
                copy_of_human_trace_greedy.remove(human_trace_opt[i])
            else:
                only_in_human_opt.append(human_trace_opt[i])
        only_in_human_greedy.extend(copy_of_human_trace_greedy)

        print("only_in_human_opt = ", only_in_human_opt)
        print('only_in_human_greedy = ', only_in_human_greedy)

        robot_in_both = []
        only_in_robot_opt = []
        only_in_robot_greedy = []
        copy_of_robot_trace_greedy = copy.deepcopy(robot_trace_greedy)
        for i in range(len(robot_trace_opt)):
            if robot_trace_opt[i] in copy_of_robot_trace_greedy:
                robot_in_both.append(robot_trace_opt[i])
                copy_of_robot_trace_greedy.remove(robot_trace_opt[i])
            else:
                only_in_robot_opt.append(robot_trace_opt[i])
        only_in_robot_greedy.extend(copy_of_robot_trace_greedy)

        print("human_trace_opt = ", human_trace_opt)
        print("human_greedy_alt = ", human_greedy_alt)
        print("only_in_human_opt = ", only_in_human_opt)
        print('only_in_human_greedy = ', only_in_human_greedy)
        for i in range(len(only_in_human_opt)):
            print("players_to_reward[1][only_in_human_opt[i]] = ", players_to_reward[h_player_idx][only_in_human_opt[i]])
            print("players_to_reward[1][only_in_human_greedy[i]] = ", players_to_reward[h_player_idx][only_in_human_greedy[i]])
            if players_to_reward[h_player_idx][only_in_human_opt[i]] <= players_to_reward[h_player_idx][only_in_human_greedy[i]]:
                human_altruism_n_instances += 1
            # else:
            #     robot_altruism_n_instances += 1
            # if human_trace_opt[i] in human_in_both:
            #     human_in_both.remove(human_trace_opt[i])
            # elif human_trace_opt[i] in only_in_human_opt:
            #     if human_trace_opt[i] != human_greedy_alt[i]:
            #         print(f"i={i}: human_trace_opt[i]={human_trace_opt[i]}")
            #         human_altruism_n_instances += 1
            #     only_in_human_opt.remove(human_trace_opt[i])
        print("human_altruism_n_instances = ", human_altruism_n_instances)
        print()

        print("robot_trace_opt", robot_trace_opt)
        print("robot_trace_greedy", robot_trace_greedy)
        print()
        print("robot_trace_opt = ", robot_trace_opt)
        print("robot_greedy_alt = ", robot_greedy_alt)
        print("only_in_robot_opt = ", only_in_robot_opt)
        print('only_in_robot_greedy = ', only_in_robot_greedy)
        for i in range(len(only_in_robot_opt)):
            print("players_to_reward[0][only_in_robot_opt[i]] = ", players_to_reward[r_player_idx][only_in_robot_opt[i]])
            print("players_to_reward[0][only_in_robot_greedy[i]] = ", players_to_reward[r_player_idx][only_in_robot_greedy[i]])
            if players_to_reward[r_player_idx][only_in_robot_opt[i]] <= players_to_reward[r_player_idx][only_in_robot_greedy[i]]:
                robot_altruism_n_instances += 1
            # else:
            #     robot_altruism_n_instances

            # if robot_trace_opt[i] in only_in_robot_opt:
            #     if robot_trace_opt[i] != robot_greedy_alt[i]:
            #         print(f"i={i}: robot_trace_opt[i]={robot_trace_opt[i]}")
            #         robot_altruism_n_instances += 1
            #     only_in_robot_opt.remove(robot_trace_opt[i])
            #
            # elif robot_trace_opt[i] in robot_in_both:
            #     print(f"Removing i={i}: robot_trace_opt[i]={robot_trace_opt[i]}")
            #     robot_in_both.remove(robot_trace_opt[i])
            #     print("robot_in_both", robot_in_both)
        print("robot_altruism_n_instances = ", robot_altruism_n_instances)
        print()
        print()

    enablePrint()
    return optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances


def randomList(m, n):
    # Create an array of size m where
    # every element is initialized to 0
    arr = [0] * m

    # To make the sum of the final list as n
    for i in range(n):
        # Increment any random element
        # from the array by 1
        arr[random.randint(0, n) % m] += 1

    # Print the generated list
    return arr

def sample_configurations():
    blockPrint()
    total = 1000
    n_mutual = 0
    n_human = 0
    n_robot = 0
    n_greedy = 0

    for i in range(total):
        # print("i = ", i)
        initial_state = randomList(4, 10)

        corpus = (np.round(np.random.uniform(-1,1), 1), np.round(np.random.uniform(-1,1), 1),
                  np.round(np.random.uniform(-1,1), 1), np.round(np.random.uniform(-1,1), 1))

        robot_rewards = corpus
        human_rewards = list(corpus)
        random.shuffle(human_rewards)
        human_rewards = tuple(human_rewards)

        h_player_idx = np.random.choice([0, 1])
        r_player_idx = 1 - h_player_idx
        if h_player_idx == 0:
            players_to_reward = [human_rewards, robot_rewards]
        else:
            players_to_reward = [robot_rewards, human_rewards]
        # players_to_reward = [robot_rewards, human_rewards]

        print(f"initial_state = {initial_state}, players_to_reward={players_to_reward}")

        optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances = compare_optimal_to_greedy(
            players_to_reward, initial_state, h_player_idx, r_player_idx)

        if i % 100==0:
            enablePrint()
            print()
            print("i = ", i)
            print(f"optimal_rew = {optimal_rew}, greedy_rew = {greedy_rew}")
            if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
                print("mutual")
            elif human_altruism_n_instances > 0:
                print("robot altruism")
            elif robot_altruism_n_instances > 0:
                print("human altruism")
            else:
                print("greedy optimal")
            blockPrint()

        if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
            n_mutual += 1
        elif human_altruism_n_instances > 0:
            n_human += 1
        elif robot_altruism_n_instances > 0:
            n_robot += 1
        else:
            print(f"optimal_rew = {optimal_rew}, greedy_rew = {greedy_rew}")
            eps = 0.01
            assert abs(optimal_rew - greedy_rew) < eps
            n_greedy += 1

    enablePrint()
    n_mutual /= total
    n_human /= total
    n_robot /= total
    n_greedy /= total
    plt.bar(['mutual', 'human', 'robot', 'greedy'], [n_mutual, n_human, n_robot, n_greedy], color='maroon',
            width=0.4)

    plt.xlabel("Game Type")
    plt.ylabel(f"Percent of Instances in {total} rounds")
    plt.title("Frequency of Game Types")
    plt.show()

if __name__ == "__main__":

    sample_configurations()
    # players_to_reward = [(0.9, -0.9, 0.1, 0.3), (0.9, 0.1, -0.9, 0.2)]
    # initial_state = [2, 5, 1, 2]
    # optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances = compare_optimal_to_greedy(players_to_reward, initial_state)
    # print(f"optimal_rew = {optimal_rew}, greedy_rew = {greedy_rew}")


