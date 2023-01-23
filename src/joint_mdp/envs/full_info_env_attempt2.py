import pdb

import numpy as np
import copy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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



class Joint_MDP:
    def __init__(self, index_of_human, index_of_robot, players_to_reward, initial_state):

        self.initial_state = [[0, initial_state[0]], [0, initial_state[1]], [0, initial_state[2]],
                              [0, initial_state[3]], [0]]

        self.index_of_human = index_of_human
        self.index_of_robot = index_of_robot

        self.player_to_reward_function = players_to_reward

        self.actions = ACTION_LIST
        self.state = copy.deepcopy(self.initial_state)


    def get_all_joint_actions(self):
        all_joint_actions = []
        for p1_color in ACTION_LIST:
            for p2_color in ACTION_LIST:
                all_joint_actions.append((p1_color, p2_color))
        return all_joint_actions


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

        return valid_joint_actions
        # return ACTION_LIST


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
                total_reward += rew
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
                total_reward += rew
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

        max_rew = -1
        best_action_color = NO_MOVE
        for action_color in ACTION_LIST:
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

            # if state_tup == (1, 3, 0, 0, 9, 9, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0):
            #     pdb.set_trace()
            # if state_tup ==  (0, 1, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0):
            #     pdb.set_trace()
            # if state_tup == (2, 3, 0, 0, 2, 19, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0):
            #     pdb.set_trace()


            # get the neighbors of this state by looping through possible actions
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
        print("NUMBER OF STATES", len(states))
        print("NUMBER OF ACTIONS", len(actions))
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

        while not done:
        # for i in range(10):
            current_state = copy.deepcopy(self.state)
            # print(f"current_state = {current_state}")
            current_state_tup = self.flatten_to_tuple(current_state)
            state_idx = self.state_to_idx[current_state_tup]

            action_selected = self.idx_to_action[int(policy[state_idx, 0])]

            greedy_action = self.select_greedy_action_pair(current_state)
            if action_selected != greedy_action:
                if current_state[-1][0]==0:
                    robot_altruism_n_instances += 1
                else:
                    human_altruism_n_instances += 1

            next_state, rew, done, _ = self.step(action_selected)
            # print(f"curr state = {current_state[-1][0]}, action = {action_selected}, reward = {rew}, done = {done}")
            # print(f"joint_action_selected = {action_selected}, rews  = {rew}")
            total_reward += rew

            # pdb.set_trace()

        # print(f"FINAL REWARD for optimal team = {total_reward}")
        # print()
        return total_reward, human_altruism_n_instances, robot_altruism_n_instances


    def rollout_full_game_greedy_pair(self, savename='example_rollout.png'):
        self.reset()
        done = False
        total_reward = 0
        # while not done:
        self.reset()
        done = False
        total_reward = 0
        while not done:
            # for i in range(10):
            current_state = copy.deepcopy(self.state)

            action_selected = self.select_greedy_action_pair(current_state)
            # pdb.set_trace()

            # pdb.set_trace()
            # is_valid = self.check_is_valid_joint_action(current_state, joint_action_selected)

            # print(f"state_idx = {current_state_tup}, color_selected = {joint_action_selected}, is_valid = {is_valid}")
            # pdb.set_trace()

            next_state, rew, done, _ = self.step(action_selected)
            # print(f"curr state = {current_state[-1][0]}, action = {action_selected}, reward = {rew}, done = {done}")
            # print(f"joint_action_selected = {action_selected}, rews  = {rew}")
            total_reward += rew

        # print(f"FINAL REWARD for team = {total_reward}")
        # print()
        return total_reward


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
    n_actions = transitions.shape[2]

    # initialize value function
    pi = np.zeros((n_states, 1))
    vf = np.zeros((n_states, 1))
    Q = np.zeros((n_states, n_actions))
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
            for color in [0,1,2,3]:
                if initial_state[color][1]- initial_state[color][0] > 0:
                    possible_actions.append(color)

            # print("possible_actions", possible_actions)
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

            vf[s] = np.max(Q[s, :], 0)
            delta = np.max((delta, np.abs(old_v - vf[s])[0]))
        # check for convergence
        if delta < epsilson:
            # print("finished at i=", i)
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
        for color in [0, 1, 2, 3]:
            if initial_state[color][1] - initial_state[color][0] > 0:
                possible_actions.append(color)


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
        pi[s] = np.argmax(Q[s, :], 0)
        # policy[s] = Q[s, :]

    return vf, pi


def compare_optimal_to_greedy():
    players_to_reward = [(0.9, -0.9, 0.1, 0.3), (0.9, 0.1, -0.9, 0.2)]
    initial_state = [2,5,1,2]
    env = Joint_MDP(index_of_human=1, index_of_robot=2, players_to_reward=players_to_reward, initial_state=initial_state)
    transitions, rewards, state_to_idx, idx_to_action, idx_to_state, action_to_idx = env.enumerate_states()

    # compute optimal policy with value iteration
    print("running value iteration...")
    values, policy = value_iteration(transitions, rewards, idx_to_state, state_to_idx )
    # policy = find_policy(n_states=len(state_to_idx), n_actions=len(idx_to_action), transition_probabilities=transitions,
    #                              reward=rewards, discount=0.99,
    #             threshold=1e-2, v=None, stochastic=True, max_iters=10)
    print("...completed value iteration")
    # print("policy", policy)
    # return

    optimal_rew, human_altruism_n_instances, robot_altruism_n_instances = env.rollout_full_game_vi_policy(policy)
    print(f"FINAL REWARD for optimal team = {optimal_rew}")

    if human_altruism_n_instances > 0 and robot_altruism_n_instances > 0:
        print("mutual")
    elif human_altruism_n_instances > 0:
        print("human altruism")
    elif robot_altruism_n_instances > 0:
        print("robot altruism")
    else:
        print("greedy optimal")

    # env = Joint_MDP_Original(index_of_human=1, index_of_robot=2, players_to_reward=players_to_reward)
    greedy_rew = env.rollout_full_game_greedy_pair()
    print(f"FINAL REWARD for greedy team = {greedy_rew}")

    return optimal_rew, greedy_rew, human_altruism_n_instances, robot_altruism_n_instances


if __name__ == '__main__':
    compare_optimal_to_greedy()