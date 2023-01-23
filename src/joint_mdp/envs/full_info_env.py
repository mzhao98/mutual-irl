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
ACTION_LIST = [BLUE, GREEN, RED, YELLOW, NO_MOVE]
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
NUM_PLAYERS = 2
COLOR_TO_TEXT = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}



class Joint_MDP_Original:
    def __init__(self, index_of_human, index_of_robot, players_to_reward, initial_state=None):
        if initial_state is None:
            self.initial_state = [[0, 2, 0, 0], [0, 5, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]
        else:
            self.initial_state = initial_state

        self.index_of_human = index_of_human
        self.index_of_robot = index_of_robot

        self.player_to_reward_function = players_to_reward

        self.actions = self.get_all_joint_actions()
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
        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (state[color][1] - state[color][0])

        if num_remaining == 0:
            return []

        valid_joint_actions = []
        for joint_action in self.actions:
            if self.check_is_valid_joint_action(state, joint_action):
                valid_joint_actions.append(joint_action)

        return valid_joint_actions

    def check_is_valid_joint_action(self, current_state, joint_action):
        state = copy.deepcopy(current_state)

        is_valid = True
        for player_idx in range(NUM_PLAYERS):
            action_color = joint_action[player_idx]

            if action_color == NO_MOVE:
                # do_nothing = True
                if self.get_num_remaining_objects_in_state(state) > 0:
                    is_valid = False

            else:
                num_remain = state[action_color][1] - state[action_color][0]
                # No more colors remain, invalid action
                if num_remain <= 0:
                    is_valid = False
                else:
                    state[action_color][0] = state[action_color][0] + 1

        return is_valid



    def set_to_state(self, state):
        self.state = state


    def step_given_state(self, current_state, joint_action):
        # joint_action = (p1 action color, p2 action color)

        state = copy.deepcopy(current_state)

        total_reward = 0
        # Resolve player action
        for player_idx in range(NUM_PLAYERS):
            action_color = joint_action[player_idx]

            if action_color == NO_MOVE:
                rew = -1
                total_reward += rew
            else:
                num_remain = state[action_color][1] - state[action_color][0]
                # more colors remain
                # pdb.set_trace()
                if num_remain > 0:
                    state[action_color][0] = state[action_color][0] + 1
                    rew = self.player_to_reward_function[player_idx][action_color]
                    total_reward += rew

        num_remaining = 0
        for color in COLOR_LIST:
            num_remaining += (state[color][1] - state[color][0])
        done = (True if num_remaining == 0 else False)

        return state, total_reward, done


    def step(self, joint_action):
        # joint_action = (p1 action color, p2 action color)
        individual_rewards = []
        total_reward = 0
        # Resolve player action
        for player_idx in range(NUM_PLAYERS):
            action_color = joint_action[player_idx]

            if action_color == NO_MOVE:
                # rew = -100
                if self.get_num_remaining_objects() == 0:
                    rew = 0
                else:
                    rew = -1
                total_reward += rew
            else:
                num_remain = self.state[action_color][1] - self.state[action_color][0]
                # more colors remain
                # pdb.set_trace()
                if num_remain > 0:
                    self.state[action_color][0] = self.state[action_color][0] + 1
                    rew = self.player_to_reward_function[player_idx][action_color]

                    total_reward += rew
                else:
                    rew = -1
                    total_reward += rew


            individual_rewards.append(rew)

        done = self.is_game_over()
        # if done:
        #     total_reward *= 10
        # print("individual rews = ", individual_rewards)
        return self.state, total_reward, done, individual_rewards

    def select_greedy_action_pair(self, input_state):
        current_state = copy.deepcopy(input_state)
        action_pair = []
        valid_actions = self.get_possible_actions(current_state)

        for player_idx in range(NUM_PLAYERS):
            max_rew = -1
            best_action_color = NO_MOVE
            for action_color in ACTION_LIST:
                if action_color == NO_MOVE:
                    cand_rew = -1
                else:
                    if current_state[action_color][1] - current_state[action_color][0] == 0:
                        cand_rew = -100
                    else:
                        cand_rew = self.player_to_reward_function[player_idx][action_color]
                if cand_rew > max_rew:
                    max_rew = cand_rew
                    best_action_color = action_color
            action_pair.append(best_action_color)
            if player_idx == 0:
                current_state, _rew, _done = self.step_given_state(current_state, (best_action_color, NO_MOVE))
                if _done:
                    action_pair.append(NO_MOVE)
                    break

        action_pair = tuple(action_pair)
        # pdb.set_trace()
        assert action_pair in valid_actions
        return action_pair



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
                self.set_to_state(copy.deepcopy(state))

                # take the action
                new_state, rew, done, _= self.step(action)

                # if state_tup == (0, 3, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0):
                #     pdb.set_trace()

                # if done:
                #     pdb.set_trace()
                # convert new state to tuple
                new_state_tup = self.flatten_to_tuple(new_state)

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
        self.transition_mat, self.reward_mat, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx



    def rollout_full_game_vi_policy(self, policy, savename='example_rollout.png'):
        self.reset()
        done = False
        total_reward = 0
        while not done:
            current_state = self.state
            # print(f"current_state = {current_state}")
            current_state_tup = self.flatten_to_tuple(current_state)
            state_idx = self.state_to_idx[current_state_tup]

            joint_action_selected = self.idx_to_action[int(policy[state_idx, 0])]
            # pdb.set_trace()

            # pdb.set_trace()
            is_valid = self.check_is_valid_joint_action(current_state, joint_action_selected)

            # print(f"state_idx = {current_state_tup}, color_selected = {joint_action_selected}, is_valid = {is_valid}")
            # pdb.set_trace()

            next_state, joint_reward, done, individual_rewards = self.step(joint_action_selected)
            # print(f"next_state = {next_state}, reward = {joint_reward}, done = {done}")
            print(f"joint_action_selected = {joint_action_selected}, rews  = {individual_rewards}")
            total_reward += joint_reward
            # pdb.set_trace()

        # print(f"FINAL REWARD for optimal team = {total_reward}")
        # print()
        return total_reward


    def rollout_full_game_greedy_pair(self, savename='example_rollout.png'):
        self.reset()
        done = False
        total_reward = 0
        while not done:
            current_state = self.state
            # print(f"current_state = {current_state}")
            current_state_tup = self.flatten_to_tuple(current_state)

            joint_action_selected = self.select_greedy_action_pair(current_state)
            # pdb.set_trace()

            # pdb.set_trace()
            is_valid = self.check_is_valid_joint_action(current_state, joint_action_selected)

            # print(f"state_idx = {current_state_tup}, color_selected = {joint_action_selected}, is_valid = {is_valid}")
            # print("joint_action_selected = ", joint_action_selected)
            # pdb.set_trace()
            next_state, joint_reward, done, individual_rewards = self.step(joint_action_selected)
            # print(f"next_state = {next_state}, reward = {joint_reward}, done = {done}")

            print(f"joint_action_selected = {joint_action_selected}, rews  = {individual_rewards}")
            total_reward += joint_reward
            # print()

        # print(f"FINAL REWARD for team = {total_reward}")
        # print()
        return total_reward




