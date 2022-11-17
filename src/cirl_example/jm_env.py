import copy
import pdb
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from value_iteration import value_iteration, find_policy


EMPTY_VAL = 0
POS_VAL = 1
PRIZE_VAL = 2

NORTH = (-1, 0)
SOUTH = (1, 0)
EAST = (0, 1)
WEST = (0, -1)
NO_ACTION = (0, 0)

# even is human
# odd is robot


class JMEnv():
    def __init__(self):
        self.state = [np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0]]), 0]

        self.n_rows, self.n_cols = self.state[0].shape
        self.actions = [NORTH, SOUTH, EAST, WEST, NO_ACTION]
        self.prize_val_locations = [(1,1), (1,3)]
        self.max_steps = 5

    def get_candidate_trajs(self):
        candidate_trajs = [[(2,2), (1,2), (1,1)],
                           [(2,2), (1,2), (1,3)],
                           [(2,2), (1,2), (1,3), (1,2), (1,1)]]

        return candidate_trajs


    def step(self, action):
        current_position = list(zip(*np.where(self.state[0] == POS_VAL)))[0]
        # pdb.set_trace()
        new_position = (current_position[0] + action[0], current_position[1] + action[1])
        is_valid = True
        if new_position[0] < 0 or new_position[0] >= self.n_rows:
            is_valid = False
        if new_position[1] < 0 or new_position[1] >= self.n_cols:
            is_valid = False

        rew = -1
        if action == NO_ACTION:
            rew = 0
        done = False
        self.state[1] += 1
        if is_valid:
            self.state[0][current_position] = EMPTY_VAL

            # if self.state[0][new_position] == PRIZE_VAL:
            if new_position in self.prize_val_locations:
                self.prize_val_locations.remove(new_position)
                # pdb.set_trace()
                rew = 10

            self.state[0][new_position] = POS_VAL

            if self.state[1] >= self.max_steps:
                done = True
        return self.state, rew, done


    def reset(self):
        self.state = [np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0]]), 0]
        self.prize_val_locations = [(1, 1), (1, 3)]

    def is_done_state(self, state):
        if state[1] >= self.max_steps:
            return True
        return False

    def get_available_actions_in_state(self, state):
        available = []
        for action in self.actions:
            current_position = list(zip(*np.where(state[0] == POS_VAL)))[0]
            # pdb.set_trace()
            new_position = (current_position[0] + action[0], current_position[1] + action[1])
            is_valid = True
            if new_position[0] < 0 or new_position[0] >= self.n_rows:
                is_valid = False
            if new_position[1] < 0 or new_position[1] >= self.n_cols:
                is_valid = False

            if is_valid:
                available.append(action)

        return available


    def set_to_state(self, flat_state):
        state_array = flat_state[:-1]
        timestep = flat_state[-1]

        state_grid = np.reshape(state_array, (3, 5))
        self.state = [state_grid, timestep]


    def flatten_to_tuple(self, state):
        # return tuple(list(sum(state, ())))
        grid = state[0]
        timestep = state[1]
        flat_state = [item for sublist in grid for item in sublist]
        flat_state.append(timestep)
        return tuple(flat_state)

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

            if self.is_done_state(state):
                available_actions = []
            else:
                available_actions = self.get_available_actions_in_state(state)


            # available_actions = self.actions




            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(available_actions):
                # set the environment to the current state
                self.set_to_state(copy.deepcopy(state))

                # take the action
                new_state, rew, done = self.step(action)

                # convert new state to tuple
                new_state_tup = self.flatten_to_tuple(new_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(new_state))

                # add edge to graph from current state to new state with weight equal to reward
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


        self.transition_mat, self.reward_mat, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx
        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def rollout_full_game_vi_policy(self, policy, savename='example_rollout.png'):
        self.reset()
        done = False
        total_reward = 0
        print(f"initial state = \n{self.state}")
        while not done:
            current_state = self.state
            # print(f"current_state = {current_state}")
            current_state_tup = self.flatten_to_tuple(current_state)
            state_idx = self.state_to_idx[current_state_tup]

            action_selected = self.idx_to_action[int(policy[state_idx, 0])]


            next_state, rew, done = self.step(action_selected)
            # print(f"next_state = {next_state}, reward = {joint_reward}, done = {done}")
            print(f"\n\nstate = \n{next_state}, rews  = {rew}")

            total_reward += rew
            # pdb.set_trace()

        # print(f"FINAL REWARD for optimal team = {total_reward}")
        # print()
        return total_reward


    def select_br_trajectory(self):
        candidate_trajs = self.get_candidate_trajs()
        expert_policy_traj = [(2,2), (1,2), (1,1)]

        expert_traj_feature = self.expected_feat_traj(expert_policy_traj)

        eta = 0.2
        max_val = -1000
        best_traj = None
        for cand_traj in candidate_trajs:
            cand_traj_feat = self.expected_feat_traj(cand_traj)
            l_dist = l2(expert_traj_feature, cand_traj_feat)
            rew = np.dot(cand_traj_feat, [10,10])
            val = rew - eta * l_dist
            if val > max_val:
                max_val = val
                best_traj = cand_traj

        return best_traj


    def expected_feat_traj(self, traj):
        featurized = []
        for elem in traj:
            current_position = elem

            d1 = l2(current_position, (1, 1))
            d2 = l2(current_position, (1, 3))

            f1 = 0
            f2 = 0
            if d1 == 0:
                f1 = 1
            if d2 == 0:
                f2 = 1

            featurized.append([f1, f2])

        expected = [sum([elem[0] for elem in featurized]), sum([elem[1] for elem in featurized])]
        return expected

    def featurize_state(self, state):
        featurized = []
        grid = state[0]
        current_position = list(zip(*np.where(grid == POS_VAL)))[0]
        d1 = l2(current_position, (1,1))
        d2 = l2(current_position, (1, 3))

        f1 = 0
        f2 = 0
        if d1 == 0:
            f1 = 1
        if d2 == 0:
            f2 = 1

        return [f1, f2]


def l2(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


def test_vi_functionality():
    env = JMEnv()
    transitions, rewards, state_to_idx, idx_to_action, idx_to_state, action_to_idx = env.enumerate_states()

    best_br_traj = env.select_br_trajectory()
    print("best_br_traj", best_br_traj)

    # # compute optimal policy with value iteration
    # print("running value iteration...")
    # values, policy = value_iteration(transitions, rewards)
    # # policy = find_policy(n_states=len(state_to_idx), n_actions=len(idx_to_action), transition_probabilities=transitions,
    # #                              reward=rewards, discount=0.99,
    # #             threshold=1e-2, v=None, stochastic=True, max_iters=10)
    # print("...completed value iteration")
    # # print("policy", policy)
    # # return
    #
    # vi_rew = env.rollout_full_game_vi_policy(policy)
    # print(f"FINAL REWARD for VI = {vi_rew}")


if __name__ == "__main__":
    test_vi_functionality()