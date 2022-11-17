import copy

import numpy as np
import networkx as nx
from collections import defaultdict

from environment import BaseEnv
from value_iteration import value_iteration


class PartnerPolicy():
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)

        self.moves = [self.NORTH, self.SOUTH, self.EAST, self.WEST]
        self.idx_to_moves = dict(enumerate(self.moves))
        self.moves_to_idx = {v: k for k, v in self.idx_to_moves.items()}

        self.grid = np.zeros((5,5))
        self.grid_to_action = np.array([[self.EAST, self.EAST, self.EAST, self.EAST, self.EAST],
                                        [self.NORTH, self.NORTH, self.NORTH, self.EAST, self.NORTH],
                                        [self.NORTH, self.NORTH, self.EAST, self.EAST, self.NORTH],
                                        [self.NORTH, self.EAST, self.SOUTH, self.EAST, self.NORTH],
                                        [self.NORTH, self.EAST, self.EAST, self.EAST, self.NORTH]])


    def policy(self, position):
        # print("position", position)
        selected_action = self.grid_to_action[position]
        # print("selected_action", selected_action)
        # print("moves_to_idx", self.moves_to_idx)
        selected_action_idx = self.moves_to_idx[tuple(selected_action)]

        # return the index of action
        return selected_action_idx


class JMEnv(BaseEnv):
    """
    Joint Manipulation MDP environment. 2 player task where they must visit a series of locations on the grid.
    """

    def __init__(self, feat_weights: np.array):
        self.feat_weights = feat_weights

        # define actions
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)

        self.moves = [self.NORTH, self.SOUTH, self.EAST, self.WEST]

        # define grid
        self.grid = np.array([[0,0,0,0,0],
                              [0,0,-1,-1,0],
                              [0,0,0,-1,0],
                              [0,-1,0,0,0],
                              [0,0,0,0,0],])
        self.grid_shape = self.grid.shape

        # define puddle locations
        self.puddle_locations = [(3,1), (1,2), (1,3), (2,3)]


        # define start positions of joint particle
        self.start_position = (4,0)

        # define end goal position of joint particle
        self.goal_position = [(0,4), (0,3), (1,4)]


        # store a map of whether each position on the grid is puddle
        self.is_puddle = {}
        for i in range(5):
            for j in range(5):
                self.is_puddle[(i, j)] = (i, j) in self.puddle_locations

        # construct partner policy (makes more sense here than to be passed in)
        self.partner = PartnerPolicy()

        # define indices of state vector
        # self.ego_idx = range(0, 2)
        # self.partner_idx = range(2, 4)
        # self.victim_idxs = range(4, len(self.victim_locs) + 4)
        # self.n = self.victim_idxs[-1] + 1

        # check that all victim locations are within grid bounds

        # Preferred policy
        self.preferred_actions = np.array([[self.EAST, self.EAST, self.EAST, self.EAST, self.EAST],
                                            [self.EAST, self.SOUTH, self.SOUTH, self.EAST, self.NORTH],
                                            [self.EAST, self.EAST, self.SOUTH, self.EAST, self.NORTH],
                                            [self.SOUTH, self.SOUTH, self.EAST, self.EAST, self.NORTH],
                                            [self.EAST, self.EAST, self.EAST, self.EAST, self.NORTH]])

        self.previous_ego_action = None
        self.previous_partner_action = None
        self.previous_position = None
        self.state = None
        # self.state = self.featurize_state(ego_action, partner_action, self.current_position, self.previous_position)

        self.reset()

    def step(self, action):
        # action contains ego agent's action
        self.previous_position = self.current_position

        # print("action", action)
        ego_action = self.moves[action]

        # each action is an array [x movement, y movement]
        # print("self. current", self.current_position)

        partner_action_i = self.partner.policy(self.current_position)
        partner_action = self.moves[partner_action_i]



        new_candidate_position = (self.current_position[0] + ego_action[0] + partner_action[0],
                                  self.current_position[1] + ego_action[1] + partner_action[1])


        # if can move
        if 0 <= new_candidate_position[0] < 5 and 0 <= new_candidate_position[1] < 5:
            self.current_position = new_candidate_position



        # check if each agent has reached a victim
        # reward = 0
        # if self.is_puddle[self.current_position]:
        #     reward -= 10

        # update state
        self.state = self.featurize_state(ego_action, partner_action, self.current_position, self.previous_position)

        # check if all victims have been visited
        if self.current_position in self.goal_position:
            # reward = 100
            reward = self.state @ self.feat_weights
            done = True
        else:
            reward = self.state @ self.feat_weights
            done = False

        # update previous
        self.previous_ego_action = ego_action
        # self.previous_partner_action = partner_action


        return self.state, reward, done, partner_action_i



    def featurize_state(self, ego_action, partner_action, current_pos, prev_pos):
        # featurize based on desired reward function
        # current featurization: [following preferred policy, following prev partner, in puddle, reached destination, step cost]

        following_prev_partner = 0
        if ego_action == partner_action:
            following_prev_partner = 1

        following_ego_preferred_policy = 0
        # print(f"ego_action = {ego_action}, {self.preferred_actions[prev_pos]}")
        if ego_action == tuple(self.preferred_actions[prev_pos]):
            following_ego_preferred_policy = 1

        in_puddle = 0
        if current_pos in self.puddle_locations:
            in_puddle = 1

        reached_destination = 0
        if current_pos in self.goal_position:
            reached_destination = 1

        step = 1

        # Make state
        state = [following_ego_preferred_policy, following_prev_partner, in_puddle, reached_destination, step]

        return np.array(state)



    def reset(self):
        self.current_position = self.start_position

        return


    def set_to_state(self, state):
        """
        Set the environment to a given state.
        """
        pass

    def enumerate_states(self):
        self.reset()

        states = [(i,j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1])]

        actions = list(self.moves)
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}
        action_idx = {action: i for i, action in enumerate(self.moves)}
        idx_to_action = {i: action for i, action in enumerate(self.moves)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        # print("idx_to_state", idx_to_state)

        for i in range(len(states)):
            for a in range(len(actions)):
                current_loc = states[i]

                ego_a = actions[a]

                partner_a = actions[self.partner.policy(current_loc)]

                new_candidate_position = (current_loc[0] + ego_a[0] + partner_a[0],
                                          current_loc[1] + ego_a[1] + partner_a[1])

                # if can move
                if 0 <= new_candidate_position[0] < 5 and 0 <= new_candidate_position[1] < 5:
                    new_loc = new_candidate_position
                else:
                    new_loc = current_loc

                next_state_i = state_to_idx[new_loc]

                transition_mat[i, next_state_i, a] = 1.0

                # if new_loc in self.goal_position:
                #     rew = 100
                # else:
                rew = self.featurize_state(ego_a, partner_a, current_loc, new_loc) @ self.feat_weights
                reward_mat[i, a] = rew

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        for i in range(len(states)):
            for j in range(len(actions)):
                assert np.isclose(np.sum(transition_mat[i, :, j]), 1.0) or np.isclose(np.sum(transition_mat[i, :, j]),
                                                                                      0.0)

        return transition_mat, reward_mat, state_to_idx, idx_to_action



if __name__ == "__main__":

    # env = SAREnv(victim_locs, feat_weights=np.array([0.5, 0.4, -1]))
    #  [following preferred policy, following prev partner, in puddle, reached destination, step cost]
    env = JMEnv(np.array([1, 0, -1, 1, -1]))
    transitions, rewards, state_to_idx, idx_to_action = env.enumerate_states()

    # compute optimal policy with value iteration
    values, policy = value_iteration(transitions, rewards)

    print("values", values)
    print("policy", policy)

    # try rolling out policy
    env.reset()
    print(env)
    done = False
    iters = 0

    move_to_text = {0: 'North', 1: 'South', 2: 'East', 3: 'West'}

    while not done and iters < 20:
        iters += 1
        # get index of current state in states
        state_i = state_to_idx[env.current_position]
        # get optimal action from this state using policy
        action_i = int(policy[state_i, 0])
        # get action from action_idx
        # action = idx_to_action[action_i]
        # take action
        next_state, reward, done, partner_action_i = env.step(action_i)

        print(f"Ego: {move_to_text[action_i]}, Partner: {move_to_text[partner_action_i]}")

        grid_to_print = copy.deepcopy(env.grid)
        grid_to_print[env.current_position] = 3
        print(grid_to_print)
        print(reward)
        print()