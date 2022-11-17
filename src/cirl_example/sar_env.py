import numpy as np
import networkx as nx
from collections import defaultdict

from environment import NavigationEnv
from value_iteration import value_iteration

class PartnerPolicy():
    def __init__(self, victim_locs, xbound, ybound):
        self.victim_locs = victim_locs

        self.xbound = xbound
        self.ybound = ybound

        # construct graph of gridworld
        self.graph = nx.Graph()
        for i in range(self.xbound[0], self.xbound[1]):
            for j in range(self.ybound[0], self.ybound[1]):
                self.graph.add_node((i, j))
        # add edges between adjacent squares
        for i in range(self.xbound[0], self.xbound[1]):
            for j in range(self.ybound[0], self.ybound[1]):
                if i < self.xbound[1] - 1:
                    self.graph.add_edge((i, j), (i + 1, j))
                if j < self.ybound[1] - 1:
                    self.graph.add_edge((i, j), (i, j + 1))

    def path_to_action(self, path):
        return np.array(path[1]) - np.array(path[0])

    def policy(self, position, victim_locs):
        if len(victim_locs) == 0:
            return np.array([0, 0])
        # find the closest victim
        dists = np.linalg.norm(victim_locs - position, axis=1)
        closest_victim = victim_locs[np.argmin(dists)]
        
        # find the path to the closest victim
        closest_key = tuple(closest_victim.astype(int))
        # NOTE: is it redundant to recompute this every time?
        path = nx.shortest_path(self.graph, tuple(position.astype(int)), closest_key)
        
        # return the first action along this path
        return self.path_to_action(path)

class SAREnv(NavigationEnv):
    """
    Search and Rescue MDP environment. 2 player task where they must visit a series of locations on the grid.
    """
    def __init__(self, victim_locs : np.array, feat_weights : np.array):
        self.victim_locs = victim_locs
        self.feat_weights = feat_weights

        # define bounds of grid
        self.xbound = np.array([0, 5])
        self.ybound = np.array([0, 5])

        # store vector of which vicitms have been visited
        self.visited = np.zeros(len(self.victim_locs))
        self.loc_to_idx = {tuple(victim_loc.astype(int)): i for i, victim_loc in enumerate(self.victim_locs)}
        self.idx_to_loc = {i: tuple(victim_loc.astype(int)) for i, victim_loc in enumerate(self.victim_locs)}
        # store a map of whether each position on the grid has a victim
        self.is_victim = {}
        victim_tup = tuple(map(tuple, self.victim_locs))
        for i in range(self.xbound[0], self.xbound[1]):
            for j in range(self.ybound[0], self.ybound[1]):
                self.is_victim[(i, j)] = (i, j) in victim_tup

        # construct partner policy (makes more sense here than to be passed in)
        self.partner = PartnerPolicy(self.victim_locs, self.xbound, self.ybound)

        # define indices of state vector
        self.ego_idx = range(0, 2)
        self.partner_idx = range(2, 4)
        self.victim_idxs = range(4, len(self.victim_locs) + 4)
        self.n = self.victim_idxs[-1] + 1

        # check that all victim locations are within grid bounds
        for victim_loc in self.victim_locs:
            assert self.xbound[0] <= victim_loc[0] <= self.xbound[1]
            assert self.ybound[0] <= victim_loc[1] <= self.ybound[1]

        self.state = self.reset()

    def step(self, action):
        # action contains ego agent's action
        ego_action = action
        # each action is an array [x movement, y movement]
        ego_position, partner_position, _ = self.decode_state(self.state)
        remaining_victims = self.get_remaining_victims()
        partner_action = self.partner.policy(partner_position, remaining_victims)
        ego_position += ego_action
        partner_position += partner_action

        # if ego agent is outside bounds, push it back inside
        ego_position[0] = max(self.xbound[0], min(ego_position[0], self.xbound[1]-1))
        ego_position[1] = max(self.ybound[0], min(ego_position[1], self.ybound[1]-1))
        # if partner agent is outside bounds, push it back inside
        partner_position[0] = max(self.xbound[0], min(partner_position[0], self.xbound[1]-1))
        partner_position[1] = max(self.ybound[0], min(partner_position[1], self.ybound[1]-1))

        # convert to integer coordinates
        ego_key = tuple(ego_position.astype(int))
        partner_key = tuple(partner_position.astype(int))

        # check if each agent has reached a victim
        if self.is_victim[ego_key]:
            self.visited[self.loc_to_idx[ego_key]] = 1
            self.is_victim[ego_key] = False
        if self.is_victim[partner_key]:
            self.visited[self.loc_to_idx[partner_key]] = 1
            self.is_victim[partner_key] = False

        # update state
        self.state = self.encode_state(ego_position, partner_position, self.visited)

        # check if all victims have been visited
        if np.all(self.visited):
            reward = 100
            done = True
        else:
            reward = self.featurize_state(self.state) @ self.feat_weights
            done = False

        return self.state, reward, done, partner_action

    def get_remaining_victims(self):
        return self.victim_locs[np.where(self.visited == 0)[0]]

    def featurize_state(self, state):
        # featurize based on desired reward function
        # current featurization: [distance to closest victim, distance to partner, step cost]

        ego_position, partner_position, _ = self.decode_state(state)

        # find closest victim
        victims = self.get_remaining_victims()
        if len(victims) == 0:
            closest_dist = 0
        else:
            dists = np.linalg.norm(victims - ego_position, axis=1)
            closest_dist = np.argmin(dists)

        # distance to partner
        partner_dist = np.linalg.norm(partner_position - ego_position)

        return np.array([closest_dist, partner_dist, 1])

    def encode_state(self, ego_loc, partner_loc, visited):
        state_vec = np.zeros(self.n)
        state_vec[self.ego_idx] = ego_loc
        state_vec[self.partner_idx] = partner_loc
        state_vec[self.victim_idxs] = visited

        return state_vec

    def decode_state(self, state):
        ego_position = state[self.ego_idx]
        partner_position = state[self.partner_idx]
        visited = state[self.victim_idxs]
        return ego_position, partner_position, visited

    def reset(self):
        ego_start = np.array([0, 2])
        partner_start = np.array([0, 3])

        start_state = self.encode_state(ego_start, partner_start, np.zeros(len(self.victim_locs)))
        self.set_to_state(start_state)

        return start_state

    def __repr__(self):
        # construct a string representation of the state, representing the victims as X, the ego agent as E, and the partner agent as P
        state_str = ''
        victim_tup = tuple(map(tuple, self.get_remaining_victims()))
        for j in range(self.ybound[1]-1, self.ybound[0]-1, -1):
            for i in range(self.xbound[0], self.xbound[1]):
                if (i, j) in victim_tup:
                    state_str += 'X'
                elif (i, j) == tuple(self.decode_state(self.state)[0].astype(int)):
                    state_str += 'E'
                elif (i, j) == tuple(self.decode_state(self.state)[1].astype(int)):
                    state_str += 'P'
                else:
                    state_str += '.'
            state_str += '\n'
        return state_str

    def set_to_state(self, state):
        """
        Set the environment to a given state.
        """
        self.state = state
        _, _, visited = self.decode_state(state)
        self.visited = visited
        # create is_victim dictionary as defaultdict with default value of False
        self.is_victim = defaultdict(lambda: False)
        # loop through victims and set is_victim to True for each victim
        for victim_loc in self.victim_locs:
            # if victim has not been visited, set is_victim to True
            if not self.visited[self.loc_to_idx[tuple(victim_loc)]]:
                self.is_victim[tuple(victim_loc)] = True

if __name__ == "__main__":
    victim_locs = np.array([[1,0],
                            [2, 0],
                            [1, 3],
                            [1, 4],
                            [2, 3],
                            [4, 2]])
    # featurization: [distance to closest victim, distance to partner, step cost]
    env = SAREnv(victim_locs, feat_weights=np.array([5, -1.0, 0.0])) # profile 1 weights
    # env = SAREnv(victim_locs, feat_weights=np.array([0.0, 0.0, -10])) # profile 2 weights
    transitions, rewards, state_to_idx, idx_to_action = env.enumerate_states()

    # compute optimal policy with value iteration
    values, policy = value_iteration(transitions, rewards)

    # try rolling out policy
    env.reset()
    print(env)
    done = False
    while not done:
        # get index of current state in states
        state_i = state_to_idx[tuple(env.state.astype(int))]
        # get optimal action from this state using policy
        action_i = int(policy[state_i,0])
        # get action from action_idx
        action = idx_to_action[action_i]
        # take action
        next_state, reward, done = env.step(action)
        print(env)
        # print(reward)
        print()