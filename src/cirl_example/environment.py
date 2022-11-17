from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

class BaseEnv(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def enumerate_states(self):
        pass

class NavigationEnv(BaseEnv):
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def set_to_state(self, state):
        pass

    def enumerate_states(self):
        self.reset()
        actions = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1]])
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()
        stack = [(self.state.copy(), False)]
        while stack:
            state, done = stack.pop()
            # convert state to tuple
            state_tup = tuple(state.astype(int))
            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            if done:
                for action in actions:
                    G.add_edge(state_tup, state_tup, weight=200, action=tuple(action.astype(int)))
                continue

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(actions):
                # set the environment to the current state
                self.set_to_state(state)
                # take the action
                new_state, rew, done, _ = self.step(action)
                # if next state has not been visited, add it to the stack
                new_state_tup = tuple(new_state.astype(int))
                if new_state_tup not in visited_states:
                    stack.append((new_state.copy(), done))
                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=rew, action=tuple(action.astype(int)))

        states = list(G.nodes)
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}
        action_idx = {tuple(action.astype(int)): i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])
        for i in range(len(states)):
            # get all outgoing edges from current state
            edges = G.out_edges(states[i], data=True)
            for edge in edges:
                # get index of action in action_idx
                action_idx_i = action_idx[edge[2]['action']]
                # get index of next state in node list
                next_state_i = states.index(edge[1])
                # add edge to transition matrix
                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = edge[2]['weight']

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        for i in range(len(states)):
            for j in range(len(actions)):
                assert np.isclose(np.sum(transition_mat[i,:,j]), 1.0) or np.isclose(np.sum(transition_mat[i,:,j]), 0.0)
                        
        return transition_mat, reward_mat, state_to_idx, idx_to_action
