import pdb

import numpy as np
import copy
# implementation of tabular value iteration
def value_iteration(transitions, rewards, idx_to_state, state_to_idx, epsilson=0.0001, gamma=1.0, maxiter=100000):
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
    players_to_reward = [(0.9, -0.9, 0.1, 0.3), (0.9, 0.1, -0.9, 0.2)]
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

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2, max_iters=1000):
    """
    Find the value function associated with a policy.
    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    for iteration in range(max_iters):
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = np.argmax(policy[s, :])
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

        if diff <= threshold:
            break

    return v


def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True, max_iters=10):
    """
    Find the optimal policy.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    #### Initialize random values and random policy
    v = np.random.normal(loc=0.0, scale=1.0, size=n_states)
    policy = np.random.choice(range(n_actions), size=(n_states, n_actions))


    policy_iteration_converged = False
    n_iterations = 0

    while not policy_iteration_converged:
        n_iterations += 1
        # policy evaluation
        v = value(policy, n_states, transition_probabilities, reward, discount,
              threshold=1e-2, max_iters=max_iters)

        policy_improvement_is_stable = True
        for s in range(n_states):
            a_old = np.argmax(policy[s, :])
            tp = transition_probabilities[s,:,:]
            a_new = np.argmax(np.dot(tp, reward + discount*v))
            policy[s, :] = np.dot(tp, reward + discount*v)

            if a_old != a_new:
                policy_improvement_is_stable = False

        if policy_improvement_is_stable or n_iterations > max_iters:
            policy_iteration_converged = True
            break

    # print("policy shape", policy.shape)
    return policy


# def value_iteration(self):
#     self.epsilson = 0.0001
#     self.gamma = 0.999999999999
#     self.maxiter = 100
#
#
#     """
#     Parameters
#     ----------
#         transitions : array_like
#             Transition probability matrix. Of size (# states, # states, # actions).
#         rewards : array_like
#             Reward matrix. Of size (# states, # actions).
#         epsilson : float, optional
#             The convergence threshold. The default is 0.0001.
#         gamma : float, optional
#             The discount factor. The default is 0.99.
#         maxiter : int, optional
#             The maximum number of iterations. The default is 10000.
#     Returns
#     -------
#         value_function : array_like
#             The value function. Of size (# states, 1).
#         pi : array_like
#             The optimal policy. Of size (# states, 1).
#     """
#     n_states = self.transitions.shape[0]
#     n_actions = self.transitions.shape[2]
#
#     # initialize value function
#     pi = np.zeros((n_states, 1))
#     vf = np.zeros((n_states, 1))
#     Q = np.zeros((n_states, n_actions))
#
#     for i in range(self.maxiter):
#         # initalize delta
#         delta = 0
#         # perform Bellman update
#         for s in range(n_states):
#             # store old value function
#             old_v = vf[s].copy()
#
#             # compute new Q values
#
#             if self.vi_type == 'mmvi':
#                 # Add partner rew
#                 initial_state = copy.deepcopy(list(self.idx_to_state[s]))
#                 for action_idx in range(n_actions):
#                     r_sa = self.rewards[s][action_idx]
#                     r_s1aH = 0
#                     robot_action = self.idx_to_action[action_idx]
#
#                     # have the robot act
#                     current_state = copy.deepcopy(list(self.idx_to_state[s]))
#
#                     # update state
#                     if current_state[robot_action] > 0:
#                         current_state[robot_action] -= 1
#
#                         # update human model's model of robot based on last robot action
#                         copy_human_model = copy.deepcopy(self.human_model)
#
#                         copy_human_model.update_with_partner_action(initial_state, robot_action, False)
#                         human_action = copy_human_model.act(current_state, [], [])
#
#                         # update state and human's model of robot
#                         if human_action is not None and current_state[human_action] > 0:
#                             r_s1aH += (self.human_rew[human_action])
#                             current_state[human_action] -= 1
#
#                     s11 = self.state_to_idx[tuple(current_state)]
#                     joint_reward = r_sa + (r_s1aH * self.h_scalar)
#                     V_s11 = vf[s11]
#                     Q[s,action_idx] = joint_reward + (self.gamma * V_s11)
#
#             elif self.vi_type == 'mmvi_nh':
#                 # Add partner rew
#                 initial_state = copy.deepcopy(list(self.idx_to_state[s]))
#                 for action_idx in range(n_actions):
#                     r_sa = self.rewards[s][action_idx]
#                     r_s1aH = 0
#                     robot_action = self.idx_to_action[action_idx]
#
#                     # have the robot act
#                     current_state = copy.deepcopy(list(self.idx_to_state[s]))
#
#                     # update state
#                     if current_state[robot_action] > 0:
#                         current_state[robot_action] -= 1
#
#                         # update human model's model of robot based on last robot action
#                         copy_human_model = copy.deepcopy(self.human_model)
#                         # copy_human_model.update_with_partner_action(initial_state, robot_action)
#                         human_action = copy_human_model.act(current_state, [], [])
#
#                         # update state and human's model of robot
#                         if human_action is not None and current_state[human_action] > 0:
#                             r_s1aH += (self.human_rew[human_action])
#                             current_state[human_action] -= 1
#
#                     s11 = self.state_to_idx[tuple(current_state)]
#                     joint_reward = r_sa + (r_s1aH * self.h_scalar)
#                     V_s11 = vf[s11]
#                     Q[s,action_idx] = joint_reward + (self.gamma * V_s11)
#
#             else:
#                 # Add partner rew
#                 initial_state = copy.deepcopy(list(self.idx_to_state[s]))
#                 for action_idx in range(n_actions):
#                     r_sa = self.rewards[s][action_idx]
#                     r_s1aH = 0
#                     robot_action = self.idx_to_action[action_idx]
#
#                     # have the robot act
#                     current_state = copy.deepcopy(list(self.idx_to_state[s]))
#
#                     # update state
#                     if current_state[robot_action] > 0:
#                         current_state[robot_action] -= 1
#
#                     s11 = self.state_to_idx[tuple(current_state)]
#                     joint_reward = r_sa + r_s1aH
#                     V_s11 = vf[s11]
#                     Q[s, action_idx] = joint_reward + (self.gamma * V_s11)
#
#
#             # print("Q[s,:]", Q[s,:])
#
#             # print("shape Q[s,:]", np.shape(Q[s,:]))
#             vf[s] = np.max(Q[s,:], 0)
#             # print("vf[s]", vf[s])
#             # vf[s] = np.max(np.sum((rewards[s]) * transitions[s, :, :], 0))
#             # compute delta
#             delta = np.max((delta, np.abs(old_v - vf[s])[0]))
#
#             # if s == 148:
#             #     action = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
#             #     print("action = ", action)
#             # pdb.set_trace()
#         # check for convergence
#         # print(f'delta = {delta}, iteration {i}')
#         if delta < self.epsilson:
#             # print("DONE at iteration ", i)
#             break
#     # compute optimal policy
#     policy = {}
#
#     self.vf = vf
#     self.pi = pi
#     self.policy = policy
#     # print("self.pi", self.pi)
#     return vf, pi