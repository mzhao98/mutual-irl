import pdb

import numpy as np

# implementation of tabular value iteration
def value_iteration(transitions, rewards, epsilson=0.0001, gamma=0.999, maxiter=100000):
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
    n_states = transitions.shape[0]
    n_actions = transitions.shape[2]

    # initialize value function
    pi = np.zeros((n_states, 1))
    vf = np.zeros((n_states, 1))

    for i in range(maxiter):
        # initalize delta
        delta = 0
        # perform Bellman update
        for s in range(n_states):
            # store old value function
            old_v = vf[s].copy()
            # compute new value function
            # vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s,:,:],0))
            vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
            # vf[s] = np.max(np.sum((rewards[s]) * transitions[s, :, :], 0))
            # compute delta
            delta = np.max((delta, np.abs(old_v - vf[s])[0]))

            # if s == 148:
            #     action = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
            #     print("action = ", action)
                # pdb.set_trace()
        # check for convergence
        if delta < epsilson:
            break
    # compute optimal policy
    for s in range(n_states):
        # pdb.set_trace()
        # pi[s] = np.argmax(np.sum(vf * transitions[s,:,:],0))
        pi[s] = np.argmax(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
        # pi[s] = np.sum(vf * transitions[s,:,:],0)

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