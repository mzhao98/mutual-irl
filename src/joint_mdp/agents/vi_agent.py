import copy
import pdb

import numpy as np
import random
import math
from envs.utils import *


class ValueIterationAgent_Simple:
    def __init__(self, reward_dict, player_idx, mdp):
        self.reward_dict = reward_dict
        self.player_idx = player_idx
        self.set_mdp(mdp)
        self.color_to_text = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}



    def set_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def update_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def set_policy(self, policy, vf):
        self.policy = policy
        self.vf = vf

    def set_parameters(self, state_to_idx, idx_to_action, idx_to_state, action_to_idx):
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx




    # implementation of tabular value iteration
    def value_iteration(self, transitions, rewards, epsilson=0.0001, gamma=0.99, maxiter=10000):
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
                vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
            # check for convergence
            if delta < epsilson:
                break
        # compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * transitions[s, :, :], 0))
            # pi[s] = np.sum(vf * transitions[s, :, :], 0)

        self.vf = vf
        self.policy = pi

        n_states = len(self.state_to_idx)
        for s in range(n_states):
            # print(s)
            print(f"State {self.idx_to_state[s]}: Action {self.color_to_text[self.idx_to_action[int(self.policy[s, 0])]]}")
        # print("values", values)
        # print("policy", policy)

        return vf, pi


    def value(self, policy, n_states, transition_probabilities, reward, discount,
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

    def find_policy(self, n_states, n_actions, transition_probabilities, reward, discount,
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
            v = self.value(policy, n_states, transition_probabilities, reward, discount,
                      threshold=1e-2, max_iters=max_iters)

            policy_improvement_is_stable = True
            for s in range(n_states):
                a_old = np.argmax(policy[s, :])
                tp = transition_probabilities[s, :, :]
                a_new = np.argmax(np.dot(tp, reward + discount * v))
                policy[s, :] = np.dot(tp, reward + discount * v)

                if a_old != a_new:
                    policy_improvement_is_stable = False

            if policy_improvement_is_stable or n_iterations > max_iters:
                policy_iteration_converged = True
                break
        print(policy)
        print("policy shape", policy.shape)
        return policy


    def select_action(self, state):



        possible_actions = self.cleanup_mdp.get_possible_actions(state)

        state_featurized = tuple(self.cleanup_mdp.featurize_state(state))

        print("state_featurized", state_featurized)

        state_idx = self.state_to_idx[state_featurized]

        color_selected = self.idx_to_action[int(self.policy[state_idx, 0])]

        print(f"state_idx = {state_featurized}, color_selected = {color_selected}")
        # pdb.set_trace()


        selected_action = None
        for c_action in possible_actions:
            obj_color = c_action.obj_color

            if obj_color == color_selected:
                selected_action = c_action

        if selected_action is None:
            print("SELECTING RANDOM")
            selected_action = np.random.choice(possible_actions)

        return selected_action

    def select_random_action(self, state):
        possible_actions = self.cleanup_mdp.get_possible_actions()
        action = np.random.choice(possible_actions)
        return action



class FromValueIterationAgent:
    def __init__(self, reward_dict, player_idx, mdp):
        self.reward_dict = reward_dict
        self.player_idx = player_idx
        self.set_mdp(mdp)
        self.color_to_text = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}

    def set_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def update_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def set_policy(self, policy, vf):
        self.policy = policy
        self.vf = vf

    def set_parameters(self, state_to_idx, idx_to_action, idx_to_state, action_to_idx):
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx


    def select_action(self, state):
        possible_actions = self.cleanup_mdp.get_possible_actions(state)

        state_featurized = tuple(self.cleanup_mdp.featurize_state_for_vi(state))

        # if state_featurized not in self.state_to_idx:
        #     print("issue")
        #     print(self.state_to_idx)

        state_idx = self.state_to_idx[state_featurized]

        color_selected = self.idx_to_action[int(self.policy[state_idx, 0])]

        print(f"state_idx = {state_featurized}, color_selected = {color_selected}")
        # pdb.set_trace()

        colors_remaining = set([elem.obj_color for elem in possible_actions])
        find_matching_color = False
        if color_selected in colors_remaining:
            find_matching_color = True

        max_c_rew = -1000
        min_dist = 10000
        selected_action = None

        sample = []
        for c_action in possible_actions:
            obj_color = c_action.obj_color
            obj_loc = c_action.obj_loc
            curr_player_position = state.player_positions[self.player_idx]


            c_rew_self = self.reward_dict[obj_color]

            c_rew = 1 * c_rew_self

            # print("state.sink_locations[obj_color]", state.sink_locations)
            c_dist = 0.5 * l2(curr_player_position, obj_loc) + \
                     0.5 * l2(state.sink_locations[obj_color], obj_loc)

            if find_matching_color:
                if color_selected == obj_color:
                    sample.append((c_action, c_dist))

            else:
                if c_rew > max_c_rew:
                    sample = []
                    sample.append((c_action, c_dist))

                elif c_rew == max_c_rew:
                    sample.append((c_action, c_dist))

        if len(sample) > 0:


            max_sample = max([elem[1] for elem in sample])+1
            sample_weights = [max_sample - elem[1] for elem in sample]

            sum_sample_weights = sum([elem for elem in sample_weights])
            # if sum_sample == 0:
            #     print([elem[1] for elem in sample])
            selected_action = np.random.choice([elem[0] for elem in sample],
                                               p=[elem / sum_sample_weights for elem in sample_weights])


        return selected_action

    def select_random_action(self, state):
        possible_actions = self.cleanup_mdp.get_possible_actions()
        action = np.random.choice(possible_actions)
        return action



class FromValueIterationAgent_BeliefUpdate:
    def __init__(self, reward_dict, player_idx, mdp):
        self.reward_dict = reward_dict
        self.player_idx = player_idx
        self.set_mdp(mdp)
        self.color_to_text = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}
        self.sample_human_policies()
        self.color_to_weight_idx = {RED: 0, GREEN: 1, BLUE: 2, YELLOW: 3}


    def sample_human_policies(self):
        np.random.seed(2)
        num_to_sample = 100
        possible_weights = {}
        self.policy_for_each_belief = {}
        self.reciprocal_policy_for_each_belief = {}
        for i in range(num_to_sample):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            w = random.uniform(-1, 1)
            r = math.sqrt(x * x + y * y + z * z + w * w)
            x /= r
            y /= r
            z /= r
            w /= r
            weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))
            possible_weights[weight_vector] = 1 / num_to_sample
            self.policy_for_each_belief[weight_vector] = None
            self.reciprocal_policy_for_each_belief[weight_vector] = None
        # print("possible_weights", possible_weights)
        self.beliefs = possible_weights

        self.hypothesized_human_beliefs = possible_weights


    def update_beliefs(self):

        partner_idx = 2 if self.player_idx == 1 else 1
        partner_history = self.cleanup_mdp.board_state.player_history[partner_idx]
        if len(partner_history) == 0:
            return
        partner_recent_action = partner_history[-1]
        color_to_object_positions, obj_loc, obj_color, rew = partner_recent_action
        epsilon = 0.0000001
        total_weight = 0
        for weight_vector in self.beliefs:
            prob_theta = self.beliefs[weight_vector]
            weight_idx = self.color_to_weight_idx[obj_color]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / sum(weight_vector_normed_positive) for e in
                                             weight_vector_normed_positive]
            prob_action_given_theta = weight_vector_normed_positive[weight_idx]
            posterior = prob_theta * prob_action_given_theta
            self.beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in self.beliefs:
            self.beliefs[weight_vector] /= total_weight


    def update_human_beliefs(self):

        ego_idx = 2 if self.player_idx == 1 else 1
        ego_history = self.cleanup_mdp.board_state.player_history[ego_idx]
        if len(ego_history) == 0:
            return
        ego_recent_action = ego_history[-1]
        color_to_object_positions, obj_loc, obj_color, rew = ego_recent_action
        epsilon = 0.0000001
        total_weight = 0
        for weight_vector in self.hypothesized_human_beliefs:
            prob_theta = self.hypothesized_human_beliefs[weight_vector]
            weight_idx = self.color_to_weight_idx[obj_color]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / sum(weight_vector_normed_positive) for e in
                                             weight_vector_normed_positive]
            prob_action_given_theta = weight_vector_normed_positive[weight_idx]
            posterior = prob_theta * prob_action_given_theta
            self.hypothesized_human_beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in self.hypothesized_human_beliefs:
            self.hypothesized_human_beliefs[weight_vector] /= total_weight


    def hypothesize_update_beliefs_compute_info_gain(self, hyp_ego_recent_action, color_to_object_positions):


        obj_loc, obj_color = hyp_ego_recent_action.obj_loc, hyp_ego_recent_action.obj_color

        epsilon = 0.0000001
        total_weight = 0

        # Compute original entropy
        prior_entropy = 0
        for weight_vector in self.hypothesized_human_beliefs:
            prob_theta = self.hypothesized_human_beliefs[weight_vector]
            add = prob_theta * np.log(prob_theta)
            prior_entropy += add

        prior_entropy = -1 * prior_entropy

        copy_hypothesized_human_beliefs = copy.deepcopy(self.hypothesized_human_beliefs)
        for weight_vector in copy_hypothesized_human_beliefs:
            prob_theta = copy_hypothesized_human_beliefs[weight_vector]
            weight_idx = self.color_to_weight_idx[obj_color]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / sum(weight_vector_normed_positive) for e in
                                             weight_vector_normed_positive]
            prob_action_given_theta = weight_vector_normed_positive[weight_idx]
            posterior = prob_theta * prob_action_given_theta
            copy_hypothesized_human_beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in copy_hypothesized_human_beliefs:
            copy_hypothesized_human_beliefs[weight_vector] /= total_weight

        # Compute posterior entropy
        posterior_entropy = 0
        for weight_vector in copy_hypothesized_human_beliefs:
            prob_theta = copy_hypothesized_human_beliefs[weight_vector]
            add = prob_theta * np.log(prob_theta)
            posterior_entropy += add

        posterior_entropy = -1 * posterior_entropy
        info_gain = prior_entropy - posterior_entropy
        return info_gain

    def get_max_belief(self):
        max_prob = 0
        best_weights = None
        for weight_vector in self.beliefs:
            if self.beliefs[weight_vector] > max_prob:
                max_prob = self.beliefs[weight_vector]
                best_weights = weight_vector

        return best_weights, max_prob

    def set_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def update_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def set_policy(self, policy, vf):
        self.policy = policy
        self.vf = vf

    def set_parameters(self, state_to_idx, idx_to_action, idx_to_state, action_to_idx):
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx


    def select_action_using_information_gain(self, state):
        self.update_beliefs()
        self.update_human_beliefs()

        best_weights, max_prob = self.get_max_belief()
        team_weights = np.array([1, 1, 1, 1])
        reciprocal_weights = team_weights - np.array(list((best_weights)))
        print("best_weights", best_weights)
        print("reciprocal_weights", reciprocal_weights)
        print("max_prob", max_prob)
        partner_policy_approx = self.policy_for_each_belief[best_weights]
        reciprocal_to_partner_policy_approx = self.reciprocal_policy_for_each_belief[best_weights]

        possible_actions = self.cleanup_mdp.get_possible_actions(state)
        state_featurized = tuple(self.cleanup_mdp.featurize_state_for_vi(state))
        state_idx = self.state_to_idx[state_featurized]

        color_selected_self = self.idx_to_action[int(self.policy[state_idx, 0])]
        color_selected_recip = self.idx_to_action[int(partner_policy_approx[state_idx, 0])]

        max_info_gain = -10000
        best_action = None
        for c_action in possible_actions:
            obj_color = c_action.obj_color
            obj_loc = c_action.obj_loc
            curr_player_position = state.player_positions[self.player_idx]
            info_gain_candidate = self.hypothesize_update_beliefs_compute_info_gain(c_action, self.cleanup_mdp.board_state.color_to_object_positions)

            if info_gain_candidate >= max_info_gain and self.reward_dict[obj_color] > 0:
                max_info_gain = info_gain_candidate
                best_action = c_action

        # if self.reward_dict[best_action.obj_color] < 0:
        #     color_selected = color_selected_self
        #
        #     colors_remaining = set([elem.obj_color for elem in possible_actions])
        #     find_matching_color = False
        #     if color_selected in colors_remaining:
        #         find_matching_color = True
        #
        #     max_c_rew = -1000
        #     min_dist = 10000
        #     selected_action = None
        #
        #     sample = []
        #     for c_action in possible_actions:
        #         obj_color = c_action.obj_color
        #         obj_loc = c_action.obj_loc
        #         curr_player_position = state.player_positions[self.player_idx]
        #
        #         c_rew_self = self.reward_dict[obj_color]
        #
        #         c_rew = 1 * c_rew_self
        #
        #         # print("state.sink_locations[obj_color]", state.sink_locations)
        #         c_dist = 0.5 * l2(curr_player_position, obj_loc) + \
        #                  0.5 * l2(state.sink_locations[obj_color], obj_loc)
        #
        #         if find_matching_color:
        #             if color_selected == obj_color:
        #                 sample.append((c_action, c_dist))
        #
        #         else:
        #             if c_rew > max_c_rew:
        #                 sample = []
        #                 sample.append((c_action, c_dist))
        #
        #             elif c_rew == max_c_rew:
        #                 sample.append((c_action, c_dist))
        #
        #     if len(sample) > 0:
        #         max_sample = max([elem[1] for elem in sample]) + 1
        #         sample_weights = [max_sample - elem[1] for elem in sample]
        #
        #         sum_sample_weights = sum([elem for elem in sample_weights])
        #         # if sum_sample == 0:
        #         #     print([elem[1] for elem in sample])
        #         best_action = np.random.choice([elem[0] for elem in sample],
        #                                            p=[elem / sum_sample_weights for elem in sample_weights])

        print(f"state_idx = {state_featurized}, info gain {max_info_gain}, color_selected = {best_action.obj_color}")


        return best_action

    def select_action(self, state):
        self.update_beliefs()
        best_weights, max_prob = self.get_max_belief()
        team_weights = np.array([1, 1, 1, 1])
        reciprocal_weights = team_weights - np.array(list((best_weights)))
        print("best_weights", best_weights)
        print("reciprocal_weights", reciprocal_weights)
        print("max_prob", max_prob)
        partner_policy_approx = self.policy_for_each_belief[best_weights]
        reciprocal_to_partner_policy_approx = self.reciprocal_policy_for_each_belief[best_weights]


        possible_actions = self.cleanup_mdp.get_possible_actions(state)

        state_featurized = tuple(self.cleanup_mdp.featurize_state_for_vi(state))

        # if state_featurized not in self.state_to_idx:
        #     print("issue")
        #     print(self.state_to_idx)

        state_idx = self.state_to_idx[state_featurized]

        color_selected_self = self.idx_to_action[int(self.policy[state_idx, 0])]
        color_selected_recip = self.idx_to_action[int(partner_policy_approx[state_idx, 0])]
        #
        # color_selected_self = np.random.choice([RED, GREEN, BLUE, YELLOW], p=self.policy[state_idx])
        # color_selected_recip = np.random.choice([RED, GREEN, BLUE, YELLOW], p=partner_policy_approx[state_idx])

        color_selected = color_selected_self
        # if color_selected_self == color_selected_recip:
        #     color_selected = color_selected_self
        # else:
        #     if self.reward_dict[color_selected_recip] < 0:
        #         color_selected = color_selected_self
        #     else:
        #         color_selected = color_selected_recip

        print(f"state_idx = {state_featurized}, color_selected = {color_selected}")
        # pdb.set_trace()

        colors_remaining = set([elem.obj_color for elem in possible_actions])
        find_matching_color = False
        if color_selected in colors_remaining:
            find_matching_color = True

        max_c_rew = -1000
        min_dist = 10000
        selected_action = None

        sample = []
        for c_action in possible_actions:
            obj_color = c_action.obj_color
            obj_loc = c_action.obj_loc
            curr_player_position = state.player_positions[self.player_idx]


            c_rew_self = self.reward_dict[obj_color]

            c_rew = 1 * c_rew_self

            # print("state.sink_locations[obj_color]", state.sink_locations)
            c_dist = 0.5 * l2(curr_player_position, obj_loc) + \
                     0.5 * l2(state.sink_locations[obj_color], obj_loc)

            if find_matching_color:
                if color_selected == obj_color:
                    sample.append((c_action, c_dist))

            else:
                if c_rew > max_c_rew:
                    sample = []
                    sample.append((c_action, c_dist))

                elif c_rew == max_c_rew:
                    sample.append((c_action, c_dist))

        if len(sample) > 0:


            max_sample = max([elem[1] for elem in sample])+1
            sample_weights = [max_sample - elem[1] for elem in sample]

            sum_sample_weights = sum([elem for elem in sample_weights])
            # if sum_sample == 0:
            #     print([elem[1] for elem in sample])
            selected_action = np.random.choice([elem[0] for elem in sample],
                                               p=[elem / sum_sample_weights for elem in sample_weights])


        return selected_action

    def select_random_action(self, state):
        possible_actions = self.cleanup_mdp.get_possible_actions()
        action = np.random.choice(possible_actions)
        return action


class ValueIterationAgent_Simple_w_Belief_Update:
    def __init__(self, reward_dict, player_idx, mdp):
        self.reward_dict = reward_dict
        self.player_idx = player_idx
        self.set_mdp(mdp)
        self.color_to_text = {RED: 'r', GREEN: 'g', BLUE: 'b', YELLOW: 'y'}
        self.sample_human_policies()

    def sample_human_policies(self):
        num_to_sample = 100
        possible_weights = {}
        for i in range(num_to_sample):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            w = random.uniform(-1, 1)
            r = math.sqrt(x * x + y * y + z * z + w * w)
            x /= r
            y /= r
            z /= r
            w /= r
            weight_vector = (np.round(x, 1), np.round(y, 1), np.round(z, 1), np.round(w, 1))
            possible_weights[weight_vector] = 1 / num_to_sample
        # print("possible_weights", possible_weights)
        self.beliefs = possible_weights

    def update_beliefs(self):

        partner_idx = 2 if self.player_idx == 1 else 1
        partner_history = self.cleanup_mdp.board_state.player_history[partner_idx]
        if len(partner_history) == 0:
            return
        partner_recent_action = partner_history[-1]
        color_to_object_positions, obj_color, rew = partner_recent_action
        epsilon = 0.001
        total_weight = 0
        for weight_vector in self.beliefs:
            prob_theta = self.beliefs[weight_vector]
            weight_idx = self.color_to_weight_idx[obj_color]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / sum(weight_vector_normed_positive) for e in
                                             weight_vector_normed_positive]
            prob_action_given_theta = weight_vector_normed_positive[weight_idx]
            posterior = prob_theta * prob_action_given_theta
            self.beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in self.beliefs:
            self.beliefs[weight_vector] /= total_weight


    def update_human_beliefs(self):

        ego_idx = 2 if self.player_idx == 1 else 1
        ego_history = self.cleanup_mdp.board_state.player_history[ego_idx]
        if len(ego_history) == 0:
            return
        ego_recent_action = ego_history[-1]
        color_to_object_positions, obj_loc, obj_color, rew = ego_recent_action
        epsilon = 0.0000001
        total_weight = 0
        for weight_vector in self.hypothesized_human_beliefs:
            prob_theta = self.hypothesized_human_beliefs[weight_vector]
            weight_idx = self.color_to_weight_idx[obj_color]
            weight_vector_normed_positive = [e - min(weight_vector) + epsilon for e in weight_vector]
            weight_vector_normed_positive = [e / sum(weight_vector_normed_positive) for e in
                                             weight_vector_normed_positive]
            prob_action_given_theta = weight_vector_normed_positive[weight_idx]
            posterior = prob_theta * prob_action_given_theta
            self.hypothesized_human_beliefs[weight_vector] = posterior
            total_weight += posterior

        for weight_vector in self.hypothesized_human_beliefs:
            self.hypothesized_human_beliefs[weight_vector] /= total_weight

    def get_max_belief(self):
        max_prob = 0
        best_weights = None
        for weight_vector in self.beliefs:
            if self.beliefs[weight_vector] > max_prob:
                max_prob = self.beliefs[weight_vector]
                best_weights = weight_vector

        return best_weights

    def set_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def update_mdp(self, mdp):
        self.cleanup_mdp = mdp

    def set_policy(self, policy, vf):
        self.policy = policy
        self.vf = vf

    def set_parameters(self, state_to_idx, idx_to_action, idx_to_state, action_to_idx):
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx

    # implementation of tabular value iteration
    def value_iteration(self, transitions, rewards, epsilson=0.0001, gamma=0.99, maxiter=10000):
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
                vf[s] = np.max(np.sum((rewards[s] + gamma * vf) * transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
            # check for convergence
            if delta < epsilson:
                break
        # compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * transitions[s, :, :], 0))

        self.vf = vf
        self.policy = pi

        n_states = len(self.state_to_idx)
        for s in range(n_states):
            # print(s)
            print(
                f"State {self.idx_to_state[s]}: Action {self.color_to_text[self.idx_to_action[int(self.policy[s, 0])]]}")
        # print("values", values)
        # print("policy", policy)

        return vf, pi

    def value(self, policy, n_states, transition_probabilities, reward, discount,
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

    def find_policy(self, n_states, n_actions, transition_probabilities, reward, discount,
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
            v = self.value(policy, n_states, transition_probabilities, reward, discount,
                           threshold=1e-2, max_iters=max_iters)

            policy_improvement_is_stable = True
            for s in range(n_states):
                a_old = np.argmax(policy[s, :])
                tp = transition_probabilities[s, :, :]
                a_new = np.argmax(np.dot(tp, reward + discount * v))
                policy[s, :] = np.dot(tp, reward + discount * v)

                if a_old != a_new:
                    policy_improvement_is_stable = False

            if policy_improvement_is_stable or n_iterations > max_iters:
                policy_iteration_converged = True
                break
        print(policy)
        print("policy shape", policy.shape)
        return policy

    def select_action_using_information_gain(self, state):
        self.update_beliefs()
        self.update_human_beliefs()

        best_weights, max_prob = self.get_max_belief()
        team_weights = np.array([1, 1, 1, 1])
        reciprocal_weights = team_weights - np.array(list((best_weights)))
        print("best_weights", best_weights)
        print("reciprocal_weights", reciprocal_weights)
        print("max_prob", max_prob)
        partner_policy_approx = self.policy_for_each_belief[best_weights]
        reciprocal_to_partner_policy_approx = self.reciprocal_policy_for_each_belief[best_weights]

        possible_actions = self.cleanup_mdp.get_possible_actions(state)
        state_featurized = tuple(self.cleanup_mdp.featurize_state_for_vi(state))
        state_idx = self.state_to_idx[state_featurized]

        color_selected_self = self.idx_to_action[int(self.policy[state_idx, 0])]
        color_selected_recip = self.idx_to_action[int(partner_policy_approx[state_idx, 0])]

        max_info_gain = -10000
        best_action = None
        for c_action in possible_actions:
            obj_color = c_action.obj_color
            obj_loc = c_action.obj_loc
            curr_player_position = state.player_positions[self.player_idx]
            info_gain_candidate = self.hypothesize_update_beliefs_compute_info_gain(c_action)

            if info_gain_candidate > max_info_gain:
                max_info_gain = info_gain_candidate
                best_action = c_action


        print(f"state_idx = {state_featurized}, color_selected = {best_action.obj_color}")


        return best_action

    def select_action(self, state):
        self.update_beliefs()
        best_weights = self.get_max_belief()
        team_weights = np.array([1, 1, 1, 1])
        reciprocal_weights = team_weights - np.array(list((best_weights)))

        possible_actions = self.cleanup_mdp.get_possible_actions(state)

        state_featurized = tuple(self.cleanup_mdp.featurize_state(state))

        print("state_featurized", state_featurized)

        state_idx = self.state_to_idx[state_featurized]

        color_selected = self.idx_to_action[int(self.policy[state_idx, 0])]

        print(f"state_idx = {state_featurized}, color_selected = {color_selected}")
        # pdb.set_trace()

        selected_action = None
        for c_action in possible_actions:
            obj_color = c_action.obj_color

            if obj_color == color_selected:
                selected_action = c_action

        if selected_action is None:
            print("SELECTING RANDOM")
            selected_action = np.random.choice(possible_actions)

        return selected_action

    def select_random_action(self, state):
        possible_actions = self.cleanup_mdp.get_possible_actions()
        action = np.random.choice(possible_actions)
        return action