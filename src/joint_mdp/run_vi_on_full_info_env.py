import pdb
from envs.full_info_env import Joint_MDP_Original
from envs.full_info_env_attempt2 import Joint_MDP
from algs.value_iteration import value_iteration, find_policy
from envs.utils import *


# def basic_test():
#     cleanup_game = CleanupEnv_for_VI()
#     cleanup_game.set_players_preconstructed(BayesianCombinationHumanAgent, BayesianCombinationHumanAgent)
#     cleanup_game.rollout_full_game('bayesian-agent_bayesian-human.png')

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

def test_vi_functionality():
    players_to_reward = [(0.9, -0.9, 0.1, 0.3), (0.9, 0.1, -0.9, 0.2)]
    env = Joint_MDP(index_of_human=1, index_of_robot=2, players_to_reward=players_to_reward)
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

    vi_rew = env.rollout_full_game_vi_policy(policy)
    print(f"FINAL REWARD for optimal team = {vi_rew}")

    # env = Joint_MDP_Original(index_of_human=1, index_of_robot=2, players_to_reward=players_to_reward)
    greedy_rew = env.rollout_full_game_greedy_pair()
    print(f"FINAL REWARD for greedy team = {greedy_rew}")



    # Rollout game
    # env.reset()
    # done = False
    # total_reward = 0
    # while not done:
    #     current_state = env.state
    #     print(f"current_state = {current_state}")
    #     current_state_tup = env.flatten_to_tuple(current_state)
    #     state_idx = state_to_idx[current_state_tup]
    #
    #
    #     joint_action_selected = idx_to_action[int(policy[state_idx, 0])]
    #     # pdb.set_trace()
    #
    #     # pdb.set_trace()
    #     is_valid = env.check_is_valid_joint_action(current_state, joint_action_selected)
    #
    #     print(f"state_idx = {current_state_tup}, color_selected = {joint_action_selected}, is_valid = {is_valid}")
    #     # pdb.set_trace()
    #     next_state, joint_reward, done = env.step(joint_action_selected)
    #     print(f"next_state = {next_state}, reward = {joint_reward}, done = {done}")
    #     total_reward += joint_reward
    #
    # print(f"FINAL REWARD for team = {total_reward}")



if __name__ == "__main__":
    test_vi_functionality()












