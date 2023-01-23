import numpy as np
from gym import spaces


# self._moves = ["NORTH", "SOUTH", "EAST", "WEST"]
#
# self._moves_as_coords = [ (-1, 0), (1, 0), (0, 1), (0, -1)]

class Action:
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1,0)
        self.EAST = (0,1)
        self.WEST = (0,-1)
        self.TRIAGE = 'TRIAGE'
        self.STAY = (0,0)
        # self.PLANT = 'CONSUME'


class Board():
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)
        self.TRIAGE = 'TRIAGE'
        self.STAY = (0,0)

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        # apple -- 3

        self.action_idx_to_coord = {0: self.NORTH, 1: self.SOUTH, 2:self.EAST, 3:self.WEST, 4:self.TRIAGE, 5:self.STAY}

        self.random_seed = 0
        self.grid_w, self.grid_l =  4, 4
        self.grid_shape = (self.grid_w, self.grid_l)
        self.empty_grid = np.zeros((self.grid_w, self.grid_l))
        self.index_to_position = dict(enumerate([(i,j) for i in range(self.empty_grid.shape[0]) for j in range(self.empty_grid.shape[1])]))

        self.start_num_regular_victims = 2
        self.num_victims_regular_remaining = 2

        self.start_num_critical_victims = 1
        self.num_victims_critical_remaining = 1


        self.regular_victim_locations = []
        self.critical_victim_locations = []

        self.num_players = 2
        self.player_positions = {i: None for i in range(self.num_players)}
        self.player_orientations = {i: self.NORTH for i in range(self.num_players)}

        self.current_timestep = 0
        self.max_timesteps = 10
        self.player_rewards = {i:0 for i in range(self.num_players)}

        self.random_reset(self.random_seed)


        self.previous_actions = [None, None]

    def random_reset(self, seed=0):
        np.random.seed(seed)

        self.grid = np.zeros((self.grid_w, self.grid_l))
        self.regular_victim_locations = [(2,0), (3,3)]
        self.critical_victim_locations = [(2,2)]
        self.player_positions[0] = (3,0)
        self.player_positions[1] = (0,3)

        self.grid[self.player_positions[0]] = 1
        self.grid[self.player_positions[1]] = 2
        for victim_loc in self.regular_victim_locations:
            self.grid[victim_loc] = 3

        for victim_loc in self.critical_victim_locations:
            self.grid[victim_loc] = 4

        self.player_orientations = {i: self.NORTH for i in range(self.num_players)} # reinit orientations to north
        self.current_timestep = 0
        self.player_rewards = {i: 0 for i in range(self.num_players)}

        self.num_total_victims_remaining = len(self.regular_victim_locations) + len(self.critical_victim_locations) 
        self.previous_actions = [None, None]
        self.victims_saved = [0,0]

    def set_previous_actions(self, previous_actions):
        self.previous_actions[0] = previous_actions['player_1']
        self.previous_actions[1] = previous_actions['player_2']

    def can_step(self, player_index, action):
        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        can_move = False
        if check_next_move[0] >= 0 and check_next_move[0] < self.grid_w:
            if check_next_move[1] >= 0 and check_next_move[1] < self.grid_l:
                if self.grid[check_next_move] == 0:
                    can_move = True

        return can_move

    def is_valid_regular_triage(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_victim_loc = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_save = False
        if check_victim_loc[0] >= 0 and check_victim_loc[0] < self.grid_w:
            if check_victim_loc[1] >= 0 and check_victim_loc[1] < self.grid_l:
                if self.grid[check_victim_loc] == 3:
                    can_save = True

        return can_save

    def perform_direction_move(self, player_index, action):
        # print("action", action)
        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        successful = False
        can_step = False
        if check_next_move[0] >= 0 and check_next_move[0] < self.grid_w:
            if check_next_move[1] >= 0 and check_next_move[1] < self.grid_l:
                if self.grid[check_next_move] == 0:
                    can_step = True

        if can_step:
            self.grid[current_player_pos] = 0
            self.grid[check_next_move] = player_index + 1
            self.player_positions[player_index] = check_next_move
            self.player_orientations[player_index] = action
            successful = True
        else:
            if self.player_orientations[player_index] != action:
                successful = True
            self.player_orientations[player_index] = action

        return 0, successful

    def perform_regular_triage(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_victim_loc = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_consume = False
        successful = False
        reward = 0
        # if self.num_apples_remaining < 7:
        #     reward = -1
        if check_victim_loc[0] >= 0 and check_victim_loc[0] < self.grid_w:
            if check_victim_loc[1] >= 0 and check_victim_loc[1] < self.grid_l:
                if self.grid[check_victim_loc] == 3:
                    self.grid[check_victim_loc] = 0
                    assert self.grid[check_victim_loc] != 3 and self.grid[check_victim_loc] == 0
                    self.regular_victim_locations.remove(check_victim_loc)
                    self.num_total_victims_remaining -= 1
                    reward = 1
                    # if self.num_apples_remaining < 6 and player_index == 0:
                    #     reward = -1
                    self.victims_saved[player_index] += 1
                    successful = True
        return reward, successful

    def perform_critical_triage(self):
        
        successful = False
        current_player_pos = self.player_positions[0]
        current_player_or = self.player_orientations[0]
        check_victim_loc_p0 = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        current_player_pos = self.player_positions[1]
        current_player_or = self.player_orientations[1]
        check_victim_loc_p1 = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        if check_victim_loc_p0 != check_victim_loc_p1:
            return (0,0), (successful, successful)
        else:
            check_victim_loc = check_victim_loc_p0
            reward = 0
            # if self.num_apples_remaining < 7:
            #     reward = -1
            if check_victim_loc[0] >= 0 and check_victim_loc[0] < self.grid_w:
                if check_victim_loc[1] >= 0 and check_victim_loc[1] < self.grid_l:
                    if self.grid[check_victim_loc] == 4:
                        self.grid[check_victim_loc] = 0
                        assert self.grid[check_victim_loc] != 4 and self.grid[check_victim_loc] == 0
                        self.critical_victim_locations.remove(check_victim_loc)
                        self.num_total_victims_remaining -= 1
                        reward = 10
                        # if self.num_apples_remaining < 6 and player_index == 0:
                        #     reward = -1
                        self.victims_saved[0] += 1
                        self.victims_saved[1] += 1
                        successful = True

        return (reward, reward), (successful, successful)

    def is_done(self):
        if self.current_timestep > self.max_timesteps:
            return True
        if self.num_total_victims_remaining == 0:
            return True
        else:
            return False

    def step_joint_action(self, joint_action):
        p1_action, p2_action = joint_action
        p1_action = self.action_idx_to_coord[p1_action]
        p2_action = self.action_idx_to_coord[p2_action]

        p1_triaged, p2_triaged = False, False
        p1_success, p2_success = False, False

        p1_rew = 0
        p2_rew = 0

        if p1_action == self.TRIAGE:
            p1_rew, p1_success = self.perform_regular_triage(player_index=0)
        else:
            p1_rew, p1_success = self.perform_direction_move(player_index=0, action=p1_action)

        if p2_action == self.TRIAGE:
            p2_rew, p2_success = self.perform_regular_triage(player_index=1)
        else:
            p2_rew, p2_success = self.perform_direction_move(player_index=1, action=p2_action)

        if (p1_action, p2_action) == (self.TRIAGE, self.TRIAGE):
            if p1_success is not True and p2_success is not True:
                (p1_rew, p2_rew), (p1_success, p2_success) = self.perform_critical_triage()

        return (p1_rew, p2_rew), (p1_success, p2_success)




    def get_final_team_reward(self):
        return self.player_rewards[0] + self.player_rewards[1]



    def observation_as_vector(self, player_idx):
        # obs = spaces.Box(low=0, high=1, shape=(27, 1), dtype=np.int8)
        ego_idx = player_idx
        partner_idx = 1-player_idx
        num_actions = len(self.action_idx_to_coord)


        observation_list = []

        # Get last partner actions
        partner_last_action = [0]*num_actions
        if self.previous_actions[partner_idx] is not None:
            partner_last_action[self.previous_actions[partner_idx]] = 1
        observation_list.extend(partner_last_action)

        # print("observation_list", observation_list)

        # Get last ego action
        ego_last_action = [0] * num_actions
        if self.previous_actions[partner_idx] is not None:
            ego_last_action[self.previous_actions[ego_idx]] = 1
        observation_list.extend(ego_last_action)

        # print("observation_list", observation_list)

        # get ego position
        ego_pos = self.player_positions[ego_idx]
        observation_list.append(ego_pos[0])
        observation_list.append(ego_pos[1])

        # print("observation_list", observation_list)

        # get partner position
        partner_pos = self.player_positions[partner_idx]
        observation_list.append(partner_pos[0])
        observation_list.append(partner_pos[1])

        # print("observation_list", observation_list)

        # number of apples remaining
        victim_locations = [0] * (2*(self.start_num_critical_victims + self.start_num_regular_victims))

        for i in range(self.start_num_regular_victims):
            if i < len(self.regular_victim_locations):
                victim_locations.append(self.regular_victim_locations[i][0])
                victim_locations.append(self.regular_victim_locations[i][1])

            else:
                victim_locations.append(0)
                victim_locations.append(0)

        for i in range(self.start_num_critical_victims):
            if i < len(self.critical_victim_locations):
                victim_locations.append(self.critical_victim_locations[i][0])
                victim_locations.append(self.critical_victim_locations[i][1])

            else:
                victim_locations.append(0)
                victim_locations.append(0)
        observation_list.extend(victim_locations)


        
        observation = np.array(observation_list).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)
        # print("observation_list", observation_list)
        # print("observation", observation.shape)

        return observation



    def observation_as_stacked_array(self, player_idx):
        # obs = spaces.Box(low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]))

        grid_with_ego_position = np.zeros(self.grid_shape)
        grid_with_ego_position[self.player_positions[player_idx]] = 1

        grid_with_partner_position = np.zeros(self.grid_shape)
        grid_with_partner_position[self.player_positions[1-player_idx]] = 1

        grid_with_apples = np.zeros(self.grid_shape)
        for pos in self.apple_locations:
            grid_with_apples[pos] = 1


        observation = np.stack([grid_with_ego_position, grid_with_partner_position, grid_with_apples], axis=0)
        # print("observation", observation.shape)
        return observation


    def respawn_apples(self):
        spawn_prob = 0
        if self.num_apples_remaining < 5:
            spawn_prob = 0

        elif self.num_apples_remaining == 5:
            spawn_prob = 0.5

        elif self.num_apples_remaining > 5:
            spawn_prob = 1

        if spawn_prob > 0 and self.current_timestep % 5 == 0:
            loc_index = np.random.choice(len(self.index_to_position), replace=False)
            new_apple_loc = self.index_to_position[loc_index]

            rand = np.random.uniform(0, 1)
            if rand < spawn_prob:
                if self.grid[new_apple_loc] == 0:
                    self.grid[new_apple_loc] = 3
                    self.apple_locations.append(new_apple_loc)
                    self.number_of_apples_respawned += 1
                    self.num_apples_remaining += 1
                    # print(f"RESPAWN: number_of_apples_respawned = {self.number_of_apples_respawned}, num consumed = {self.apples_consumed}")


        return












