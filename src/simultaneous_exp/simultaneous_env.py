import pdb

import numpy as np
import operator
import copy
import random
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from multiprocessing import Pool, freeze_support

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]

import pickle
import json
import sys
import os

COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow', None:'none'}

class Toy():
    def __init__(self, color, weight):
        self.color = color
        self.weight = weight




class Simultaneous_Cleanup():
    def __init__(self, first_player, robot, human, starting_objects):
        self.robot = robot
        self.human = human
        self.first_player = first_player

        self.total_reward = {'team': 0, 'robot': 0, 'human': 0}
        self.state_remaining_objects = {}

        self.starting_objects = starting_objects
        self.reset()

    def reset(self):
        self.state_remaining_objects = {}
        for obj_tuple in self.starting_objects:
            if obj_tuple not in self.state_remaining_objects:
                self.state_remaining_objects[obj_tuple] = 1
            else:
                self.state_remaining_objects[obj_tuple] += 1

    def is_done(self):
        if sum(self.state_remaining_objects.values()) == 0:
            return True
        return False

    def step(self, joint_action):

        (robot_action_color, robot_action_weight) = joint_action['robot']
        (human_action_color, human_action_weight) = joint_action['human']

        if 



        return self.state, rew, rew_pair, self.is_done(), robot_action, human_action

    def run_full_game(self, round, no_update=False):
        self.reset()
        iteration_count = 0
        robot_history = []
        human_history = []

        human_only_reward = 0
        robot_only_reward = 0
        while self.is_done() is False:
            _, rew, rew_pair, _, r_action, h_action = self.step(iteration=iteration_count, no_update=no_update)
            # print(f"\nTimestep: {iteration_count}")
            # print(f"Current state: {self.state}")
            # print(f"Robot took action: {COLOR_TO_TEXT[r_action]}")
            # print(f"Robot took action: {COLOR_TO_TEXT[h_action]}")
            # print(f"Achieved reward: {rew_pair}")
            # print(f"Total reward: {self.total_reward}")

            self.rew_history.append(rew_pair)
            self.total_reward += rew
            if h_action is not None:
                human_only_reward += self.human.ind_rew[h_action]

            if r_action is not None:
                robot_only_reward += self.robot.ind_rew[r_action]

            robot_history.append(r_action)
            human_history.append(h_action)

            iteration_count += 1
        return self.total_reward, robot_history, human_history, human_only_reward, robot_only_reward

