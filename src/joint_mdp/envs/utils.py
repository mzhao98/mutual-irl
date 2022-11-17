from __future__ import division

import time
import math
import random
import numpy as np
import copy

PLAYER_1 = 1
PLAYER_2 = 2
RED = 3
GREEN = 4
BLUE = 5
YELLOW = 6
RED_SINK = 7
GREEN_SINK = 8
BLUE_SINK = 9
YELLOW_SINK = 10

NO_ACTION = -1

NUM_COLORS = 4
NUM_OBJECTS_PER_COLOR = 3
NUM_PLAYERS = 2

scale = 1
# NUM_RED = 1 * scale
# NUM_BLUE = 1 * scale
# NUM_YELLOW = 3 * scale
# NUM_GREEN = 3 * scale

# NUM_RED = np.random.randint(1, 10) * scale
# NUM_BLUE = np.random.randint(1, 10) * scale
# NUM_YELLOW = np.random.randint(1, 10) * scale
# NUM_GREEN = np.random.randint(1, 10) * scale

# COLOR_TO_NUM_OBJ = {RED: NUM_RED, GREEN: NUM_GREEN, BLUE: NUM_BLUE, YELLOW: NUM_YELLOW}

def l2(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)