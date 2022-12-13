
# importing libraries
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt


BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
COLOR_TO_TEXT = {BLUE: 'blue', GREEN:'green', RED:'red', YELLOW:'yellow'}

# creating subplots
# fig = plt.figure(figsize=(10, 8))
n_games = 10
n_rows = n_games
n_cols = 6

def plot_game(ax, robot_history, human_history):
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 400])

    ax.set_xticks([])
    ax.set_yticks([])

    robot_circle = plt.Circle((50, 300), radius=30, color='#ffcccc')
    ax.add_patch(robot_circle)
    label = ax.annotate("R", xy=(50, 280), fontsize=8, ha="center")

    human_circle = plt.Circle((50, 110), radius=30, color='#86b6e2')
    ax.add_patch(human_circle)
    label = ax.annotate("H", xy=(50, 90), fontsize=8, ha="center")

    robot_x = 120
    robot_y = 280
    for i in range(len(robot_history)):
        rect = matplotlib.patches.Rectangle(((robot_x + 100*i), robot_y), 50, 50, color=COLOR_TO_TEXT[robot_history[i]])
        ax.add_patch(rect)

    human_x = 170
    human_y = 90
    for i in range(len(human_history)):
        rect = matplotlib.patches.Rectangle(((human_x + 100 * i), human_y), 50, 50,
                                            color=COLOR_TO_TEXT[human_history[i]])
        ax.add_patch(rect)

    # plt.show()

if __name__ == '__main__':

    fig, axs = plt.subplots(10, 6, figsize=(20, 18))
    plt.subplots_adjust(wspace=None, hspace=None)

    for round_no in range(n_rows):
        start_state = [1,3,1,1]
        robot_history = [BLUE, GREEN, GREEN]
        human_history = [RED, YELLOW, GREEN]

        plot_game(axs[round_no, 0], robot_history, human_history)

    plt.savefig('test.png')
    plt.close()
    # ax = axs[0, 0]
    # here we are creating sub plots
    # figure, ax = plt.subplots(figsize=(8, 8))

    # plt.xlim([-500, 600])
    # plt.ylim([-500, 600])
    #
    # circle = plt.Circle((-450, 550), radius=30, color='red')
    # ax.add_patch(circle)
    #
    # label = ax.annotate("robot", xy=(-450, 550), fontsize=8, ha="center")
    #
    #
    # curr_x = -300
    # curr_y = -300
    # rect1 = matplotlib.patches.Rectangle((curr_x, curr_y), 100, 100, color='blue')
    # rect2 = matplotlib.patches.Rectangle((-100, -300), 100, 100, color='green')
    # rect3 = matplotlib.patches.Rectangle((100,-300), 100, 100, color='red')
    # rect4 = matplotlib.patches.Rectangle((300,-300), 100, 100, color='yellow')
    #
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)
    # ax.add_patch(rect3)
    # ax.add_patch(rect4)
    #
    # plt.show()

#
# # Loop
# for _ in range(50):
#     # creating new Y values
#     # curr_x += 100
#     curr_y += 100
#
#     # updating data values
#     rect1.set_y(curr_y)
#
#     rect5 = matplotlib.patches.Rectangle((200, np.random.randint(-300,300)), 100, 100, color='red')
#     ax.add_patch(rect5)
#
#     # drawing updated values
#     figure.canvas.draw()
#
#     # This will run the GUI event
#     # loop until all UI events
#     # currently waiting have been processed
#     figure.canvas.flush_events()
#
#     time.sleep(0.1)