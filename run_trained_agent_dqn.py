"""This script tests a trained agent."""

import random
from abc import ABCMeta, abstractmethod
# import pdb
from minesweeper import *
from bokeh.plotting import figure, show, output_file
import numpy as np
from sklearn.linear_model import Ridge
from itertools import compress
import matplotlib.pyplot as plt
from AgentDQN import *
import time

if __name__ == "__main__":

    GAMES_COUNT = 2000
    WIDTH = 10
    HEIGHT = 10
    MINES_COUNT = 12
    viz = None  # GameVisualizer(2)

    dictInfo = {
        'discountFactor': 0.9,
        'memorySize': 10000,
        'epsilonProb': 0.2,
        'savePath': r"res/SGD"
    }

    config = GameConfig(width=WIDTH, height=HEIGHT, num_mines=MINES_COUNT)
    game = Game(config)
    ai = Agent(config, game, dictInfo)
    ai.LoadParams(r"trained_model/unet_weights.h5")
    counter = 0
    lstSteps = []
    counterWins = 0

    time.sleep(4)
    while counter < GAMES_COUNT:
        stepsCount = 0
        game = Game(config)
        ai.ResetAgentState(game)
        if viz:
            viz.start(game)
        while not game.is_game_over():
            coords = ai.next()
            result = game.select(*coords)
            if result is None:
                continue
            if not result.explosion:
                stepsCount += 1
                ai.update(result)
                game.set_flags(ai.get_flags())
                if game.num_exposed_squares == game.num_safe_squares:
                    if viz:
                        viz.update(game)
                    counterWins += 1
            else:
                lstSteps.append(stepsCount)
            if viz:
                viz.update(game)
        if viz:
            viz.finish()
        counter += 1

    fig = plt.figure(figsize=(10, 10))
    plt.hist(lstSteps, normed=0, bins=np.max(lstSteps), edgecolor='black')
    plt.xlabel('Game duration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.title('Testing trained agent: %s/%s wins' % (counterWins, GAMES_COUNT))
    plt.savefig("testing_agent_dqn.png")
    plt.close()
