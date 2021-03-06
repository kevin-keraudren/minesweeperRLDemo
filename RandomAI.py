import random
from abc import ABCMeta, abstractmethod
import pdb
from minesweeper import *
from bokeh.plotting import figure, show, output_file
import numpy as np
from sklearn.linear_model import Ridge
from itertools import compress
import matplotlib.pyplot as plt


class RandomAI(GameAI):
    def __init__(self):
        self.width = 0
        self.height = 0
        self.exposed_squares = set()

    def init(self, config):
        self.width = config.width
        self.height = config.height
        self.exposed_squares.clear()

    def next(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.exposed_squares:
                break
        return x, y

    def update(self, result):
        for position in result.new_squares:
            self.exposed_squares.add((position.x, position.y))


if __name__ == "__main__":

    GAMES_COUNT = 2000
    WIDTH = 10
    HEIGHT = 10
    MINES_COUNT = 12

    ai = RandomAI()
    config = GameConfig(width=WIDTH, height=HEIGHT, num_mines=MINES_COUNT)
    game = Game(config)
    viz = None  # GameVisualizer(1)

    counter = 0
    lstSteps = []
    counterWins = 0

    while counter < GAMES_COUNT:
        stepsCount = 0
        game = Game(config)
        ai.init(config)
        ai.game = game

        if viz: viz.start(game)

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
                    if viz: viz.update(game)
                    counterWins += 1
            else:
                lstSteps.append(stepsCount)
            if viz: viz.update(game)
        if viz: viz.finish()
        counter += 1

    fig = plt.figure(figsize=(10, 10))
    plt.hist(lstSteps, normed=0, bins=np.max(lstSteps), edgecolor='black')
    plt.xlabel('Game duration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.title('Testing random agent: %s/%s wins' % (counterWins, GAMES_COUNT))
    plt.savefig("testing_random_agent.png")
    plt.close()
