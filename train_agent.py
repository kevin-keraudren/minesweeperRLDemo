"""
This scripts trains the RL Agent.
"""

from minesweeper import *
import random
import pdb
from Agent import *

ITERATION_COUNT = int(2e6)
WIDTH = 10
HEIGHT = 10
MINES_COUNT = 12

dictInfo = {
    'discountFactor': 0.9,
    'memorySize': 10000,
    'epsilonProb': 0.2,
    'savePath': r"res/SGD"
}

results = []
config = GameConfig(width=WIDTH, height=HEIGHT, num_mines=MINES_COUNT)
game = Game(config)
ai = Agent(config, game, dictInfo)
stop = False

for x in range(ITERATION_COUNT):
    isExplosion = ai.MoveToNextState()
    if isExplosion:
        game = Game(config)
        ai.ResetAgentState(game)
    else:
        if game.num_exposed_squares == game.num_safe_squares:
            ai.COUNTERWINS += 1

