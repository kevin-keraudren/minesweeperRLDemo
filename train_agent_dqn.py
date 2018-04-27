"""
This scripts trains the RL Agent.
"""

from minesweeper import *
import random
import pdb
from AgentDQN import *

ITERATION_COUNT = int(2e6)
WIDTH = 16
HEIGHT = 16
MINES_COUNT = 12

dictInfo = {
    'discountFactor': 0.9,
    'memorySize': 10000,
    'epsilonProb': 0.2,
    'savePath': r"trained_model"
}

results = []
config = GameConfig(width=WIDTH, height=HEIGHT, num_mines=MINES_COUNT)
game = Game(config)
ai = Agent(config, game, dictInfo)
stop = False

for x in range(ITERATION_COUNT):
    ai.MoveToNextState()
    if game.is_game_over():
        game = Game(config)
        ai.ResetAgentState(game)
