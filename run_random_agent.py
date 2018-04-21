"""Simple demo of a Random Agent playing Minesweeper."""
import minesweeper as ms
import random
import pdb
from RandomAI import RandomAI

"""
Beginner       9   9   10
Intermediate   16  16  40
Expert         16  30  99
"""

if __name__ == "__main__":
    config = ms.GameConfig(width=10, height=10, num_mines=12)
    ai = RandomAI()
    viz = ms.GameVisualizer(1)
    results = ms.run_games(config, 1, ai, viz)
    if results[0].success:
        print('Success!')
    else:
        print('Boom!')
    print('Game lasted {0} moves'.format(results[0].num_moves))



