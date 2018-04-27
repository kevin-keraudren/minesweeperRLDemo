Using Reinforcement Learning to solve Minesweeper
=================================================

This code is based on a fork of https://github.com/HaniAlmousli/minesweeperRLDemo .
The minesweeper code originates from https://github.com/cash/minesweeper .

The main changes from @HaniAlmousli are:

  - In the original implementation, a regressor is trained per action to predict the Q value given a state, and
    at every update, a brand new set of regressors are trained.    
    Instead, we propose to use a regressor with a partial_fit method:
    http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning

 - AgentDQN.py has been added to experiment with the DQN approach using Keras.

 Training Results for DQN agent
 ------------------------------

CNN model:

<img src="img/unet.png" height="400">

<img src="img/avg_q_values.png" height="400"> <img src="img/training_loss.png" height="400">

<img src="img/avg_game_duration.png" height="400"> <img src="img/percentage_of_wins.png" height="400">

The advantage of using a CNN is that the model can be trained on a specific game size but can then be applied to games of any size. 

 Testing results
 ---------------

Random Agent:

 <img src="img/testing_random_agent.png" height="400">


DQN Agent:

 <img src="img/testing_agent_dqn.png" height="400">

 Results from @HaniAlmousli
 --------------------------

Random Agent:

 <img src="img/RandomAI.png" height="400">

Ridge Regressor Agent:

 <img src="img/TrainedAgent.png" height="400">

 Conclusion
 ----------

 A Linear Model that takes the whole Minesweeper grid as input works better than a dense CNN that only looks at the game with a sliding window (twice better in this case). But the CNN approach has the advantage of being independent of the size of the game (for instance it was trained on a 16x16 grid and tested on a 10x10 grid).
