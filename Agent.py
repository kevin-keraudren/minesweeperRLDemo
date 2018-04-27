from minesweeper import GameAI
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, SGDRegressor
import os
import joblib


def one_hot_encoding(game_state):
    # Convert the minesweeper into one hot big state vector (every cell has 10 one hot rep)
    cellsStateCount = 10
    cells = np.reshape(np.asarray(game_state), -1)
    cells[np.isnan(cells.astype(float))] = 9
    return np.reshape(np.eye(cellsStateCount)[np.asarray(cells, 'int')], [-1])


class Agent(GameAI):
    """
    Description from https://hanialmousli.wordpress.com/2017/10/11/minesweeper-using-reinforcement-learning/
    Our state is represented as a one hot encoding vector which is 10 x 10 x 10.
    10 x 10 is the width and  height and every cell is represented by a binary vector of 10.
    For example if the cell is not clicked then the first value is 1 and the others are zeros.
    A linear model is trained to predict the Q-value of every possible action (place where we can click),
    given the current state and Epsilon greedy is used to choose the place to click.

    In the original implementation, we train a regressor per action that predicts the Q value given a state,
    at every update, we train a brand new set of regressors.

    Instead, we propose to use a regressor with a partial_fit method:
    http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
    """

    def __init__(self, config, game, inputDic):

        self.config = config
        self.game = game

        self.COUNTERMEM = 0
        self.COUNTERWINS = 0

        self.width = self.config.width
        self.height = self.config.height
        self.exposed_squares = set()
        self.exposed_squares.clear()

        self.rng = np.random.RandomState(123)
        self.actionsId = np.arange(self.width * self.height)
        self.n_actions = self.width * self.height

        # Convert the minesweeper into one hot big state vector (every cell has 10 one hot rep)
        self.currentState = one_hot_encoding(self.game.get_state())

        # Extract Info from inputDic
        self.inputDic = inputDic
        self.discountFactor = inputDic['discountFactor']
        self.epsilonProb = inputDic['epsilonProb']
        self.memorySize = inputDic['memorySize']
        self.savePath = inputDic['savePath']

        # Add the data for memory
        self.recorded_states = np.zeros([self.memorySize, self.currentState.shape[0]])
        self.recorded_actions = np.zeros([self.memorySize])
        self.recorded_targets = np.zeros([self.memorySize])

        self.regressors = [SGDRegressor(warm_start=True) for _ in range(self.n_actions)]

        self.n_wins = []

    def MoveToNextState(self):
        """
        Move to the next state from the current
        This returns if there was explosion or not in order to reset the game from the controller.
        """
        self.recorded_states[self.COUNTERMEM, :] = self.currentState

        # Choose action and play it
        result, selectedAction = self.selectActionUsingEpsilonGreedy(self.currentState)
        if not result.explosion:
            self.update(result)
            self.game.set_flags(self.get_flags())
        self.recorded_actions[self.COUNTERMEM] = selectedAction

        # Get the new state
        if result.explosion:
            self.recorded_targets[self.COUNTERMEM] = result.reward
        else:
            self.currentState = one_hot_encoding(self.game.get_state())
            self.recorded_targets[self.COUNTERMEM] = result.reward + self.discountFactor * self.maxQ(self.currentState)
        self.COUNTERMEM += 1
        if self.COUNTERMEM == self.memorySize:
            self.updateParams()
            self.COUNTERMEM = 0
        return result.explosion

    def predict_Q_value(self, currentState):
        Q = []
        for k in range(self.n_actions):
            try:
                Q.append(np.squeeze(self.regressors[k].predict(currentState[np.newaxis, ...])))
            except NotFittedError:
                Q.append(1 / self.n_actions)  # equiprobability
        return np.array(Q)

    def selectActionUsingEpsilonGreedy(self, currentState):
        Q = self.predict_Q_value(currentState)
        takeMaxAction = self.rng.binomial(n=1, p=1 - self.epsilonProb, size=1)[0]
        selectedActionId = self.getValidAction(Q, takeMaxAction)
        coords = selectedActionId // self.config.width, selectedActionId % self.config.width
        result = self.game.select(*coords)
        return result, selectedActionId

    def getValidAction(self, Q, maxAction):
        expo = np.reshape(np.asarray(self.game.exposed), -1)
        tmp = np.asarray(np.logical_not(expo), 'float')
        tmp[tmp == 0] = -np.inf
        Q[expo] = np.abs(Q[expo])
        if maxAction:
            validActionsQ = tmp * Q
            return np.argmax(validActionsQ)
        else:
            indices = np.arange(len(self.actionsId))
            self.rng.shuffle(indices)
            counter = 0
            while tmp[indices[counter]] == -np.inf:
                counter += 1
            return indices[counter]

    def maxQ(self, currentState):
        """
        Get The max Q Value using the best action
        """
        Q = self.predict_Q_value(currentState)
        expo = np.reshape(np.asarray(self.game.exposed), -1)
        tmp = np.asarray(np.logical_not(expo), 'float')
        tmp[tmp == 0] = -np.inf
        Q[expo] = np.abs(Q[expo])
        validActionsQ = tmp * Q
        return np.max(validActionsQ)

    def ResetAgentState(self, game):
        """
        This is called when a game reaches end and we need to start a new game. The new game is passed as a param
        """
        self.game = game
        self.currentState = one_hot_encoding(self.game.get_state())

    def updateParams(self):
        print("UPDATING Model")
        selectedActions = np.asarray(self.recorded_actions)
        for k in range(len(self.actionsId)):
            select = selectedActions == k
            A = self.recorded_states[select]
            b = self.recorded_targets[select]
            if A.shape[0] > 0:
                self.regressors[k].fit(A, b.flatten())
        self.SaveParams()
        print("MODEL WAS SAVED")
        # decrease epsilon probability for epsilon greedy selection
        self.epsilonProb = np.maximum(0, self.epsilonProb - 0.02)
        print(self.COUNTERWINS)
        # self.COUNTERWINS = 0
        self.n_wins.append(self.COUNTERWINS)

    def update(self, result):
        for position in result.new_squares:
            self.exposed_squares.add((position.x, position.y))

    def next(self):
        currentState = one_hot_encoding(self.game.get_state())
        counter = 0
        Q = self.predict_Q_value(currentState)
        zipped = list(zip(Q, self.actionsId))
        zipped.sort(key=lambda t: t[0], reverse=True)
        q, aIds = zip(*zipped)
        row = aIds[counter] // self.config.width
        col = aIds[counter] % self.config.width
        while not self.game.IsActionValid(row, col):
            counter += 1
            row = aIds[counter] // self.config.width
            col = aIds[counter] % self.config.width
        return row, col

    def SaveParams(self):
        folder = self.savePath + str(self.COUNTERWINS)
        os.makedirs(folder, exist_ok=True)
        for k in range(self.n_actions):
            filename = os.path.join(folder, "regressor_%s" % k)
            joblib.dump(self.regressors[k], filename)
        np.savetxt(os.path.join(folder, "n_wins.txt"), np.array(self.n_wins, dtype='int'))

    def LoadParams(self, folder):
        for k in range(self.n_actions):
            filename = os.path.join(folder, "regressor_%s" % k)
            self.regressors[k] = joblib.load(filename)
