"""
Deepmind recipe for DQN:
  - experience replay
  - eval_net and target_net
  - clip the gradient of the squared error between -1 and 1
"""

from minesweeper import GameAI
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, SGDRegressor
import os
import joblib
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras import regularizers


def regression_error(y_true, y_pred):
    inside = K.cast(K.not_equal(y_true, 1e8), K.floatx())
    return K.sum(inside * K.square(y_pred - y_true), axis=[1, 2, 3]) / K.sum(inside, axis=[1, 2, 3])
    # # https://github.com/devsisters/DQN-tensorflow/issues/16
    # x = y_pred - y_true
    # # "where" not available in Keras backend
    # error = tf.where(K.abs(x) < 1.0, 0.5 * K.square(x), K.abs(x) - 0.5)  # condition, true, false
    # return K.sum(inside * error, axis=[1, 2, 3]) / K.sum(inside, axis=[1, 2, 3])


def regression_error_metric(y_true, y_pred):
    return K.mean(regression_error(y_true, y_pred))


def get_unet(width, height):
    # from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    inputs = Input((height, width, 11))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(conv1)

    conv6 = Conv2D(1, (1, 1), activation='linear', kernel_regularizer=regularizers.l2(0.01))(
        concatenate([conv1, conv2], axis=-1))

    model = Model(inputs=[inputs], outputs=[conv6])

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=regression_error, metrics=[regression_error_metric])

    return model


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
        self.padding = 6

        self.COUNTERMEM = 0
        self.COUNTERWINS = 0
        self.COUNTERUPDATES = 0

        self.width = self.config.width
        self.height = self.config.height
        self.exposed_squares = set()
        self.exposed_squares.clear()

        self.rng = np.random.RandomState(123)
        self.actionsId = np.arange(self.width * self.height)
        self.n_actions = self.width * self.height

        self.currentState = self.one_hot_encoding(self.game.get_state())

        # Extract Info from inputDic
        self.inputDic = inputDic
        self.discountFactor = inputDic['discountFactor']
        self.epsilonProb = inputDic['epsilonProb']
        self.memorySize = inputDic['memorySize']
        self.savePath = inputDic['savePath']

        # Add the data for memory
        self.recorded_states = np.zeros([self.memorySize, *self.currentState.shape], dtype='float32')
        self.recorded_targets = np.zeros([self.memorySize,
                                          self.width + 2 * self.padding,
                                          self.height + 2 * self.padding],
                                         dtype='float32')

        self.eval_net = get_unet(self.width + 2 * self.padding, self.height + 2 * self.padding)
        self.target_net = get_unet(self.width + 2 * self.padding, self.height + 2 * self.padding)

        self.q_values = []
        self.game_durations = []

        self.avg_game_duration = []
        self.avg_q_values = []
        self.avg_wins = []
        self.training_loss = []

    def one_hot_encoding(self, game_state):
        # Convert the minesweeper into one hot big state vector (every cell has 10 one hot rep)
        cells = np.asarray(game_state)
        cellsStateCount = 11
        cells[np.isnan(cells.astype(float))] = 9
        cells = np.pad(cells, self.padding, mode='constant', constant_values=10)  # boundary cells
        encoded_state = np.eye(cellsStateCount)[np.asarray(cells, 'int')]
        return encoded_state

    def MoveToNextState(self):
        """
        Move to the next state from the current
        This returns if there was explosion or not in order to reset the game from the controller.
        """
        self.recorded_states[self.COUNTERMEM] = self.currentState
        self.recorded_targets[self.COUNTERMEM] = 1e8

        # Choose action and play it
        result, coords = self.selectActionUsingEpsilonGreedy(self.currentState)

        # Get the new state
        if self.game.is_game_over():
            self.recorded_targets[self.COUNTERMEM, coords[0] + self.padding, coords[1] + self.padding] = result.reward
            self.game_durations.append(self.game.num_moves)
            if self.game.victory():
                self.COUNTERWINS += 1
        else:
            self.update(result)
            self.game.set_flags(self.get_flags())
            self.currentState = self.one_hot_encoding(self.game.get_state())
            q_value = result.reward + self.discountFactor * self.maxQ(self.currentState)
            self.recorded_targets[self.COUNTERMEM,
                                  coords[0] + self.padding,
                                  coords[1] + self.padding] = q_value
            self.q_values.append(q_value)
        self.COUNTERMEM += 1
        if self.COUNTERMEM == self.memorySize:
            self.updateParams()
            self.COUNTERMEM = 0
        return

    def predict_Q_value(self, currentState):
        return np.squeeze(self.eval_net.predict(currentState[np.newaxis, ...]))[self.padding:-self.padding,
               self.padding:-self.padding]

    def selectActionUsingEpsilonGreedy(self, currentState):
        Q = self.predict_Q_value(currentState)
        takeMaxAction = self.rng.binomial(n=1, p=1 - self.epsilonProb, size=1)[0]
        coords = self.getValidAction(Q, takeMaxAction)
        result = self.game.select(*coords)
        return result, coords

    def getValidAction(self, Q, maxAction):
        expo = np.asarray(self.game.exposed)
        Q[expo] = -np.inf
        if maxAction:
            return np.unravel_index(np.argmax(Q), Q.shape)
        else:
            coords = np.argwhere(expo == 0)
            self.rng.shuffle(coords)
            return coords[0]

    def maxQ(self, currentState):
        """
        Get The max Q Value using the best action
        """
        Q = np.squeeze(self.target_net.predict(currentState[np.newaxis, ...]))[self.padding:-self.padding,
            self.padding:-self.padding]
        expo = np.asarray(self.game.exposed)
        Q[expo] = -np.inf
        return np.max(Q)

    def ResetAgentState(self, game):
        """
        This is called when a game reaches end and we need to start a new game. The new game is passed as a param
        """
        self.game = game
        self.currentState = self.one_hot_encoding(self.game.get_state())

    def updateParams(self):
        print(self.COUNTERWINS)
        print("UPDATING eval net")
        history = self.eval_net.fit(self.recorded_states, self.recorded_targets[..., np.newaxis], batch_size=128,
                                    epochs=5)
        self.training_loss.append(history.history['loss'][-1])
        self.COUNTERUPDATES += 1

        if (self.COUNTERUPDATES % 5) == 0:
            print("UPDATING target net")
            self.target_net.set_weights(self.eval_net.get_weights())

        if (self.COUNTERUPDATES % 5) == 0:
            self.SaveParams()
            print("MODEL WAS SAVED")

        # decrease epsilon probability for epsilon greedy selection
        self.epsilonProb = np.maximum(0, self.epsilonProb - 0.02)

        self.avg_wins.append(self.COUNTERWINS / self.memorySize)
        self.avg_q_values.append(np.mean(self.q_values))
        self.q_values = []
        self.avg_game_duration.append(np.mean(self.game_durations))
        self.game_durations = []
        self.COUNTERWINS = 0
        self.plot()

    def update(self, result):
        for position in result.new_squares:
            self.exposed_squares.add((position.x, position.y))

    def next(self):
        currentState = self.one_hot_encoding(self.game.get_state())
        Q = self.predict_Q_value(currentState)
        return self.getValidAction(Q, True)

    def SaveParams(self):
        self.save_model(os.path.join(self.savePath, 'unet'))

    def LoadParams(self, filename):
        self.eval_net.load_weights(filename)

    def plot(self):
        os.makedirs(self.savePath, exist_ok=True)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.avg_q_values)
        plt.xlabel('Training epochs')
        plt.ylabel('Average action value (Q)')
        plt.grid(True)
        plt.title('Average action value (Q) per epoch')
        plt.savefig(os.path.join(self.savePath, "avg_q_values.png"))
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.avg_game_duration)
        plt.xlabel('Training epochs')
        plt.ylabel('Average game duration')
        plt.grid(True)
        plt.title('Average game duration per epoch')
        plt.savefig(os.path.join(self.savePath, "avg_game_duration.png"))
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.avg_wins)
        plt.xlabel('Training epochs')
        plt.ylabel('Percentage of wins')
        plt.grid(True)
        plt.title('Percentage of wins per epoch')
        plt.savefig(os.path.join(self.savePath, "percentage_of_wins.png"))
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.training_loss)
        plt.xlabel('Training epochs')
        plt.ylabel('Training loss')
        plt.grid(True)
        plt.title('Training loss per epoch')
        plt.savefig(os.path.join(self.savePath, "training_loss.png"))
        plt.close()

    def save_model(self, model_name):
        from keras.utils import plot_model
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        json_string = self.eval_net.to_json()
        open(model_name + '_architecture.json', 'w').write(json_string)
        self.eval_net.save_weights(model_name + '_weights.h5', overwrite=True)
        plot_model(self.eval_net, to_file=model_name + '.png', show_shapes=True, show_layer_names=True)
