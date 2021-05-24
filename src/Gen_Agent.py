import numpy as np
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianNoise

from sem_game import Board

class Gen_Agent:
    def __init__(self, model_size = (32, 32)):
        inputs = keras.Input(shape=((3,4,1)))
        x = Dense(32, activation="relu", name="dense_1")(inputs)
        y = Dense(32, activation="relu", name="dense_2")(x)
        outputs = Dense(10, activation="softmax", name="predictions")(y)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # self.model = Sequential()
        # self.model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        # self.model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        # self.model.add(Dense(12, activation='sigmoid'))
        # compile the model
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    def predict(self, board):
        return self.model.predict(board)
    def _call(self, inputs):
        print(input)
        x = self.model.layers[0](inputs)
        print(x)
        y = self.model.layers[1](x)
        return self.model.layers[2](y)

ga = Gen_Agent()
#ga.model = ga.model.layers.GaussianNoise()
board = Board()
board.make_move((0,0))
print(ga.predict(board.get_one_hot().reshape(-1, 3, 4, 1)))
print(GaussianNoise(ga.model))
print(ga.predict(board.get_one_hot().reshape(-1, 3, 4, 1)))
