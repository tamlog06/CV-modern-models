import Keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization


class AlexNet:
    def __init__(self, num_classes):
        self.model = Sequential()

        # 1st layer
        self.model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(224, 224, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 2nd layer
        self.model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd layer
        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))

        # 4th layer
        self.model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))

        # 5th layer
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())

        # 6th layer
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        # 7th layer
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        # 8th layer
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

        return self.model




