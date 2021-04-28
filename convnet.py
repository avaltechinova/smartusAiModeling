from keras import layers
from keras import models
from keras import optimizers
import numpy as np


class ConvNet:
    def __init__(self, conv_base=None):
        self.__history = None
        self.__filter_power = 6
        self.__model = models.Sequential()

        if conv_base is None:
            # simplest stacked convolutional-pooling layer
            self.__model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(130, 240, 3)))
            # self.__model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            self.__model.add(layers.MaxPool2D(2, 2))
        elif conv_base == 'VGG16':
            from keras.applications import VGG16
            self.__conv_base = VGG16(weights='imagenet',
                                     include_top=False,
                                     input_shape=(130, 240, 3))

    def add_layers(self, nr_layers):
        # layers set by the user
        for i in range(nr_layers):
            # if self.__filter_power <= 7:
            self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), padding='same', activation='relu'))
            # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
            self.__model.add(layers.MaxPool2D(2, 2))
            # self.__model.summary()
            #     self.__filter_power += 1
            # else:
            #     self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), padding='same', activation='relu'))
            #     # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
            #     # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
            #     self.__model.add(layers.MaxPool2D(2, 2))

            if self.__filter_power <= 9:
                self.__filter_power += 1

        # flatten conv. layer
        self.__model.add(layers.Flatten())
        # dropout layer
        self.__model.add(layers.Dropout(0.5))

    def add_dense_layers(self, nr_hidden_neurons=256, activation_func='relu'):
        # dense layers
        self.__model.add(layers.Dense(nr_hidden_neurons, activation=activation_func, input_dim=4 * 7 * 512))
        # dropout layer
        self.__model.add(layers.Dropout(0.5))
        # output layer
        self.__model.add(layers.Dense(1))

    def configure(self):
        self.__model.compile(loss='mean_squared_error',
                             optimizer=optimizers.RMSprop(lr=1e-4),
                             metrics=['mean_squared_error'])

    def train(self, train_generator, validation_generator):
        self.__history = self.__model.fit(train_generator,
                                          steps_per_epoch=128,
                                          epochs=30,
                                          validation_data=validation_generator,
                                          validation_steps=32)

    def train(self, train_features, train_labels, val_features, val_labels, batch_size):
        self.__history = self.__model.fit(train_features,
                                          train_labels,
                                          epochs=30,
                                          batch_size=batch_size,
                                          validation_data=(val_features, val_labels))

    def extract_features(self, generator, batch_size, nr_samples):
        features = np.zeros(shape=(nr_samples, 4, 7, 512))
        labels = np.zeros(shape=nr_samples)
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = self.__conv_base.predict(inputs_batch)
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
            i += 1

            if i * batch_size >= nr_samples:
                break
        return features, labels

    @property
    def model(self):
        return self.__model

    @property
    def conv_base(self):
        return self.__conv_base

    @property
    def history(self):
        return self.__history
