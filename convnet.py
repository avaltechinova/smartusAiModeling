import numpy
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras_preprocessing.image import ImageDataGenerator
from multipledispatch import dispatch
import numpy as np


class ConvNet:
    def __init__(self, conv_base_name=None, augment_data=False, fine_tuning=False):
        self.__history = None
        self.__filter_power = 6
        self.__model = models.Sequential()

        if conv_base_name is None:
            # simplest stacked convolutional-pooling layer
            self.__model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(180, 240, 3)))
            # self.__model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            self.__model.add(layers.MaxPool2D(2, 2))
        elif conv_base_name == 'VGG16':
            from keras.applications import VGG16
            self.__conv_base = VGG16(weights='imagenet',
                                     include_top=False,
                                     input_shape=(180, 240, 3))

            if augment_data:
                # unfreeze the top layers of VGG16 and fine tune them
                if fine_tuning:
                    self.__conv_base.trainable = True

                    for layer in self.__conv_base.layers:
                        if 'block5_conv' in layer.name:
                            layer.trainable = True
                        else:
                            layer.trainable = False

                    self.__model.add(self.__conv_base)
                    self.__model.add(layers.Flatten())
                    dense_layers = models.load_model('model.hdf5')
                    self.__model.add(dense_layers)
                else:
                    # train model end to end with data augmentation
                    self.__conv_base.trainable = False
                    self.__model.add(self.__conv_base)
                    self.__model.add(layers.Flatten())

    def add_conv_layers(self, nr_layers):
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

    def add_dense_layers(self, drop_out, nr_hidden_neurons=256, activation_func='relu', regularizer=None):
        # dense layers
        self.__model.add(layers.Dense(nr_hidden_neurons,
                                      activation=activation_func,
                                      kernel_regularizer=regularizers.l1_l2(regularizer['l1'], regularizer['l2'])))

        if drop_out is not None:
            # dropout layer
            self.__model.add(layers.Dropout(drop_out))

        # output layer
        self.__model.add(layers.Dense(1))

    def configure(self, learning_rate):
        self.__model.compile(loss='mean_squared_error',
                             optimizer=optimizers.RMSprop(learning_rate=learning_rate),
                             metrics=['mean_squared_error'])

    @dispatch(ImageDataGenerator, ImageDataGenerator, int, int, int, int)
    def train(self, train_generator, validation_generator, steps_per_epoch, nr_epochs, val_steps, batch_size):
        self.__history = self.__model.fit(train_generator,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=nr_epochs,
                                          validation_data=validation_generator,
                                          validation_steps=val_steps,
                                          batch_size=batch_size)

    @dispatch(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int)
    def train(self, train_features, train_labels, val_features, val_labels, nr_of_epochs, batch_size):
        self.__history = self.__model.fit(train_features,
                                          train_labels,
                                          epochs=nr_of_epochs,
                                          batch_size=batch_size,
                                          validation_data=(val_features, val_labels))

    def extract_features(self, generator, batch_size, nr_samples):
        features = np.zeros(shape=(nr_samples, 5, 7, 512))
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
