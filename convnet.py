from keras import layers
from keras import models
from keras import optimizers


class ConvNet:
    def __init__(self):
        self.__history = None
        self.__filter_power = 6
        self.__model = models.Sequential()
        # base layers
        self.__model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(130, 240, 3)))
        # self.__model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.__model.add(layers.MaxPool2D(2, 2))

    def add_layers(self, nr_layers):
        # layers set by the user
        for i in range(nr_layers):
            if self.__filter_power <= 7:
                self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
                # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
                self.__model.add(layers.MaxPool2D(2, 2))
                self.__model.summary()
                self.__filter_power += 1
            else:
                self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
                # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
                # self.__model.add(layers.Conv2D(2 ** self.__filter_power, (3, 3), activation='relu'))
                self.__model.add(layers.MaxPool2D(2, 2))

                if self.__filter_power <= 9:
                    self.__filter_power += 1

            # if nr_layers == 3:
            #     self.__model.summary()

        # flatten conv. layer
        self.__model.add(layers.Flatten())
        # dense layers
        self.__model.add(layers.Dense(512, activation='relu'))
        self.__model.add(layers.Dense(1))

    def configure(self):
        self.__model.compile(loss='mean_squared_error',
                             optimizer=optimizers.RMSprop(lr=1e-4),
                             metrics=['mean_squared_error'])

    def train(self, train_generator, validation_generator):
        self.__history = self.__model.fit(train_generator,
                                          steps_per_epoch=117,
                                          epochs=1,
                                          validation_data=validation_generator,
                                          validation_steps=30)

    def save_model(self):
        self.__model.save('test.hdf5')

    @property
    def history(self):
        return self.__history
