from keras import layers
from keras import models
from keras import optimizers


class ConvNet:
    def __init__(self):
        self.__history = None
        # base layers
        self.__model = models.Sequential()
        self.__model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(130, 240, 3)))
        self.__model.add(layers.MaxPool2D(2, 2))
        self.__model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(layers.MaxPool2D(2, 2))
        self.__model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.__model.add(layers.MaxPool2D(2, 2))

    def add_layers(self, nr_layers):
        # layers set by the user
        for i in range(nr_layers):
            self.__model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.__model.add(layers.MaxPool2D(2, 2))

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
                                          epochs=30,
                                          validation_data=validation_generator,
                                          validation_steps=30)

    def save_model(self):
        self.__model.save('test.hdf5')

    @property
    def history(self):
        return self.__history
