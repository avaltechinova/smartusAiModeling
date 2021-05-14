import numpy as np


class CrossValidationConfig:
    def __init__(self, cv_type='k_fold', nr_splits=5, shuffle=False, batch_size=32):
        self.__cv_type = cv_type
        self.__nr_splits = nr_splits
        self.__shuffle = shuffle
        self.__batch_size = batch_size

    @property
    def cross_validation_type(self):
        return self.__cv_type

    @property
    def nr_splits(self):
        return self.__nr_splits

    @property
    def shuffle(self):
        return self.__shuffle

    def validation_steps(self, nr_samples):
        return np.ceil(nr_samples / self.__batch_size)


class TrainingConfig:
    def __init__(self, batch_size=32, nr_epochs=10, data_augmentation=False, outlier_detect=False, fine_tuning=False):
        self.__batch_size = batch_size
        self.__nr_epochs = nr_epochs
        self.__data_augmentation = data_augmentation
        self.__outlier_detect = outlier_detect
        self.__fine_tuning =  fine_tuning

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def nr_epochs(self):
        return self.__nr_epochs

    @property
    def data_augmentation(self):
        return self.__data_augmentation

    @property
    def outlier_detect(self):
        return self.__outlier_detect

    @property
    def fine_tuning(self):
        return self.__fine_tuning

    def steps_per_epoch(self, nr_samples):
        return np.ceil(nr_samples / self.__batch_size)


class ConvNetConfig:
    def __init__(self, conv_base=None,
                 nr_hidden_neurons=256,
                 activation='relu',
                 drop_out=None,
                 learning_rate=1e-4,
                 regularizers=dict()):
        self.__conv_base = conv_base
        self.__nr_hidden_neurons = nr_hidden_neurons
        self.__activation = activation
        self.__drop_out = drop_out
        self.__learning_hate = learning_rate
        self.__regularizers = regularizers

    @property
    def conv_base(self):
        return self.__conv_base

    @property
    def nr_hidden_neurons(self):
        return self.__nr_hidden_neurons

    @property
    def activation(self):
        return self.__activation

    @property
    def drop_out(self):
        return self.__drop_out

    @property
    def learning_rate(self):
        return self.__learning_hate

    @property
    def regularizers(self):
        return self.__regularizers


class DataConfig:
    def __init__(self, animal_weight=False):
        self.__animal_weight = animal_weight

    @property
    def animal_weight(self):
        return self.__animal_weight


def save_configuration(path, validation_config, train_config, cnn_config, data_config):
    with open(path + '/modeling_config.txt', 'w') as f:
        print('---------------------------------------------------------', file=f)
        print('Validation', file=f)
        print('---------------------------------------------------------', file=f)
        print(f'type: {validation_config.cross_validation_type}', file=f)
        print(f'number of splits: {validation_config.nr_splits}', file=f)
        print(f'shuffle: {validation_config.shuffle}', file=f)
        print('\n', file=f)
        print('---------------------------------------------------------', file=f)
        print('Training', file=f)
        print('---------------------------------------------------------', file=f)
        print(f'data augmentation: {train_config.data_augmentation}', file=f)
        print(f'fine tuning: {train_config.fine_tuning}', file=f)
        print(f'outlier detection: {train_config.outlier_detect}', file=f)
        print(f'batch size: {train_config.batch_size}', file=f)
        print(f'number of epochs: {train_config.nr_epochs}', file=f)
        print('\n', file=f)
        print('---------------------------------------------------------', file=f)
        print('CNN', file=f)
        print('---------------------------------------------------------', file=f)
        print(f'convolutional base: {cnn_config.conv_base}', file=f)
        print(f'number of hidden neurons: {cnn_config.nr_hidden_neurons}', file=f)
        print(f'activation function: {cnn_config.activation}', file=f)
        print(f'drop out: {cnn_config.drop_out}', file=f)
        print(f'learning rate: {cnn_config.learning_rate}', file=f)
        print(f'regularizers: {cnn_config.regularizers}', file=f)
        print('\n', file=f)
        print('---------------------------------------------------------', file=f)
        print('Data', file=f)
        print('---------------------------------------------------------', file=f)
        print(f'add animal data: {data_config.animal_weight}', file=f)
