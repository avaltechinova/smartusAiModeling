# -*- coding: utf-8 -*-
import csv

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from convnet import ConvNet
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import set_memory_size_tensorflow as sms


def validate_image_generation(image_generator):
    _counter = 0
    img = None

    for data_batch, labels_batch in image_generator:
        print(f'data batch size: {data_batch.shape}')
        print(f'label batch size: {labels_batch.shape}')
        if _counter > 5:
            print(f'shape of image 0: {data_batch[0].shape}')
            img = data_batch[0]
            break
        _counter += 1


def plot_mse(training_loss, validation_loss, k_fold_path):
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, 'b', label='Training MSE')
    plt.plot(epochs, validation_loss, 'r', label='Validation MSE')
    plt.title('Training and validation MSE')
    plt.legend()
    plt.savefig(k_fold_path + '/train_vs_validation_mse' + '.png')
    plt.clf()


def show_rgb_matrix_as_image(rgb_matrix):
    plt.imshow(rgb_matrix)
    plt.show()


def save_loss_vec(k_fold_path, file_name, loss_vec):
    writer = csv.writer(open(k_fold_path + file_name, 'w'))
    writer.writerow(loss_vec)


def save_model_summary(path, model):
    # open the file
    with open(path + '/model_summary.txt', 'w') as fh:
        # pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def save_model(path, model):
    model.save(path + '/model.hdf5')


def run_k_fold_cv(save_model_path):
    sms.set_memory_size()
    # maximum number of additional stacked convolutional-pooling layers
    max_n_extra_layers = 1
    # load csv data with pandas
    df = pd.read_csv(r'/home/adriano/Desktop/ds_garsup.csv')
    # select columns data to create the dataset
    df_dataset = df[['Rump', 'IMAGE']]
    # print dataset dimensions
    print(f'dataset dimension: {df_dataset.shape}')
    # cast dataset to numpy object in order to use k-fold cross-validation functionality from sklearn
    samples = df_dataset.to_numpy()
    # k-fold cross-validation with k=5
    kf = KFold(n_splits=5, shuffle=False)
    mean_tr_mse_vec = []
    mean_vl_mse_vec = []
    cnn_counter = 0

    # loop over cnn architectures
    for i in range(max_n_extra_layers):
        k_fold_counter = 1
        tr_mse_vec = []
        vl_mse_vec = []
        # create the base cnn model
        conv_net = ConvNet()
        # add additional layers
        conv_net.add_layers(i)
        # create current cnn folder
        cnn_path = os.path.join(save_model_path, 'cnn_' + str(cnn_counter))
        os.makedirs(cnn_path)
        # loop over each fold
        for train, validation in kf.split(samples):
            # print(train)
            # print(validation)
            # create k-fold cross-validation folder
            k_fold_path = os.path.join(cnn_path, 'K_' + str(k_fold_counter))
            os.makedirs(k_fold_path)
            # split dataset into training and validation datasets
            df_train = df_dataset.loc[train]
            df_validation = df_dataset.loc[validation]
            # rescale image elements values to 0-1 range
            train_data_gen = ImageDataGenerator(rescale=1. / 255)
            val_data_gen = ImageDataGenerator(rescale=1. / 255)
            # training image generator
            train_gen = train_data_gen.flow_from_dataframe(dataframe=df_train,
                                                           x_col='IMAGE',
                                                           y_col='Rump',
                                                           target_size=(130, 240),
                                                           class_mode='raw',
                                                           batch_size=32)
            # validation image generator
            val_gen = val_data_gen.flow_from_dataframe(dataframe=df_validation,
                                                       x_col='IMAGE',
                                                       y_col='Rump',
                                                       target_size=(130, 240),
                                                       class_mode='raw',
                                                       batch_size=32)
            conv_net.configure()
            conv_net.train(train_gen, val_gen)
            train_loss = conv_net.history.history['mean_squared_error']
            val_loss = conv_net.history.history['val_mean_squared_error']
            # save training loss
            save_loss_vec(k_fold_path, '/training_loss_vec.csv', train_loss)
            # save validation loss
            save_loss_vec(k_fold_path, '/validation_loss_vec.csv', val_loss)
            # save model summary
            save_model_summary(cnn_path, conv_net.model)
            # save architecture
            save_model(cnn_path, conv_net.model)
            plot_mse(train_loss, val_loss, k_fold_path)

            tr_mse_vec.append(train_loss[-1])
            vl_mse_vec.append(val_loss[-1])
            k_fold_counter += 1

        mean_tr_mse = np.array(tr_mse_vec).mean()
        mean_vl_mse = np.array(vl_mse_vec).mean()
        mean_tr_mse_vec.append(mean_tr_mse)
        mean_vl_mse_vec.append(mean_vl_mse)
        cnn_counter += 1

    plt.plot(np.arange(3, max_n_extra_layers + 3), mean_tr_mse_vec, 'r-')
    plt.plot(np.arange(3, max_n_extra_layers + 3), mean_vl_mse_vec, 'b-')
    plt.show()

# create dataset
# X = np.arange(0, 2 * np.pi, 0.05)
# D = np.sin(X) * np.cos(2 * X)
# X = np.reshape(X, (X.shape[0], 1))
#
# # create training and test sets
# # 15% of samples for testing
# X_train, X_test, D_train, D_test = train_test_split(X, D, test_size=0.15)
#
# # scale input data
# std_scale = StandardScaler()
# std_scale.fit(X_train)
# X_train = std_scale.transform(X_train)
# X_test = std_scale.transform(X_test)
#
# # current number of hidden units
# h_neurons = 5
# max_nr_h_neurons = 50
#
# nr_neurons_vec = []
# mean_tr_mse_vec = []
# mean_vl_mse_vec = []
#
# for i in range(0, max_nr_h_neurons, 4):
#     sum_tr_mse = 0
#     sum_vl_mse = 0
#     # k-fold cross-validation with k=5
#     kf = KFold(n_splits=5, shuffle=True)
#
#     for train, validation in kf.split(X_train):
#         mlp = MLPRegressor(hidden_layer_sizes=(i + 1), activation='tanh', solver='lbfgs', max_iter=2000)
#         # mlp training with training data
#         mlp.fit(X_train[train], D_train[train])
#         # mlp prediction and mean square error for training data
#         mlp_tr_answers = mlp.predict(X_train[train])
#         tr_mse = mean_squared_error(D_train[train], mlp_tr_answers)
#         # mlp prediction mean square error for validation data
#         mlp_vl_answers = mlp.predict(X_train[validation])
#         vl_mse = mean_squared_error(D_train[validation], mlp_vl_answers)
#         sum_tr_mse += tr_mse
#         sum_vl_mse += vl_mse
#
#     print("With %d neurons:" % (i + 1))
#     print("mean training mse: %f - mean validation mse: %f" % (sum_tr_mse / 5, sum_vl_mse / 5))
#     nr_neurons_vec.append(i + 1)
#     mean_tr_mse_vec.append(sum_tr_mse / 5)
#     mean_vl_mse_vec.append(sum_vl_mse / 5)
#
# # plot error curves
# plt.plot(nr_neurons_vec, mean_tr_mse_vec, 'r-')
# plt.plot(nr_neurons_vec, mean_vl_mse_vec, 'b-')
# plt.show()
