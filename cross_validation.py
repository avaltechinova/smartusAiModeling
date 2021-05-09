# -*- coding: utf-8 -*-
import csv

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from convnet import ConvNet
from sklearn.model_selection import KFold
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


def scale_input_for_dense_layer(x_train, x_test):
    # scale input data
    # robust_scale = RobustScaler()
    # robust_scale.fit(x_train)
    # x_train = robust_scale.transform(x_train)
    # x_test = robust_scale.transform(x_test)
    
    min_max_scale = MinMaxScaler()
    min_max_scale.fit(x_train)
    x_train = min_max_scale.transform(x_train)
    x_test = min_max_scale.transform(x_test)

    return x_train, x_test


def apply_pca(x_train, x_test):
    nr_components = 1468
    print("Extracting the top %d eigen faces from %d animals."
          % (nr_components, x_train.shape[0]))
    pca = PCA(n_components=nr_components, svd_solver='randomized',
              whiten=True).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca


def predict(generator, conv_net, nr_samples, batch_size):
    predictions = np.zeros(shape=nr_samples)
    labels = np.zeros(shape=nr_samples)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_net.conv_base.predict(inputs_batch)
        features_batch = np.reshape(features_batch, (len(labels_batch), 5 * 7 * 512))
        prediction_batch = conv_net.model.predict(features_batch)
        vra = prediction_batch.reshape(-1)
        predictions[i * batch_size: (i + 1) * batch_size] = vra
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1

        if i * batch_size >= nr_samples:
            break
    return predictions, labels


def run_k_fold_cv(model_path, validation_config, train_config, cnn_config, df_dataset, target):
    sms.set_memory_size()
    # maximum number of additional stacked convolutional-pooling layers
    max_n_extra_layers = 6
    batch_size = train_config.batch_size
    # cast dataset to numpy object in order to use k-fold cross-validation functionality from sklearn
    samples = df_dataset.to_numpy()
    # normalize animals weights to the 0-1 interval
    normalized_animal_weights = df_dataset['Weight'] / df_dataset['Weight'].max()
    # k-fold cross-validation with k=5
    kf = KFold(n_splits=validation_config.nr_splits, shuffle=validation_config.shuffle)
    mean_tr_mse_vec = []
    mean_vl_mse_vec = []
    cnn_counter = 0

    if train_config.data_augmentation:
        # rescale image elements values to 0-1 range
        train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                            vertical_flip=True,
                                            horizontal_flip=True, )
    else:
        train_data_gen = ImageDataGenerator(rescale=1. / 255)

    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    # loop over cnn architectures
    for i in range(5, max_n_extra_layers):
        k_fold_counter = 1
        tr_mse_vec = []
        vl_mse_vec = []
        # create current cnn folder
        cnn_path = os.path.join(model_path, 'cnn_' + str(cnn_counter))
        os.makedirs(cnn_path)

        # loop over each fold
        for train, validation in kf.split(samples):
            # print(train)
            # print(validation)
            print('--------------------------------------------')
            print(f'CNN ID: {str(cnn_counter)} -- K-Fold: {str(k_fold_counter)}')
            print('--------------------------------------------')
            # create k-fold cross-validation folder
            k_fold_path = os.path.join(cnn_path, 'K_' + str(k_fold_counter))
            os.makedirs(k_fold_path)
            # split dataset into training and validation datasets
            df_train = df_dataset.loc[train]
            df_validation = df_dataset.loc[validation]
            # training image generator
            train_gen = train_data_gen.flow_from_dataframe(dataframe=df_train,
                                                           x_col='IMAGE',
                                                           y_col=target,
                                                           target_size=(180, 240),
                                                           class_mode='raw',
                                                           batch_size=batch_size)

            # validation image generator
            val_gen = val_data_gen.flow_from_dataframe(dataframe=df_validation,
                                                       x_col='IMAGE',
                                                       y_col=target,
                                                       target_size=(180, 240),
                                                       class_mode='raw',
                                                       batch_size=batch_size)

            if cnn_config.conv_base is None:
                # create the base cnn model
                conv_net = ConvNet()
                # add additional layers
                conv_net.add_layers(i)
                conv_net.add_dense_layers(cnn_config.nr_hidden_neurons)
                conv_net.configure(cnn_config.learning_rate)
                conv_net.train(train_gen,
                               val_gen,
                               train_config.steps_per_epoch(len(train)),
                               train_config.nr_epochs,
                               validation_config.validation_steps(len(validation)))
            else:
                conv_net = ConvNet(conv_base=cnn_config.conv_base)
                train_features, train_labels = conv_net.extract_features(train_gen, batch_size, len(train))
                validation_features, validation_labels = conv_net.extract_features(val_gen, batch_size, len(validation))
                train_features = np.reshape(train_features, (len(train), 5 * 7 * 512))
                validation_features = np.reshape(validation_features, (len(validation), 5 * 7 * 512))
                # min_feature, max_feature = train_features.min(), train_features.max()
                # min_val_feature, max_val_feature = validation_features.min(), validation_features.max()
                # scaled_train_input, scaled_validation = \
                #     scale_input_for_dense_layer(train_features, validation_features)
                # min_vra1, max_vra1 = scaled_train_input.min(), scaled_train_input.max()
                # sorted_train = np.sort(train_features)
                # sorted_val = np.sort(validation_features)
                # min_vra2, max_vra2 = scaled_validation.min(), scaled_validation.max()
                train_animal_weights = np.reshape(normalized_animal_weights[train].to_numpy(), (len(train), 1))
                val_animal_weights = np.reshape(normalized_animal_weights[validation].to_numpy(), (len(validation), 1))
                train_features = np.concatenate([train_features, train_animal_weights], axis=1)
                validation_features = np.concatenate([validation_features, val_animal_weights], axis=1)
                conv_net.add_dense_layers(drop_out=cnn_config.drop_out,
                                          nr_hidden_neurons=cnn_config.nr_hidden_neurons)
                conv_net.configure(cnn_config.learning_rate)
                conv_net.train(train_features,
                               train_labels,
                               validation_features,
                               validation_labels,
                               train_config.nr_epochs,
                               batch_size)

            predictions = predict(val_gen, conv_net, len(validation), batch_size=batch_size)
            df_outputs = pd.DataFrame.from_records(np.array(predictions).T, columns=['PREDICTION', 'TARGET'])
            df_outputs.to_csv(k_fold_path + '/cnn_outputs_vs_val_targets.csv')
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

    save_loss_vec(model_path, '/mean_train_mse_vec.csv', mean_tr_mse_vec)
    save_loss_vec(model_path, '/mean_val_mse_vec.csv', mean_vl_mse_vec)
    # plt.plot(mean_tr_mse_vec, 'b-', label='Mean Training MSE')
    # plt.plot(mean_vl_mse_vec, 'r-', label='Mean Validation MSE')
    # plt.title('CNNs performances on K-fold Cross-Validation')
    # plt.legend()
    # plt.show()

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
