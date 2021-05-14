from datetime import datetime

import modeling_config
from modeling_config import TrainingConfig, CrossValidationConfig, ConvNetConfig, DataConfig
import cross_validation as cv
import os
import pandas as pd


if __name__ == '__main__':
    prefix = '/home/adriano/Desktop/avaltech/ai/models/'
    problem = 'regression'
    body_part = 'garsup'
    current_day = datetime.now().date()
    current_time = datetime.now().strftime('%H:%M:%S')
    problem_path = os.path.join(prefix, problem)
    body_part_path = os.path.join(problem_path, body_part)
    folder_day = os.path.join(body_part_path, str(current_day))
    folder_time = os.path.join(folder_day, current_time)
    path_exists = os.path.isdir(folder_time)

    if not path_exists:
        os.makedirs(folder_time)

    # load dataset
    if body_part == 'garsup':
        target = 'Rump'
        # load csv data with pandas
        df = \
            pd.read_csv(r'/home/adriano/Desktop/avaltech/data_sets/dados_filtrados/'
                        r'ds_garsup.csv')
        # get an image sample at random from each animal's image group id
        filtered_df = df.groupby('RGN').apply(lambda x: x.sample(1)).reset_index(drop=True)
        # select columns data to create the dataset
        df_dataset = filtered_df[[target, 'IMAGE']]
    else:
        target = 'Rib Fat'
        # load csv data with pandas
        df = pd.read_csv(r'/home/adriano/Desktop/avaltech/data_sets/dados_filtrados/filter_1/depth/ds_cossup.csv')
        # get an image sample at random from each animal's image group id
        filtered_df = df.groupby('RGN').apply(lambda x: x.sample(1)).reset_index(drop=True)
        # select columns data to create the dataset
        df_dataset = filtered_df[[target, 'IMAGE']]

    # print dataset dimensions
    print(f'original dataset dimension: {df_dataset.shape}')

    # configuration of the modeling process
    # training configuration
    train_config = TrainingConfig(batch_size=64,
                                  nr_epochs=200,
                                  data_augmentation=False,
                                  outlier_detect=False,
                                  fine_tuning=False)
    # validation configuration
    validation_config = CrossValidationConfig(nr_splits=5,
                                              shuffle=True)

    # CNN configuration
    regularizers = {'l1': 0., 'l2': 0.}
    cnn_config = ConvNetConfig(conv_base='VGG16',
                               nr_hidden_neurons=256,
                               drop_out=0.5,
                               learning_rate=1e-6,
                               regularizers=regularizers)

    data_config = DataConfig(animal_weight=False)

    modeling_config.save_configuration(folder_time, validation_config, train_config, cnn_config, data_config)
    cv.run_k_fold_cv(folder_time, validation_config, train_config, cnn_config, df_dataset, target, data_config)
