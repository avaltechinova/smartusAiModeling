from datetime import datetime
import cross_validation as cv
import os


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

    cv.run_k_fold_cv(folder_time, data_augmentation=False, conv_base='VGG16', body_part=body_part)
