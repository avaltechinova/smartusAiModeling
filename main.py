from datetime import datetime
import cross_validation as cv
import os


if __name__ == '__main__':
    prefix = '/home/adriano/Desktop/avaltech/ai/models/'
    current_day = datetime.now().date()
    current_time = datetime.now().strftime('%H:%M:%S')
    folder_day = os.path.join(prefix, str(current_day))
    folder_time = os.path.join(folder_day, current_time)
    path_exists = os.path.isdir(folder_time)

    if not path_exists:
        os.makedirs(folder_time)

    cv.run_k_fold_cv(folder_time)
