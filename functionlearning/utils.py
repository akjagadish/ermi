import openml
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd


def save_real_data():
    # 'OpenML-CTR23 - A curated tabular regression benchmarking suite'
    benchmark_suite = openml.study.get_suite(353)
    num_points = 650
    data = []
    t = 0

    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        if task_id == 361618:  # ignore this task as it is not loadable
            continue
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        features, targets = task.get_X_and_y()  # get the data

        if (features.shape[1] < 99999) and (not np.isnan(features).any()):

            # Normalize the features and targets
            scaler = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(features)
            features = scaler.transform(features)
            scaler_targets = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(targets.reshape(-1, 1))
            targets = scaler_targets.transform(targets.reshape(-1, 1))

            # Select the best feature
            features = SelectKBest(
                f_classif, k=1).fit_transform(features, targets)

            if features.shape[0] < num_points:
                xs = [features]
                ys = [targets]
            else:
                xs = np.array_split(features, features.shape[0] // num_points)
                ys = np.array_split(targets, targets.shape[0] // num_points)
            for (x, y) in zip(xs, ys):
                for i in range(x.shape[0]):
                    data.append([x[i].tolist(), y[i], i, t])
                t += 1

            df = pd.DataFrame(
                data, columns=['input', 'target', 'trial_id', 'task_id'])
            df.to_csv('real_data.csv')

        else:
            print('not valid data')
