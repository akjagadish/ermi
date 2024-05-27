import openml
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

data = []
num_sets_per_id = 20
points_per_class = 50

benchmark_suite = openml.study.get_suite('OpenML-CC18')
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    if (len(task.class_labels) == 2):
        features, targets = task.get_X_and_y()
        if (features.shape[1] < 100) and (not np.isnan(features).any()) and (len(np.where(targets == 0)[0]) >= points_per_class) and (len(np.where(targets == 1)[0]) > points_per_class):
            # scaling and feature selection
            scaler = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(features)
            features = scaler.transform(features)
            features = SelectKBest(
                f_classif, k=4).fit_transform(features, targets)

            for set_id in range(num_sets_per_id):
                # balanced sampling
                class_0_idx = np.random.choice(
                    np.where(targets == 0)[0], size=(points_per_class,), replace=False)
                class_1_idx = np.random.choice(
                    np.where(targets == 1)[0], size=(points_per_class,), replace=False)
                X = np.concatenate(
                    (features[class_0_idx], features[class_1_idx]), axis=0)
                y = np.concatenate(
                    (targets[class_0_idx], targets[class_1_idx]), axis=0)

                # shuffling
                random_order = np.random.permutation(points_per_class * 2)
                X = X[random_order]
                y = y[random_order]

                for i in range(X.shape[0]):
                    data.append([X[i].tolist(), y[i], i, set_id, task_id])

df = pd.DataFrame(
    data, columns=['input', 'target', 'point_idx', 'set_idx', 'task_idx'])
df.to_csv('../data/benchmark/benchmarking_data.csv')
