import openml
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

benchmark_suite = openml.study.get_suite('OpenML-CC18')
num_points = 650
data = []
t = 0

for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    if (len(task.class_labels) == 2):
        features, targets = task.get_X_and_y()  # get the data
        if (features.shape[1] < 99999) and (not np.isnan(features).any()):
            scaler = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(features)
            features = scaler.transform(features)
            features = SelectKBest(
                f_classif, k=4).fit_transform(features, targets)

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

df = pd.DataFrame(data, columns=['input', 'target', 'trial_id', 'task_id'])
df.to_csv('../data/real_data.csv')
