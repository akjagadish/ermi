import pandas as pd
import numpy as np

num_tasks = 1000
num_features = 4
num_points = 650

data = []

for t in range(num_tasks):
    x = np.random.randn(num_points, num_features)
    w = np.random.randn(num_features)
    y = ((x @ w) > 0).astype(float)
    for i in range(num_points):
        data.append([x[i].tolist(), y[i], i, t])

df = pd.DataFrame(data, columns=['input', 'target', 'trial_id', 'task_id'])
df.to_csv('../data/linear_data.csv')
