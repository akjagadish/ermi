import pandas as pd
import numpy as np
from multiprocessing import Pool
from pm import PrototypeModel
import sys
sys.path.append('../')

def process_participant(participant_data):
    pm = PrototypeModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, learn_prototypes=False, prototypes='from_data', loss=loss)
    ll, r2, params = pm.fit_participants(participant_data, num_blocks=num_blocks)
    return ll, r2, params

df = pd.read_csv('../data/human/devraj2022rational.csv')
df = df[df['condition'] == 'control']

num_runs, num_blocks, num_iter = 1, 11, 100
loss = 'mse_transfer'
opt_method = 'minimize'
NUM_TASKS, NUM_FEATURES = 1, 6

# Create a list of dataframes, each containing the data for one participant
participants_data = [df[df['participant'] == i] for i in df['participant'].unique()]

# Create a pool of processes
with Pool() as p:
    results = p.map(process_participant, participants_data)

# Unpack the results
lls, r2s, params_list = zip(*results)

# save the r2 and ll values
lls = np.array(lls)
r2s = np.array(r2s)
np.savez(f'../data/meta_learner/parallel_pm_humans_devrajstask_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}'\
         , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)