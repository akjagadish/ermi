import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Beta, Bernoulli, Categorical
import torch.multiprocessing as mp
from model_utils import MLP
SYS_PATH = '/u/ajagadish/ermi/'


class FunctionlearningTask(nn.Module):
    """
    Function learning
    """

    def __init__(self, data, max_steps=20, sample_to_match_max_steps=False, num_dims=3, batch_size=64, mode='train', split=[0.8, 0.1, 0.1], device='cpu', num_tasks=10000, noise=0., shuffle_trials=False, shuffle_features=True, normalize_inputs=True):
        """
        Initialise the environment
        Args:
            data: path to csv file containing data
            max_steps: number of steps in each episode
            num_dims: number of dimensions in each input
            batch_size: number of tasks in each batch
        """
        super(FunctionlearningTask, self).__init__()
        data = pd.read_csv(data)
        data = data.groupby('task_id').filter(lambda x: len(x) <= max_steps)
        # reset task_ids based on number of unique task_id
        data['task_id'] = data.groupby('task_id').ngroup()
        self.data = data
        self.device = torch.device(device)
        # TODO: max steps is equal to max_steps in the dataset
        self.max_steps = max_steps
        self.sample_to_match_max_steps = sample_to_match_max_steps
        self.num_choices = 1
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.mode = mode
        self.split = (torch.tensor(
            [split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * self.data.task_id.nunique()).int()
        self.noise = noise
        self.shuffle_trials = shuffle_trials
        self.shuffle_features = shuffle_features
        self.normalize = normalize_inputs

    def return_tasks(self, mode=None):
        mode = self.mode if mode is None else mode
        if mode == 'train':
            tasks = np.random.choice(self.data.task_id.unique(
            )[:self.split[0]], self.batch_size, replace=False)
        elif mode == 'val':
            self.batch_size = self.split[1] - self.split[0]
            tasks = self.data.task_id.unique()[self.split[0]:self.split[1]]
        elif mode == 'test':
            self.batch_size = self.split[2] - self.split[1]
            tasks = self.data.task_id.unique()[self.split[1]:]

        return tasks

    def sample_batch(self):

        data = self.data[self.data.task_id.isin(self.return_tasks())]
        data['input'] = data['input'].apply(
            lambda x: list(map(float, x.strip('[]').split(','))))
        # assert that sample_to_match only when shuffle_trials is True
        assert self.shuffle_trials if self.sample_to_match_max_steps else not self.sample_to_match_max_steps, "sample_to_match_max_steps should be True only when shuffle_trials is True"
        # shuffle the order of trials within a task but keep all the trials
        if self.shuffle_trials:
            data = data.groupby('task_id').apply(
                lambda x: x.sample(frac=1)).reset_index(drop=True)
            # sample with replacement to match self.max_steps for each task_id
            if self.sample_to_match_max_steps:
                data = data.groupby('task_id').apply(lambda x: x.sample(
                    n=self.max_steps, replace=True)).reset_index(drop=True)
        # group all inputs for a task into a list
        data = data.groupby('task_id').agg(
            {'input': list, 'target': list}).reset_index()

        # off set targets by 1 trial and randomly add zeros or ones in the beggining
        data['shifted_target'] = data['target'].apply(
            lambda x: [1. if torch.rand(1) > 0.5 else 0.] + x[:-1])

        def stacked_normalized(data):
            data = np.stack(data)
            return (data - data.min())/(data.max() - data.min()+1e-6)
        stacked_task_features = [torch.from_numpy(np.concatenate((stacked_normalized(task_input_features) if self.normalize else np.stack(task_input_features), stacked_normalized(
            task_targets).reshape(-1, 1) if self.normalize else np.stack(task_targets).reshape(-1, 1)), axis=1)) for task_input_features, task_targets in zip(data.input.values, data.shifted_target.values)]
        stacked_targets = [torch.from_numpy(
            stacked_normalized(task_targets) if self.normalize else np.stack(task_targets)) for task_targets in data.target.values]
        sequence_lengths = [len(task_input_features)
                            for task_input_features in data.input.values]
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)
        if self.shuffle_features:
            # permute the order of features in packed inputs but keep the last dimension as is
            task_features = packed_inputs[..., -1]
            task_features = task_features[..., np.random.permutation(
                task_features.shape[2])]
            packed_inputs[..., :-1] = task_features

        return packed_inputs.to(self.device), sequence_lengths, stacked_targets
