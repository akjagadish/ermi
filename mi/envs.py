import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Beta, Bernoulli, Categorical, MultivariateNormal
import torch.multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from model_utils import MLP
import math
SYS_PATH = '/u/ajagadish/ermi/'


class FunctionlearningTask(nn.Module):
    """
    Function learning
    """

    def __init__(self, data, max_steps=20, sample_to_match_max_steps=False, num_dims=3, scale=0.5, batch_size=64, mode='train', split=[0.8, 0.1, 0.1], device='cpu', num_tasks=10000, noise=0., shuffle_trials=False, shuffle_features=True, normalize_inputs=True):
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
        self.scale = scale
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

        def stacked_normalized(data, scale):
            data = np.stack(data)
            return 2 * scale * (data - data.min())/(data.max() - data.min() + 1e-6) - scale
        
        stacked_task_features = [torch.from_numpy(np.concatenate((stacked_normalized(task_input_features, self.scale) if self.normalize else np.stack(task_input_features), stacked_normalized(
            task_targets, self.scale).reshape(-1, 1) if self.normalize else np.stack(task_targets).reshape(-1, 1)), axis=1)) for task_input_features, task_targets in zip(data.input.values, data.shifted_target.values)]
        stacked_targets = [torch.from_numpy(
            stacked_normalized(task_targets, self.scale) if self.normalize else np.stack(task_targets)) for task_targets in data.target.values]
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


class DecisionmakingTask(nn.Module):
    """
    Decision making task
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
        super(DecisionmakingTask, self).__init__()
        data = pd.read_csv(data)
        # filter out tasks with less than or greater than max_steps
        data = data.groupby('task_id').filter(lambda x: len(x) == max_steps)
        # reset task_ids based on number of unique task_id
        data['task_id'] = data.groupby('task_id').ngroup()
        self.data = data
        self.device = torch.device(device)
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

    def sample_batch(self, paired=False):

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

        def stacked_normalized(data):
            data = np.stack(data)
            return (data - data.mean(axis=0))/(data.std(axis=0) + 1e-6)

        stacked_task_features = []
        for task_input_features, task_targets in zip(data.input.values, data.target.values):
            if self.normalize:
                input_features = np.diff(stacked_normalized(np.stack(task_input_features)).reshape(
                    2, self.max_steps // 2, self.num_dims), axis=0).squeeze()
                targets = np.diff(np.stack(
                    task_targets).reshape(-1, 1).reshape(2, self.max_steps // 2, 1), axis=0).squeeze(0)
            else:
                input_features = np.diff(np.stack(task_input_features).reshape(
                    2, self.max_steps // 2, self.num_dims), axis=0).squeeze()
                targets = np.diff(
                    np.stack(task_targets).reshape(-1, 1).reshape(2, self.max_steps // 2, 1), axis=0).squeeze(0)

            concatenated_features = np.concatenate(
                (input_features, targets), axis=1)
            torch_features = torch.from_numpy(concatenated_features)
            stacked_task_features.append(torch_features)

        stacked_task_features = torch.stack(stacked_task_features)
        stacked_task_features[..., -1] = stacked_task_features[..., -
                                                               1] > 0. if torch.rand(1) > 0.5 else stacked_task_features[..., -1] < 0.
        stacked_targets = stacked_task_features[..., -1].clone()

        if not paired:
            # shift the targets by 1 step for all tasks
            stacked_task_features[..., -1] = torch.cat(
                (stacked_targets[:, 0].unsqueeze(1) * torch.bernoulli(torch.tensor(0.5)), stacked_targets[:, :-1]), dim=1
            )

        sequence_lengths = [len(task_input_features)
                            for task_input_features in stacked_targets]

        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)

        # permute the order of features in packed inputs but keep the last dimension as is
        if self.shuffle_features:
            task_features = packed_inputs[..., :-1].clone()
            task_features = task_features[..., np.random.permutation(
                task_features.shape[2])]
            packed_inputs[..., :-1] = task_features

        return packed_inputs.detach().to(self.device), sequence_lengths, stacked_targets.detach().to(self.device)


class SyntheticDecisionmakingTask(nn.Module):
    " Synthetic decision making task"

    def __init__(self, num_dims=2, num_choices=1, direction=False, ranking=False, dichotomized=False, max_steps=10, batch_size=64, mode='train', num_tasks=10000, synthesize_tasks=False, split=[0.8, 0.1, 0.1], noise=0., device='cpu'):
        " Initialize the environment"
        super(SyntheticDecisionmakingTask, self).__init__()
        self.num_dims = num_dims
        self.num_choices = num_choices
        self.max_steps = max_steps

        self.direction = direction
        self.ranking = ranking
        self.dichotomized = dichotomized

        self.sigma = math.sqrt(0.01)
        self.theta = 1.0 * torch.ones(num_dims)
        from pyro.distributions.lkj import LKJCorrCholesky
        self.cov_prior = LKJCorrCholesky(num_dims, eta=2.0 * torch.ones([]))

        self.mode = mode
        self.batch_size = batch_size if mode == 'train' else 1000
        # TODO: now we are not controlling for the number of tasks as we are generating task on the fly
        self.num_tasks = num_tasks
        self.split = (torch.tensor(
            [split[0], split[0]+split[1], split[0]+split[1]+split[2]]) * self.num_tasks).int()
        self.noise = noise
        self.device = torch.device(device)
        self.synthesize_tasks = synthesize_tasks

    def sample_pair(self, weights, L):
        L = L.squeeze()
        inputs_a = MultivariateNormal(torch.zeros(self.num_dims), scale_tril=torch.mm(
            torch.diag(torch.sqrt(self.theta)), L)).sample()
        inputs_b = MultivariateNormal(torch.zeros(self.num_dims), scale_tril=torch.mm(
            torch.diag(torch.sqrt(self.theta)), L)).sample()
        if self.dichotomized:
            inputs_a = (inputs_a > 0).float()
            inputs_b = (inputs_b > 0).float()
        inputs = inputs_a if self.synthesize_tasks else inputs_a - inputs_b
        preds = (weights * inputs).sum(-1, keepdim=True)
        targets = preds if self.synthesize_tasks else torch.bernoulli(
            0.5 * torch.erfc(-preds / (2 * self.sigma)))

        return inputs, targets, inputs_a, inputs_b

    def sample_pair_vectorised(self):
        weights = torch.randn(self.batch_size, self.num_dims, device=self.device)
        if self.direction:
            weights = weights.abs()
        elif self.ranking:
            absolutes = torch.abs(weights)
            _, feature_perm = torch.sort(absolutes, dim=1, descending=True)
            weights = weights.gather(1, feature_perm)

        self.weights = weights.clone()
        L = self.cov_prior.sample([self.batch_size]).to(self.device) 
        while torch.isnan(L).any():
            L = self.cov_prior.sample([self.batch_size]).to(self.device)

        inputs_a = MultivariateNormal(torch.zeros(self.batch_size, self.num_dims, device=self.device), scale_tril=L).sample([self.max_steps])
        inputs_b = MultivariateNormal(torch.zeros(self.batch_size, self.num_dims, device=self.device), scale_tril=L).sample([self.max_steps])
        inputs = inputs_a - inputs_b

        targets = torch.bernoulli(0.5 * torch.erfc(-(weights * inputs).sum(-1, keepdim=True) / (2 * self.sigma)))
        return inputs.detach().to(self.device), targets.detach().to(self.device), inputs_a.detach().to(self.device), inputs_b.detach().to(self.device)
 
    def sample_batch(self, paired=False):
        stacked_task_features = torch.empty(
             self.max_steps, self.batch_size, self.num_dims+self.num_choices, device=self.device)
        support_inputs, support_targets, support_inputs_a, support_inputs_b = self.sample_pair_vectorised()

        # this is a shitty hack to fix an earlier bug
        if self.direction and not self.synthesize_tasks:
            support_targets = 1 - support_targets

        sequence_lengths = [self.max_steps] * self.batch_size

        stacked_task_features[..., :self.num_dims] = support_inputs

        # permute the order of features to have batch_size as the first dimension
        stacked_task_features = stacked_task_features.permute(1, 0, 2)
        stacked_targets = support_targets.permute(1, 0, 2)

        if not paired:
            # shift the targets by 1 step for all tasks
            stacked_task_features[..., [-1]] = torch.cat(
                (stacked_targets[:, -1].unsqueeze(1) * torch.bernoulli(torch.tensor(0.5)), stacked_targets[:, :-1]), dim=1)
        else:
            stacked_task_features[..., [-1]] = stacked_targets

        # pad the sequence to have the same length
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)

        # support_inputs_a.detach().to(device), support_inputs_b.detach().to(device)
        return packed_inputs.detach().to(self.device), sequence_lengths, stacked_targets.detach().to(self.device).squeeze(2)

    def save_synthetic_data(self, num_tasks=5000, paired=True):

        # generate synthetic data
        num_batches = num_tasks//self.batch_size

        last_task_id = 0
        # prepare dataframe
        data = pd.DataFrame(
            columns=['task_id', 'trial_id', 'input', 'target'])

        for _ in range(num_batches):

            # generate synthetic data
            inputs, _, targets = self.sample_batch(paired)
            inputs = inputs[..., :self.num_dims]

            # save inputs and targets into the data dataframe: inputs is of shape (num_tasks, max_steps, num_dims) and targets is of shape (num_tasks, max_steps)
            for task_id, (task_inputs, task_targets) in enumerate(zip(inputs, targets)):
                for trial_id, (input, target) in enumerate(zip(task_inputs, task_targets)):
                    data = pd.concat([data, pd.DataFrame({'task_id': task_id+last_task_id, 'trial_id': trial_id,
                                                          'input': str(input.cpu().numpy().tolist()),
                                                          'target': [target.cpu().numpy().tolist()]})], ignore_index=True)

            # update last task id
            last_task_id = data['task_id'].max()+1

            # save data to csv file
            data.to_csv(
                f'{SYS_PATH}/decisionmaking/data/synthetic_decisionmaking_tasks_dim{self.num_dims}_data{self.max_steps}_tasks{num_tasks}.csv', index=False)


class Binz2022(nn.Module):
    """
    load human data from Binz et al. 2022
    """

    def __init__(self, noise=0., experiment_id=3, device='cpu'):
        super(Binz2022, self).__init__()
        DATA_PATH = f'{SYS_PATH}/decisionmaking/data/human'
        self.device = torch.device(device)
        self.experiment_id = experiment_id
        self.data = pd.read_csv(
            f'{DATA_PATH}/binz2022heuristics_exp{experiment_id}.csv')
        self.num_choices = 1
        self.num_dims = 2 if experiment_id == 3 else 4
        self.noise = noise

    def sample_batch(self, participant, paired=False):

        stacked_task_features, stacked_targets, stacked_human_targets = self.get_participant_data(
            participant, paired)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(
            stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(
            stacked_human_targets, batch_first=True)

        return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, None

    def get_participant_data(self, participant, paired=False):

        inputs_list, targets_list, human_targets_list = [], [], []

        # get data for the participant
        data_participant = self.data[self.data['participant'] == participant]

        for task_id in data_participant.task.unique():

            # filter data for the task
            data_participant_per_task = data_participant[data_participant.task == task_id]

            # get features and targets for the task
            input_features = data_participant_per_task[[
                'x0', 'x1']].values if self.experiment_id == 3 else data_participant_per_task[['x0', 'x1', 'x2', 'x3']].values
            human_targets = data_participant_per_task.choice.values
            targets = data_participant_per_task.target.values

            # flip targets and humans choices
            # if np.random.rand(1) > 0.5:
            #     targets = 1. - targets
            #     human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate(
                (input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)

            if not paired:
                # replace placeholder with shifted targets to the sampled data array
                sampled_data[:, self.num_dims] = np.concatenate((np.array(
                    [0. if np.random.rand(1) > 0.5 else 1.]), sampled_data[:-1, self.num_dims]))

            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(
                sampled_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(
                sampled_data[:, [self.num_dims+1]]))
            human_targets_list.append(
                torch.from_numpy(human_targets.reshape(-1, 1)))

        return inputs_list, targets_list, human_targets_list


class Badham2017(nn.Module):
    """
    load human data from Badham et al. 2017
    """
    
    def __init__(self, noise=0., return_prototype=False, device='cpu'):
        super(Badham2017, self).__init__()
        DATA_PATH = f'{SYS_PATH}/categorisation/data/human'
        self.device = torch.device(device)
        self.data = pd.read_csv(f'{DATA_PATH}/badham2017deficits.csv')
        self.num_choices = 1 
        self.num_dims = 3
        self.noise = noise
        self.return_prototype = return_prototype

    def sample_batch(self, participant, paired=False):
    
        stacked_task_features, stacked_targets, stacked_human_targets, stacked_prototypes = self.get_participant_data(participant)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(stacked_human_targets, batch_first=True)

        if self.return_prototype:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, stacked_prototypes, None
        else:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, None

    def get_participant_data(self, participant):
        
        inputs_list, targets_list, human_targets_list, prototype_list = [], [], [], []
    
        # get data for the participant
        data_participant = self.data[self.data['participant']==participant]
        conditions = np.unique(data_participant['condition'])
        for task_type in conditions:
            
            # get features and targets for the task
            input_features = np.stack([eval(val) for val in data_participant[data_participant.condition==task_type].all_features.values])
            human_choices = data_participant[data_participant.condition==task_type].choice
            true_choices = data_participant[data_participant.condition==task_type].correct_choice
            
            # convert human choices to 0s and 1s
            targets = np.array([1. if choice=='j' else 0. for choice in true_choices])
            human_targets = np.array([1. if choice=='j' else 0. for choice in human_choices])

            # flip features, targets and humans choices 
            if np.random.rand(1) > 0.5:
                input_features = 1. - input_features
                targets = 1. - targets  
                human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate((input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)
            
            # replace placeholder with shifted targets to the sampled data array
            sampled_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), sampled_data[:-1, self.num_dims]))
            
            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(sampled_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(sampled_data[:, [self.num_dims+1]]))
            human_targets_list.append(torch.from_numpy(human_targets.reshape(-1, 1)))

            # compute mean of each features for a category
            prototype_list.append([np.mean(input_features[targets==0], axis=0), np.mean(input_features[targets==1], axis=0)])

     
        return inputs_list, targets_list, human_targets_list, prototype_list  
       

class Devraj2022(nn.Module):
    """
    load human data from categorisation task by Devraj 2022 
    """
    
    def __init__(self, noise=0., return_prototype=False, device='cpu'):
        super(Devraj2022, self).__init__()
        DATA_PATH = f'{SYS_PATH}/categorisation/data/human'
        self.device = torch.device(device)
        self.data = pd.read_csv(f'{DATA_PATH}/devraj2022rational.csv')
        self.data = self.data[self.data.condition=='control']
        self.num_choices = 1 
        self.num_dims = 6
        self.noise = noise
        self.return_prototype = return_prototype

    def sample_batch(self, participant, paired=False):
    
        stacked_task_features, stacked_targets, stacked_human_targets, stacked_prototypes, stacked_stimulus_ids = self.get_participant_data(participant)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(stacked_human_targets, batch_first=True)
        padded_stimulus_ids = rnn_utils.pad_sequence(stacked_stimulus_ids, batch_first=True)

        if self.return_prototype:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, stacked_prototypes, padded_stimulus_ids
        else:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, padded_stimulus_ids
        
    def get_participant_data(self, participant):
        
        inputs_list, targets_list, human_targets_list, prototype_list, stimulus_id_list = [], [], [], [], []
    
        # get data for the participant
        data_participant = self.data[self.data['participant']==participant]
        conditions = np.unique(data_participant['condition'])
        for condition in conditions:
            
            # get features and targets for the task
            input_features = np.stack([eval(val) for val in data_participant[data_participant.condition==condition].all_features.values])
            human_choices = data_participant[data_participant.condition==condition].choice.values
            true_choices = data_participant[data_participant.condition==condition].correct_choice.values
            
            # covert to 0 or 1 indexing
            targets = true_choices#-1
            human_targets = human_choices#-1

            # flip features, targets and humans choices 
            if np.random.rand(1) > 0.5:
                input_features = 1. - input_features
                targets = 1. - targets  
                human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate((input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)
            
            # replace placeholder with shifted targets to the sampled data array
            sampled_data[:, self.num_dims] = np.concatenate((np.array([0. if np.random.rand(1) > 0.5 else 1.]), sampled_data[:-1, self.num_dims]))
            
            # stacking all the sampled data across all tasks
            inputs_list.append(torch.from_numpy(sampled_data[:, :(self.num_dims+1)]))
            targets_list.append(torch.from_numpy(sampled_data[:, [self.num_dims+1]]))
            human_targets_list.append(torch.from_numpy(human_targets.reshape(-1, 1)))

            # concatenate values from columns 'prototype_feature1' to 'prototype_feature6' from data_participant[data_participant.condition==condition]
            columns  = [f'prototype_feature{i+1}' for i in range(self.num_dims)]
            data_condition = data_participant[data_participant.condition==condition]
            category_prototypes = []
            for category in np.unique(data_condition.category):
                category_prototypes.append(np.array([val for val in data_condition[data_condition.category==category][columns].values[0]]))
            prototype_list.append(category_prototypes)

            # stack all stimuli_id across all tasks
            stimulus_id_list.append(torch.from_numpy(data_condition.stimulus_id.values.reshape(-1, 1)))

        return inputs_list, targets_list, human_targets_list, prototype_list, stimulus_id_list 

class Little2022(nn.Module):
    """
    load human data from Badham et al. 2017
    """
    
    def __init__(self, noise=0., return_true_values=True, scale=0.5, device='cpu'):
        super(Little2022, self).__init__()
        DATA_PATH = f'{SYS_PATH}/functionlearning/data/human'
        self.device = torch.device(device)
        self.data = pd.read_csv(f'{DATA_PATH}/little2022functionestimation.csv')
        # filter participants with less than 24 tasks
        self.data = self.data.groupby('participant').filter(lambda x: len(x.task.unique()) == 24) 
        # Fit and transform the 'x' and 'y' columns
        scaler = MinMaxScaler(feature_range=(-scale, scale))
        self.data[['x', 'y']] = scaler.fit_transform(self.data[['x', 'y']])
        self.num_dims = 1
        self.num_choices = 1
        self.return_true_values = return_true_values
        #TODO: provice these contraints a bit better
        num_points = 24 # 6 or 24
        scale = 2 # 1 is zoomed in or 2 is zoomed out
        self.sampling_rate = 48
        self.data = self.data[(self.data.num_points==num_points) & (self.data.scale==scale)]
        self.noise = noise

    def sample_batch(self, participant, paired=False):
    
        stacked_task_features, stacked_targets, stacked_human_targets, stacked_true_data = self.get_participant_data(participant, paired)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(stacked_human_targets, batch_first=True)

        if self.return_true_values:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, stacked_true_data
        else:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets

    def get_participant_data(self, participant, paired):
        
        inputs_list, targets_list, human_targets_list, true_data_list = [], [], [], []
    
        # get data for the participant
        data_participant = self.data[self.data['participant']==participant]
        tasks = np.unique(data_participant['task'])\
        

        for task_id in tasks:
            
            # get features and targets shown to participants for the task
            train_condition = (data_participant.task==task_id) & (data_participant.type=='train')
            train_features = data_participant[train_condition].x.values.reshape(-1,1)
            train_preds = data_participant[train_condition].y.values.reshape(-1,1)
            true_values = np.concatenate((train_features, train_preds), axis=1)  # concatenate train features and preds as two columns

            # get the predictions made by the participants
            test_condition = (data_participant.task==task_id) & (data_participant.type=='test')
            test_features =  data_participant[test_condition].x.values[::self.sampling_rate].reshape(-1,1)
            test_preds = data_participant[test_condition].y.values[::self.sampling_rate].reshape(-1,1)
            pred_values = np.concatenate((test_features, test_preds), axis=1)

            for (x_test, y_test) in zip(test_features, test_preds):

                input_features = np.concatenate((train_features, x_test.reshape(-1,1)), axis=0)
                targets = np.concatenate((train_preds, y_test.reshape(-1,1)), axis=0)

                # concatenate all features and targets into one array with placed holder for shifted target
                sampled_data = np.concatenate((input_features, targets, targets), axis=1)
                
                if not paired:
                    # replace placeholder with shifted targets to the sampled data array
                    sampled_data[:, self.num_dims] = np.concatenate((np.array([targets.mean()]), sampled_data[:-1, self.num_dims]))
                    
                # stacking all the sampled data across all tasks
                inputs_list.append(torch.from_numpy(sampled_data[:, :(self.num_dims+1)]))
                targets_list.append(torch.from_numpy(sampled_data[:, [self.num_dims+1]]))
            
            human_targets_list.append(torch.from_numpy(pred_values))
            true_data_list.append(true_values)
     
        return inputs_list, targets_list, human_targets_list, true_data_list  

