import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils
SYS_PATH = '/u/ajagadish/ermi'


class Badham2017(nn.Module):
    """
    load human data from Badham et al. 2017
    """

    def __init__(self, noise=0., return_prototype=False, device='cpu'):
        super(Badham2017, self).__init__()

        self.device = torch.device(device)
        self.data = pd.read_csv(
            f'{SYS_PATH}/categorisation/data/human/badham2017deficits.csv')
        self.num_choices = 1
        self.num_dims = 3
        self.noise = noise
        self.return_prototype = return_prototype

    def sample_batch(self, participant):

        stacked_task_features, stacked_targets, stacked_human_targets, stacked_prototypes = self.get_participant_data(
            participant)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(
            stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(
            stacked_human_targets, batch_first=True)

        if self.return_prototype:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, stacked_prototypes, None
        else:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, None

    def get_participant_data(self, participant):

        inputs_list, targets_list, human_targets_list, prototype_list = [], [], [], []

        # get data for the participant
        data_participant = self.data[self.data['participant'] == participant]
        conditions = np.unique(data_participant['condition'])
        for task_type in conditions:

            # get features and targets for the task
            input_features = np.stack(
                [eval(val) for val in data_participant[data_participant.condition == task_type].all_features.values])
            human_choices = data_participant[data_participant.condition ==
                                             task_type].choice
            true_choices = data_participant[data_participant.condition ==
                                            task_type].correct_choice

            # convert human choices to 0s and 1s
            targets = np.array(
                [1. if choice == 'j' else 0. for choice in true_choices])
            human_targets = np.array(
                [1. if choice == 'j' else 0. for choice in human_choices])

            # flip features, targets and humans choices
            if np.random.rand(1) > 0.5:
                input_features = 1. - input_features
                targets = 1. - targets
                human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate(
                (input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)

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

            # compute mean of each features for a category
            prototype_list.append([np.mean(input_features[targets == 0], axis=0), np.mean(
                input_features[targets == 1], axis=0)])

        return inputs_list, targets_list, human_targets_list, prototype_list


class Devraj2022(nn.Module):
    """
    load human data from categorisation task by Devraj 2022 
    """

    def __init__(self, noise=0., return_prototype=False, device='cpu'):
        super(Devraj2022, self).__init__()

        self.device = torch.device(device)
        self.data = pd.read_csv(
            f'{SYS_PATH}/categorisation/data/human/devraj2022rational.csv')
        self.data = self.data[self.data.condition == 'control']
        self.num_choices = 1
        self.num_dims = 6
        self.noise = noise
        self.return_prototype = return_prototype

    def sample_batch(self, participant):

        stacked_task_features, stacked_targets, stacked_human_targets, stacked_prototypes, stacked_stimulus_ids = self.get_participant_data(
            participant)
        sequence_lengths = [len(data)for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)
        padded_targets = rnn_utils.pad_sequence(
            stacked_targets, batch_first=True)
        padded_human_targets = rnn_utils.pad_sequence(
            stacked_human_targets, batch_first=True)
        padded_stimulus_ids = rnn_utils.pad_sequence(
            stacked_stimulus_ids, batch_first=True)

        if self.return_prototype:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, stacked_prototypes, padded_stimulus_ids
        else:
            return packed_inputs, sequence_lengths, padded_targets, padded_human_targets, padded_stimulus_ids

    def get_participant_data(self, participant):

        inputs_list, targets_list, human_targets_list, prototype_list, stimulus_id_list = [], [], [], [], []

        # get data for the participant
        data_participant = self.data[self.data['participant'] == participant]
        conditions = np.unique(data_participant['condition'])
        for condition in conditions:

            # get features and targets for the task
            input_features = np.stack(
                [eval(val) for val in data_participant[data_participant.condition == condition].all_features.values])
            human_choices = data_participant[data_participant.condition ==
                                             condition].choice.values
            true_choices = data_participant[data_participant.condition ==
                                            condition].correct_choice.values

            # covert to 0 or 1 indexing
            targets = true_choices  # -1
            human_targets = human_choices  # -1

            # flip features, targets and humans choices
            if np.random.rand(1) > 0.5:
                input_features = 1. - input_features
                targets = 1. - targets
                human_targets = 1. - human_targets

            # concatenate all features and targets into one array with placed holder for shifted target
            sampled_data = np.concatenate(
                (input_features, targets.reshape(-1, 1), targets.reshape(-1, 1)), axis=1)

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

            # concatenate values from columns 'prototype_feature1' to 'prototype_feature6' from data_participant[data_participant.condition==condition]
            columns = [f'prototype_feature{i+1}' for i in range(self.num_dims)]
            data_condition = data_participant[data_participant.condition == condition]
            category_prototypes = []
            for category in np.unique(data_condition.category):
                category_prototypes.append(np.array(
                    [val for val in data_condition[data_condition.category == category][columns].values[0]]))
            prototype_list.append(category_prototypes)

            # stack all stimuli_id across all tasks
            stimulus_id_list.append(torch.from_numpy(
                data_condition.stimulus_id.values.reshape(-1, 1)))

        return inputs_list, targets_list, human_targets_list, prototype_list, stimulus_id_list
