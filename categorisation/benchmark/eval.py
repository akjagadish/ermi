import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn.utils.rnn as rnn_utils
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier


class XGBModel:
    def __init__(self, device='cpu'):
        # load model
        self.model = None
        self.device = device

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        # specify your parameters here
        param = {'objective': 'binary:logistic'}
        self.model = xgb.train(param, dtrain)

    def score(self, X, y):
        dtest = xgb.DMatrix(X)
        y_pred = self.model.predict(dtest)
        accuracy = (y_pred.round() == y).mean()
        self.y_probs = y_pred
        return accuracy


class TabPFNModel:
    def __init__(self, device='cpu', N_ensemble_configurations=32):
        # Initialize model
        self.model = TabPFNClassifier(
            device=device, N_ensemble_configurations=N_ensemble_configurations)

    def fit(self, X, y):
        # Fit the model
        self.model.fit(X, y)

    def score(self, X, y):
        # Make predictions
        y_pred, _ = self.model.predict(X, return_winning_probability=True)
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        return accuracy

    def predict_proba(self, X):
        # Make predictions
        y_probs = self.model.predict_proba(X)
        return y_probs


class ERMI:

    def __init__(self, model_path, beta=1., policy='greedy', device='cpu'):

        # load model and set beta
        self.model = torch.load(model_path, map_location=torch.device('cpu'))[
            1].to(device)
        self.model.eval()
        self.model.beta = beta  # model beta is adjustable at test time
        self.model.device = device
        self.device = device
        self.policy = policy

    def fit(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        # shift y by one position to the right and add batch dimension
        self.shifted_y = torch.stack([torch.cat((torch.tensor([1. if torch.rand(
            1) > 0.5 else 0.]).to(self.device), target)) for target in self.y.unsqueeze(0)]).unsqueeze(2)

    def score(self, X, y):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(y)
        correct = 0.
        y_probs = []
        for (x, y) in zip(X, Y):

            # forward pass for each point individually
            y_pred, y_prob = self.predict(x)
            # take the model predictions for the last point in the sequence
            correct += (y_pred == y)
            y_probs.append(y_prob)
        # compute mean accuracy
        mean_accuracy = correct / len(X)
        self.y_probs = np.stack(y_probs)
        return mean_accuracy

    def make_input(self, x):

        stacked_task_features = torch.concat(
            (self.X, x.reshape(1, -1)), dim=0).unsqueeze(
            0)  # concatenate test input and add batch dimension
        stacked_task_features = torch.cat(
            (stacked_task_features, self.shifted_y), dim=2)  # concatenate with shifted targets
        sequence_lengths = [len(data) for data in stacked_task_features]
        packed_inputs = rnn_utils.pad_sequence(
            stacked_task_features, batch_first=True)  # TODO: check if needed?

        return packed_inputs.float().to(self.device), sequence_lengths

    def predict(self, x):
        inputs, sequence_lengths = self.make_input(x)
        with torch.no_grad():

            # forward pass for each point individually (!) and compute accuracy
            model_choice_probs = self.model(
                inputs, sequence_lengths).to(self.device)

            # sample from model choices probs using binomial distribution or take argmax
            if self.policy == 'binomial':
                model_choices = torch.distributions.Binomial(
                    probs=model_choice_probs).sample()
            elif self.policy == 'greedy':
                model_choices = model_choice_probs.round()

        return model_choices.squeeze()[-1], model_choice_probs.squeeze()[-1]


df = pd.read_csv('data/benchmark/benchmarking_data.csv')
df.input = df['input'].apply(lambda x: np.array(eval(x)))
ermi_model_path = 'trained_models/env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1.pt'
mi_model_path = 'trained_models/env=dim4synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_synthetic.pt'
pfn_model_path = 'trained_models/env=dim4synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_syntheticnonlinear.pt'

models = [LogisticRegression(), SVC(probability=True), XGBModel(), TabPFNModel(), ERMI(
    ermi_model_path), ERMI(mi_model_path), ERMI(pfn_model_path)]
model_names = ['Logistic Regression', 'SVM', 'XGBoost', 'TabPFN',
               'ERMI', 'ERMI (MI)', 'ERMI (PFN)']
num_tasks = len(df['task_idx'].unique())
num_sets = len(df['set_idx'].unique())
performance = np.zeros((len(models), num_tasks, num_sets))
aucs = np.zeros((len(models), num_tasks, num_sets))

for model_idx, model in enumerate(models):
    for task_idx_idx, task_idx in enumerate(df['task_idx'].unique()):
        df_task = df[df['task_idx'] == task_idx]
        for set_idx in df['set_idx'].unique():
            df_set = df_task[df_task['set_idx'] == set_idx]
            training_set = df_set[df_set['point_idx'] < 30]
            test_set = df_set[df_set['point_idx'] >= 30]

            X_train = np.stack(training_set['input'].to_numpy())
            y_train = training_set['target'].to_numpy()
            X_test = np.stack(test_set['input'].to_numpy())
            y_test = test_set['target'].to_numpy()

            model.fit(X_train, y_train)
            performance[model_idx, task_idx_idx,
                        set_idx] = model.score(X_test, y_test)
            # Compute AUC
            if 'Logistic' in model_names[model_idx] or 'SVM' in model_names[model_idx] or 'TabPFN' in model_names[model_idx]:
                aucs[model_idx, task_idx_idx, set_idx] = roc_auc_score(
                    y_test, model.predict_proba(X_test)[:, 1])
            else:
                aucs[model_idx, task_idx_idx, set_idx] = roc_auc_score(
                    y_test, model.y_probs)

print(performance.mean(axis=(1, 2)))
print(aucs.mean(axis=(1, 2)))
np.save('../data/benchmark/benchmark_performance.npy',
        [performance, aucs])
