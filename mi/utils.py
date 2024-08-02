import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import re
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights, TransformerDecoderRegression, TransformerDecoderRegressionLinearWeights
import ivon
from model_utils import parse_model_path, get_wd_from_std, compute_kld
from os import getenv
SYS_PATH = getenv('BERMI_DIR')

def save_klds():

    task_features = {'model_max_steps': 10}
    data='claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_unknown'
    paired=True
    device='cpu'
    std=0.1
    lr=3e-4
    klds,lambadas = [], []

    for ess in [1.0, None, 1000000.0]:
        
        # parse model parameters
        model_path = f'{SYS_PATH}/decisionmaking/trained_models/env=claude_dim2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossvariational_ess{str(ess)}_std0.1_run=0.pt'
        num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, task_features)

        # initialise model
        if paired:
            model = TransformerDecoderLinearWeights(num_input=2, num_output=1, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

        else:
            model = TransformerDecoderClassification(num_input=2, num_output=1, num_hidden=num_hidden,
                                                        num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

        
        _, model_state_dict, optmizer_state_dict, _ = torch.load(
                model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)

        lambda_ = len(pd.read_csv(f'{SYS_PATH}/decisionmaking/data/{data}.csv')) if ess is None else ess
        wd = get_wd_from_std(std, lambda_)
        optimizer = ivon.IVON(model.parameters(), lr=lr, ess=lambda_, weight_decay=wd)
        optimizer.load_state_dict(optmizer_state_dict)

        kld = compute_kld(optimizer, std)
        lambadas.append(lambda_)
        klds.append(kld)

    # save klds
    df = pd.DataFrame({'lambda': lambadas, 'kld': klds})
    df.to_csv(f'{SYS_PATH}/decisionmaking/stats/klds.csv', index=False)