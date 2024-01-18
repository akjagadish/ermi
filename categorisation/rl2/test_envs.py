import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import SmithsTask, RMCTask


# task = SmithsTask(rule='linear')
# task.sample_batch()

task = RMCTask(batch_size=64, max_steps=100)
task.sample_batch()