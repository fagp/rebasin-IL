from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import RotatedMNIST, SplitCIFAR100
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import JointTraining
import numpy as np
import torch
import os

n_experiences = 20
number_classes = 10

mb_size = 500
total_epochs = 50
lr = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scenario = RotatedMNIST(
    n_experiences, rotations_list=np.linspace(0, 180, n_experiences).tolist()
)

os.makedirs("./logs", exist_ok=True)
name = "./logs/joint.txt"

# model creation
model = SimpleMLP(
    num_classes=number_classes,
    hidden_size=256,
    hidden_layers=1,
    drop_rate=0.0,
)

# log to text file
text_logger = TextLogger(open(name, "w"))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=False, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[interactive_logger, text_logger],
)

# CREATE THE STRATEGY INSTANCE
cl_strategy = JointTraining(
    model,
    SGD(model.parameters(), lr=lr),
    CrossEntropyLoss(),
    train_mb_size=mb_size,
    train_epochs=total_epochs,
    eval_mb_size=100,
    evaluator=eval_plugin,
    device=device,
)

# TRAINING LOOP
print("Starting experiment...")
res = cl_strategy.train(scenario.train_stream)

results = []
print("Computing accuracy on the whole test set")
results.append(cl_strategy.eval(scenario.test_stream))
