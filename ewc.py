import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import RotatedMNIST
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import EWC
import numpy as np
import torch
import os

n_experiences = 20
number_classes = 10
mb_size = 10
total_epochs = 5
ewc_lambda = 100
lr = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scenario = RotatedMNIST(
    n_experiences, rotations_list=np.linspace(0, 180, n_experiences).tolist()
)

os.makedirs("./logs", exist_ok=True)
name = "./logs/ewc.txt"

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
cl_strategy = EWC(
    model,
    SGD(model.parameters(), lr=lr),
    CrossEntropyLoss(),
    ewc_lambda,
    "separate",
    decay_factor=None,
    train_mb_size=mb_size,
    train_epochs=total_epochs,
    eval_mb_size=100,
    evaluator=eval_plugin,
    device=device,
)

# TRAINING LOOP
print("Starting experiment...")
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print("Training completed")

    print("Computing accuracy on the whole test set")
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))
