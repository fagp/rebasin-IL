import torch
from torch.optim import AdamW
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
from utils import RebasinILPlugin
import numpy as np
import os
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import (
    ExperienceBalancedBuffer,
)

n_experiences = 20
memory_per_task_per_class = 5
number_classes = 10

mb_size = 500
total_epochs = 5
lr = 0.001
eta = 0.1
gamma = 0.05
beta = 0.1
alpha = 0.8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scenario = RotatedMNIST(
    n_experiences, rotations_list=np.linspace(0, 180, n_experiences).tolist()
)

replay = ReplayPlugin(
    mem_size=memory_per_task_per_class * number_classes * n_experiences,
    storage_policy=ExperienceBalancedBuffer(
        max_size=memory_per_task_per_class * number_classes * n_experiences,
        adaptive_size=False,
        num_experiences=n_experiences,
    ),
)

os.makedirs("./logs", exist_ok=True)
name = "./logs/rebasinIL.txt"

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

cl_strategy = RebasinILPlugin(
    model,
    AdamW(model.parameters(), lr=lr),
    CrossEntropyLoss(),
    train_mb_size=mb_size,
    train_epochs=total_epochs,
    il_epochs=total_epochs,
    il_alpha=alpha,
    il_lr=eta,
    residual_lr=gamma,
    residual_weight_decay=beta,
    eval_mb_size=mb_size,
    evaluator=eval_plugin,
    device=device,
    plugins=[replay],
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
