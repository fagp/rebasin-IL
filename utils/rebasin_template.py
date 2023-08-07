from typing import Sequence, Optional

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.models import avalanche_forward
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

from rebasin import RebasinNet
from functional import ILLambda05
from copy import deepcopy


class RebasinILPlugin(SupervisedTemplate):
    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        il_criterion=None,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        il_epochs: int = 10,
        il_alpha: float = 0.8,
        il_lr: float = 0.01,
        residual_lr: float = 0.05,
        residual_grad_clip: float = 0.001,
        residual_weight_decay: float = 1e-2,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator=default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        self.base_criterion = criterion
        self.il_criterion = il_criterion

        self.current_model = None

        self.il_epochs = il_epochs
        self.il_lambda = il_alpha
        self.il_lr = il_lr
        self.residual_lr = residual_lr
        self.residual_grad_clip = residual_grad_clip
        self.residual_weight_decay = residual_weight_decay

    def criterion(self):
        if self.clock.train_exp_counter == 0 or not self.is_training:
            return self.base_criterion(self.mb_output, self.mb_y)

        return self.il_criterion(self.model(), self.mb_x, self.mb_y, self.mb_task_id)

    def forward(self):
        if self.is_training:
            return avalanche_forward(self.model, self.mb_x, self.mb_y)

        return avalanche_forward(self._model_eval, self.mb_x, self.mb_y)

    def _after_backward(self, **kwargs):
        super()._after_backward(**kwargs)
        # update the residual in incremental learning mode
        if self.clock.train_exp_counter > 0:
            self.il_criterion.optim_step()

    def _before_eval_exp(self, **kwargs):
        model = self.model
        # evaluation model is set using Eq. 12
        if isinstance(model, RebasinNet):
            model.eval()
            self._model_eval = deepcopy(self.current_model)
            for p1, p2, p3 in zip(
                self._model_eval.parameters(),
                model().parameters(),
                self.il_criterion.delta.parameters(),
            ):
                p1.data.copy_(
                    p1.data * self.il_lambda + p2.data * (1 - self.il_lambda) + p3.data
                )
        # or use the first model in the first epoch
        else:
            self._model_eval = deepcopy(self.model)

        self._before_eval_dataset_adaptation(**kwargs)
        self.eval_dataset_adaptation(**kwargs)
        self._after_eval_dataset_adaptation(**kwargs)
        super()._before_eval_exp(**kwargs)

    def model_adaptation(self, model=None):
        if model is None:
            model = self.model

        if self.is_training and self.clock.train_exp_counter > 0:
            # adapts the model following Eq. 12
            if isinstance(model, RebasinNet):
                self.current_model = self.adapt_model()

            # the first model is not adapted
            else:
                self.current_model = deepcopy(model)

            # create re-basin network
            model = RebasinNet(self.current_model, tau=1, n_iter=20)
            model.to(self.device)
            model.train()

        return model.to(self.device)

    def adapt_model(self):
        # adapts the model following Eq. 12
        model = self.model
        model.eval()
        for p1, p2, p3 in zip(
            self.current_model.parameters(),
            model().parameters(),
            self.il_criterion.delta.parameters(),
        ):
            p1.data.copy_(
                (p1.data * self.il_lambda + p2.data * (1 - self.il_lambda) + p3.data)
            )

        return self.current_model

    def make_optimizer(self):
        # first task is trained normally
        if self.clock.train_exp_counter == 0:
            reset_optimizer(self.optimizer, self.model)

        else:
            # set the new number of epochs for continual learning
            self.train_epochs = self.il_epochs
            # set the learning rate for continual learning
            for p in self.optimizer.param_groups:
                p["lr"] = self.il_lr

            # reset optimizer
            reset_optimizer(self.optimizer, self.model)

            # create a new instance of the criterion
            self.il_criterion = ILLambda05(
                self.base_criterion,
                modela=self.current_model,
                gamma=self.residual_lr,
                beta=self.residual_weight_decay,
                gradient_clip=self.residual_grad_clip,
            )
