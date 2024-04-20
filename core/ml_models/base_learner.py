""" Base learner class."""

import torch


class BaseLearner:
    """Base learner class.

    Parameters
    ----------
    model: pytorch model
        Pytorch model to be trained.
    adam_params: dict, optional
        Dictionary of Adam parameters.
    lr_schedule: dict, optional
        Dictionary of learning rate scheduler parameters.

    """

    def __init__(
        self,
        model,
        adam_params=None,
        lr_schedule=None,
    ):
        self.model = model

        # set up optimizer
        if adam_params is None:
            # use default
            adam_params = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_params)

        # set up learning-rate scheduler
        self.lr_decay = lr_schedule is not None
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **lr_schedule
            )

    def lr_scheduler_step(self, metric):
        """Update learning rate."""
        if self.lr_decay:
            self.scheduler.step(metric)

    def train_step(self, batch, batch_tensor=True):
        """Wrapper function for performing a training step."""

        self.model.train()

        self.optimizer.zero_grad()
        if batch_tensor:
            x, y = batch
            loss, predictions = self.forward_pass(x, y)
        else:
            loss = 0.0
            predictions = []
            num_batch_samples = 0
            for sample in batch:
                num_batch_samples += 1
                x, y = sample
                (
                    sample_loss,
                    sample_predictions,
                ) = self.forward_pass(x, y)
                loss += sample_loss
                predictions.append(sample_predictions)
            loss /= float(num_batch_samples)

        loss.backward()
        self.optimizer.step()

        return loss, predictions

    def forward_pass(self, inputs, labels, *args, **kwargs):
        return None, None
