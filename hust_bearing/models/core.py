from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm


class Metric(ABC):
    @torch.no_grad()
    @abstractmethod
    def update(self, output_target: torch.Tensor, target_batch: torch.Tensor) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass


class Loss(Metric):
    def __init__(self, loss_func: nn.Module) -> None:
        self.loss_func = loss_func
        self.num_samples: int = 0
        self.loss_sum: float = 0.0

    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        batch_size = output_batch.shape[0]
        self.num_samples += batch_size
        self.loss_sum += self.loss_func(output_batch, target_batch).item() * batch_size

    def reset(self) -> None:
        self.num_samples = 0
        self.loss_sum = 0.0

    def compute(self) -> float:
        return self.loss_sum / self.num_samples


class Accuracy(Metric):
    def __init__(self) -> None:
        self.num_samples: int = 0
        self.num_accurate_samples: int = 0

    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        batch_size = output_batch.shape[0]
        prediction_batch = output_batch.argmax(dim=1)
        self.num_samples += batch_size
        self.num_accurate_samples += int(
            (prediction_batch == target_batch).sum().item()
        )

    def reset(self) -> None:
        self.num_samples = 0
        self.num_accurate_samples = 0

    def compute(self) -> float:
        return self.num_accurate_samples / self.num_samples


class BestModelSaver:
    def __init__(self) -> None:
        self.min_loss: float = float("inf")

    def save(self, model: nn.Module, loss: float, saved_model: Path) -> None:
        if self.min_loss <= loss:
            return
        self.min_loss = loss
        torch.save(model.state_dict(), saved_model)


class Engine:
    def __init__(self, model: nn.Module, device: str, saved_model: Path) -> None:
        self.model = model.to(device)
        self.device = device
        self.saved_model: Path = saved_model
        self.loss_func: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lr_scheduler: optim.lr_scheduler.LRScheduler | None = None
        self.loss: Loss | None = None
        self.accuracy: Accuracy = Accuracy()
        self.best_model_saver = BestModelSaver()

    def train(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        num_epochs: int,
        loss_func: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.loss_func = loss_func
        self.loss = Loss(loss_func)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        for _ in trange(num_epochs):
            self.train_one_epoch(train_dl)
            self.valid_one_step(valid_dl)

    def train_one_epoch(self, train_dl: DataLoader) -> None:
        if self.loss is None or self.optimizer is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        prog_bar = tqdm(train_dl)
        self.model.train()
        lr = self.optimizer.param_groups[0]["lr"]
        for input_batch, target_batch in prog_bar:
            self.train_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"train: loss={self.loss.compute():.4f}, acc={self.accuracy.compute():.4f}"
                f" | lr={lr:.2e}"
            )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.best_model_saver.save(self.model, self.loss.compute(), self.saved_model)
        self.loss.reset()
        self.accuracy.reset()

    def train_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self.loss_func is None or self.loss is None or self.optimizer is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.model(input_batch)
        loss = self.loss_func(output_batch, target_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss.update(output_batch, target_batch)
        self.accuracy.update(output_batch, target_batch)

    @torch.no_grad()
    def valid_one_step(self, valid_dl: DataLoader) -> None:
        if self.loss is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        prog_bar = tqdm(valid_dl)
        self.model.eval()
        for input_batch, target_batch in prog_bar:
            self.eval_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"valid: loss={self.loss.compute():.4f}, acc={self.accuracy.compute():.4f}"
            )
        self.loss.reset()
        self.accuracy.reset()

    def eval_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self.loss is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.model(input_batch)
        self.loss.update(output_batch, target_batch)
        self.accuracy.update(output_batch, target_batch)

    def test(self, test_dl: DataLoader, loss_func: nn.Module) -> None:
        self.loss_func = loss_func
        self.loss = Loss(loss_func)
        self.model.load_state_dict(torch.load(self.saved_model))
        prog_bar = tqdm(test_dl)
        self.model.eval()
        for input_batch, target_batch in prog_bar:
            self.eval_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"test: loss={self.loss.compute():.4f}, acc={self.accuracy.compute():.4f}"
            )
        self.loss.reset()
        self.accuracy.reset()

    @torch.no_grad()
    def predict(self, predict_dl: DataLoader) -> torch.Tensor:
        self.model.load_state_dict(torch.load(self.saved_model))
        self.model.eval()
        return torch.cat(
            [self.predict_one_batch(input_batch) for input_batch, *_ in predict_dl]
        )

    def predict_one_batch(self, input_batch) -> torch.Tensor:
        return self.model(input_batch.to(self.device)).argmax(dim=1)
