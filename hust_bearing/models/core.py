from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, TypeVar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm


class Metric(ABC):
    @torch.no_grad()
    @abstractmethod
    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass


class Loss(Metric):
    def __init__(self, loss_func: nn.Module) -> None:
        self._loss_func = loss_func
        self._num_samples: int = 0
        self._loss_sum: float = 0.0

    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        batch_size = output_batch.shape[0]
        self._num_samples += batch_size
        self._loss_sum += (
            self._loss_func(output_batch, target_batch).item() * batch_size
        )

    def reset(self) -> None:
        self._num_samples = 0
        self._loss_sum = 0.0

    def compute(self) -> float:
        return self._loss_sum / self._num_samples


class Accuracy(Metric):
    def __init__(self) -> None:
        self._num_samples: int = 0
        self._num_accurate_samples: int = 0

    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        batch_size = output_batch.shape[0]
        prediction_batch = output_batch.argmax(dim=1)
        self._num_samples += batch_size
        self._num_accurate_samples += int(
            (prediction_batch == target_batch).sum().item()
        )

    def reset(self) -> None:
        self._num_samples = 0
        self._num_accurate_samples = 0

    def compute(self) -> float:
        return self._num_accurate_samples / self._num_samples


class Engine:
    def __init__(self, model: nn.Module, device: str, model_file: Path) -> None:
        self._model = model.to(device)
        self._device = device
        self._model_file: Path = model_file
        self._loss_func: nn.Module | None = None
        self._optimizer: optim.Optimizer | None = None
        self._lr_scheduler: optim.lr_scheduler.LRScheduler | None = None
        self._loss: Loss | None = None
        self._accuracy: Accuracy = Accuracy()
        self._min_loss: float = float("inf")

    def train(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        num_epochs: int,
        loss_func: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self._loss_func = loss_func
        self._loss = Loss(loss_func)
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        for _ in trange(num_epochs):
            self._train_one_epoch(train_dl)
            self._valid_one_epoch(valid_dl)

    def _train_one_epoch(self, train_dl: DataLoader) -> None:
        if self._loss is None or self._optimizer is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        prog_bar = tqdm(train_dl)
        self._model.train()
        lr = self._optimizer.param_groups[0]["lr"]
        for input_batch, target_batch in prog_bar:
            self._train_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"train: loss={self._loss.compute():.4f}, acc={self._accuracy.compute():.4f}"
                f" | lr={lr:.2e}"
            )
        if self._min_loss > self._loss.compute():
            self._min_loss = self._loss.compute()
            torch.save(self._model.state_dict(), self._model_file)
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self._loss.reset()
        self._accuracy.reset()

    def _train_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self._loss_func is None or self._loss is None or self._optimizer is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        input_batch = input_batch.to(self._device)
        target_batch = target_batch.to(self._device)
        output_batch = self._model(input_batch)
        loss = self._loss_func(output_batch, target_batch)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._loss.update(output_batch, target_batch)
        self._accuracy.update(output_batch, target_batch)

    @torch.no_grad()
    def _valid_one_epoch(self, valid_dl: DataLoader) -> None:
        if self._loss is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        prog_bar = tqdm(valid_dl)
        self._model.eval()
        for input_batch, target_batch in prog_bar:
            self._eval_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"valid: loss={self._loss.compute():.4f}, acc={self._accuracy.compute():.4f}"
            )
        self._loss.reset()
        self._accuracy.reset()

    def _eval_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self._loss is None:
            raise ValueError("Loss or optimizer isn't initialized.")
        input_batch = input_batch.to(self._device)
        target_batch = target_batch.to(self._device)
        output_batch = self._model(input_batch)
        self._loss.update(output_batch, target_batch)
        self._accuracy.update(output_batch, target_batch)

    def test(self, test_dl: DataLoader, loss_func: nn.Module) -> None:
        self._loss_func = loss_func
        self._loss = Loss(loss_func)
        self._model.load_state_dict(torch.load(self._model_file))
        prog_bar = tqdm(test_dl)
        self._model.eval()
        for input_batch, target_batch in prog_bar:
            self._eval_one_step(input_batch, target_batch)
            prog_bar.set_description(
                f"test: loss={self._loss.compute():.4f}, acc={self._accuracy.compute():.4f}"
            )
        self._loss.reset()
        self._accuracy.reset()

    @torch.no_grad()
    def predict(self, predict_dl: DataLoader) -> torch.Tensor:
        self._model.load_state_dict(torch.load(self._model_file))
        self._model.eval()
        return torch.cat(
            [self._predict_one_batch(input_batch) for input_batch, *_ in predict_dl]
        )

    def _predict_one_batch(self, input_batch) -> torch.Tensor:
        return self._model(input_batch.to(self._device)).argmax(dim=1)


M = TypeVar("M", bound=nn.Module)
_model_registry: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[type[M]], type[M]]:
    def decorator(model_cls: type[M]) -> type[M]:
        _model_registry[name] = model_cls
        return model_cls

    return decorator


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in _model_registry:
        raise ValueError(f"Unregistered model: {name}")
    model_cls = _model_registry[name]
    return model_cls(num_classes)
