import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm


class Engine:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        self.loss: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def train(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        num_epochs: int,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        prog_bar = trange(num_epochs)
        self.loss = loss
        self.optimizer = optimizer
        for epoch in prog_bar:
            self.train_one_epoch(train_dl)
            self.eval_one_epoch(valid_dl)

    def train_one_epoch(self, train_dl: DataLoader) -> None:
        prog_bar = tqdm(train_dl)
        self.model.train()
        for image_batch, label_batch in prog_bar:
            self.train_one_step(image_batch, label_batch)
            self.eval_one_epoch(image_batch, label_batch)

    def train_one_step(
        self, image_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> None:
        image_batch = image_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        output_batch = self.model(image_batch)
        loss = self.loss(output_batch, label_batch)
        loss.backward()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def eval_one_epoch(self, valid_dl: DataLoader) -> None:
        prog_bar = tqdm(valid_dl)
        self.model.eval()
        for image_batch, label_batch in prog_bar:
            self.eval_one_step(image_batch, label_batch)

    def eval_one_step(
        self, image_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> None:
        image_batch = image_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        output_batch = self.model(image_batch)
        loss = self.loss(output_batch, label_batch)

    def test(self, test_dl: DataLoader, loss: torch.nn.Module) -> None:
        self.loss = loss
        self.eval_one_epoch(test_dl)

    @torch.no_grad()
    def predict(self, predict_dl: DataLoader) -> torch.Tensor:
        self.model.eval()
        return torch.cat(
            [self.predict_one_batch(image_batch) for image_batch, *_ in predict_dl]
        )

    def predict_one_batch(self, image_batch) -> torch.Tensor:
        image_batch = image_batch.to(self.device)
        return torch.argmax(self.model(image_batch.to), dim=1)
