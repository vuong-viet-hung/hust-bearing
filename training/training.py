import torch


def train(
    model: torch.nn.Module,
    train_dl: torch.utils.data.Dataloader,
    val_dl: torch.utils.data.Dataloader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
) -> None:
    pass
