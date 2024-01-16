from torch import nn


class LMMDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
