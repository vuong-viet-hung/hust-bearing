from pathlib import Path
from typing import Literal

from lightning.pytorch.cli import LightningCLI
from torch import nn

from hust_bearing.models import Classifier, LeNet5, ConvMixer
from hust_bearing.data import SpectrogramDM, Parser, CWRUParser, HUSTParser


ModelName = Literal["lenet5", "convmixer"]


MODEL_CLASSES: dict[ModelName, type[nn.Module]] = {
    "lenet5": LeNet5,
    "convmixer": ConvMixer,
}


DataName = Literal["cwru", "hust"]


PARSER_CLASSES: dict[DataName, type[Parser]] = {
    "cwru": CWRUParser,
    "hust": HUSTParser,
}


def cli_main():
    LightningCLI(model, data_module)


def model(name: ModelName, num_classes: int) -> Classifier:
    return Classifier(MODEL_CLASSES[name](num_classes), num_classes)


def data_module(
    name: DataName, data_dir: Path, batch_size: int, train_load: str
) -> SpectrogramDM:
    parser = PARSER_CLASSES[name]()
    return SpectrogramDM(parser, data_dir, batch_size, train_load)


if __name__ == "__main__":
    cli_main()
