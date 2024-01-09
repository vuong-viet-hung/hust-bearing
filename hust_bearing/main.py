from pathlib import Path

from lightning.pytorch.cli import LightningCLI
from torch import nn

from hust_bearing.models import Classifier, LeNet5, ConvMixer
from hust_bearing.data import SpectrogramDM, Parser, HUSTParser


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "lenet5": LeNet5,
    "conv_mixer": ConvMixer,
}


PARSER_CLASSES: dict[str, type[Parser]] = {
    "hust": HUSTParser,
}


def cli_main():
    LightningCLI(create_dm, create_clf)  # type: ignore


def create_clf(name: str, num_classes: int) -> Classifier:
    model = MODEL_CLASSES[name](num_classes)
    return Classifier(model, num_classes)


def create_dm(
    name: str, data_dir: Path | str, train_load: str, batch_size: int
) -> SpectrogramDM:
    parser = PARSER_CLASSES[name]()
    return SpectrogramDM(data_dir, batch_size, train_load, parser)


if __name__ == "__main__":
    cli_main()
