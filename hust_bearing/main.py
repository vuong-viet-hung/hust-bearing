from lightning.pytorch.cli import LightningCLI

from hust_bearing.models import Classifier, ConvMixer
from hust_bearing.data import BearingDataModule, HUST, CWRU


def cli_main():
    LightningCLI(
        Classifier, BearingDataModule, subclass_mode_model=True, subclass_mode_data=True
    )


if __name__ == "__main__":
    cli_main()
