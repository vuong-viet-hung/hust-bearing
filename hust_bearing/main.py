from lightning.pytorch.cli import LightningCLI

from hust_bearing.data import SpectrogramDM
from hust_bearing.models import Classifier


def cli_main():
    LightningCLI(Classifier, SpectrogramDM)


if __name__ == "__main__":
    cli_main()
