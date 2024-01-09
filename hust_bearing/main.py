from lightning.pytorch.cli import LightningCLI

from hust_bearing.data import HUSTParser
from hust_bearing.models import LeNet5, ConvMixer


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
