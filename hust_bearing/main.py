from lightning.pytorch.cli import LightningCLI

from hust_bearing.models import ConvMixer
from hust_bearing.data import HUST


def cli_main():
    LightningCLI()


if __name__ == "__main__":
    cli_main()
