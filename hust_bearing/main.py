from lightning.pytorch.cli import LightningCLI

from hust_bearing.models import classifier
from hust_bearing.data import bearing_data_module


def cli_main():
    LightningCLI(classifier, bearing_data_module)


if __name__ == "__main__":
    cli_main()
