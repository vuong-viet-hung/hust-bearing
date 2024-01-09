from lightning.pytorch.cli import LightningCLI

from hust_bearing.data import create_dm
from hust_bearing.models import create_clf


def cli_main():
    LightningCLI(create_dm, create_clf)  # type: ignore


if __name__ == "__main__":
    cli_main()
