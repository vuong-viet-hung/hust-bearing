# HUST Bearing (Work in Progress)
<p align="center">
    <img src="assets/mandevices_logo.jpg">
</p>

## Introduction
This repository contains researches dedicated to classification of bearing fault based on vibrational signals' spectrograms, using deep neural network to identify the type of defect.
The project provides an intuitive CLI for proposed algorithms.
We prioritize presenting previous researches in a readable and reproducible manner over introducing a faithful implementation.

The team behind this work is [Mandevices Laboratory](https://www.facebook.com/mandeviceslaboratory) from Hanoi University of Science and Technology (HUST). Learn more about our researches [here](#about-us).

## Prerequisite
- CUDA-enabled GPU
- Python version 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) dependency manager

## Installation

### Pip installation (Coming soon)

### Poetry installation

- [Install Poetry](https://python-poetry.org/docs/)
- Clone the repository
```commandline
git clone https://github.com/vuong-viet-hung/hust-bearing.git
cd hust-bearing
```
- Install project's dependencies
```commandline
poetry install
```

## Usage
Refer to [this guide](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment) on how to execute commands inside virtual environment.

The CLI is powered by LightningCLI. Refer to [this guide](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for advanced usage.

### Data
Obtain the dataset before advance to the proceeding steps

Open-access data will be provided soon. As of now, there are options to:
- [Contact us](mailto:vuongviethung156@gmail.com) for the dataset.
- Generate the dataset using the [accompanied repository](https://github.com/vuong-viet-hung/BearingSpectrogram.git).

The dataset should be structured as such:
```
data          <-- root directory
|---hust      <-- dataset directory
|   |---B500  <-- directory containing spectrograms
|   |---B502
|   |   ...
|   |---O504
|---cwru
    |   ...    
```

## About Us