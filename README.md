# Fault Classification of CWRU Bearing Dataset
## Introduction
Classification of bearing faults from accelerometer's signal using data from CWRU Bearing Dataset. Faults are classified into one of four classes: **Normal**, **B** (Ball), **IR** (Inner-Race), **OR** (Outer-Race). Classification is done for individual subsets: **12k_DE**, **12k_FE** and **48k_DE**.
## Prerequisites
- Linux operating system
- CUDA-capable GPU
- ```conda``` package manager
- Dependencies listed in ```environment.yml```
## Create and activate the environment
This project is managed using ```conda```. To have it installed, follow the [this](https://docs.conda.io/projects/miniconda/en/latest/) instruction. \
Create the ```bearing``` project environment and and install dependencies.
```
conda env create -f environment.yml
```
After ```bearing``` environment is created, activate the environment.
```
conda activate bearing
```
The environment must be activated before any ```python``` commands are executed. \
Deactivate the environment upon completion.
```
conda deactivate
```
## Download and preprocess the dataset
Download the dataset from [this](https://github.com/XiongMeijing/CWRU-1) repository and extract the downloaded zip file. Save the dataset as *data/* directory.

```
python download_data.py
```
Divide the download signal data into samples, assign samples into train, valid and test sets.
```
python preprocess_data.py
```
The command creates *csv/* directory which contains signal samples' metadata. For example, a csv file for valid samples belonging to the **12k_FE** subset is named as such: *12k_FE_val.csv*.
## Train the model
Train and save trained model.
```
python train.py
```
## Evaluation and inference (Work in progress)
Download pretrained weights to the *weights/* directory.
```
python download_weights.py
```
TODO: Create evaluation and inference scripts.
