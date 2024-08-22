# Diffence: Fencing Membership Privacy With Diffusion Models

This repository contains the code for the paper "Diffence: Fencing Membership Privacy With Diffusion Models," currently under review for [NDSS 2025](https://www.ndss-symposium.org/ndss2025/).

Diffence is a robust plug-and-play defense mechanism designed to enhance the membership privacy of both undefended models and models trained with state-of-the-art defenses without compromising model utility.

## Table of Contents

- [Installation](#installation)
- [Experiment Workflow](#experiment-workflow)
  - [Preparation](#preparation)
  - [Execution](#execution)
  - [Results](#results)
- [Evaluation](#evaluation)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bujuef/Diffence.git
    cd Diffence
    ```

2. Create a conda environment and install dependencies:

    ```bash
    conda env create -f environment.yaml
    conda activate diffence-env
    ```

3. If you do not have conda installed, follow the instructions on their [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Experiment Workflow

### Preparation

1. Navigate to the folder of the dataset to be tested, e.g., CIFAR-10:

    ```bash
    cd cifar10
    ```

2. Download and partition the dataset:

    ```bash
    python data_partition.py
    ```

3. Obtain the diffusion model used for Diffence:

    We provide our pretrained diffusion model checkpoints [here](https://drive.google.com/drive/folders/144wk3X0lKnRPNK5KM4FJxvjGJH-_6Xlo?usp=sharing). Copy the `diff_models` to the corresponding `diff_defense` folder, e.g., `cifar10/diff_defense/diff_models`.

    (Optional) Train the diffusion model using [this repository](https://github.com/w86763777/pytorch-ddpm).

### Execution

1. Obtain the undefended model and models with existing defenses:

    Our pretrained models are available [here](https://drive.google.com/drive/folders/1KNn7ycsjSaQ-wpfpz-UgY2cPuvrdEMLL?usp=sharing). Copy them to the `final-all-models` folder, e.g., `cifar10/final-all-models/resnet/selena.pth.tar`.

    (Optional) You can retrain specific defended models using the commands listed in `all-train-all.sh`.

2. Test model accuracy and membership privacy:

    ```bash
    cd evaluate_MIAs  # Navigate to the test script folder
    bash evaluate_mia.sh --defense [defense name]  # defense name in {undefended, selena, advreg, hamp, relaxloss}
    ```

After completion, the results of the above experiments will be saved in the `./results` folder.

### Results

The results will be saved in `Diffence/[dataset_name]/evaluate_MIAs/results`. For example, `selena` and `selena_w_diffence` correspond to the results of using SELENA defense alone and deploying Diffence on top of it, respectively.


