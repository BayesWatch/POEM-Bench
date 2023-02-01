# GATE - Generalization After Transfer Evaluation Benchmark Engine
Welcome to GATE - A benchmark framework built to evaluate a learning process on its ability to learn and generalize on previously unseen Tasks, Data domains and Modalities. GATE ensures fair benchmarking by enforcing separation between learners, models, tasks and data, enabling multiple approaches (learners) to be fairly compared against each other by using identical network backbones (models) for all learners on well defined tasks and datasets.

This repository utilises E-GATE to benchmark Partial Observation Experts Modelling (POEM) at self-supervised few-shot learning from partial observations, as introduced and described in the paper [Contrastive Meta-Learning for Partially Observable Few-Shot Learning](https://arxiv.org/abs/2301.13136) (accepted for publication at ICLR 2023).

## Installation
Create a new environment and run `install_env_nvidia_gpu.sh` or `install_dependencies.sh` for a GPU or CPU machine respectively, and optionally `install_dev_tools.sh` to install addional development tools. Create a `.env` file from `.env.template` by adding your personal environment variables.

## Results

The final results of the GATE evaluation on Partially Observable Meta-Dataset (as defined in the paper Contrastive Meta-Learning for Partially Observable Few-Shot Learning) achieved using the learners in this repository (with ResNet-18 backbones) are:

| **Test Source** | **Finetune**   | **ProtoNet** | **MAML**   | **POEM**       |
|:---------------:|:--------------:|:------------:|:----------:|:--------------:|
| **Aircraft**    | 46.5+/-0.6     | 48.5+/-1.0   | 37.5+/-0.3 | **55.3+/-0.7** |
| **Birds**       | 62.6+/-0.7     | 67.4+/-1.2   | 52.5+/-0.6 | **71.1+/-0.1** |
| **Flowers**     | 48.5+/-0.4     | 46.4+/-0.7   | 33.5+/-0.3 | **49.2+/-1.5** |
| **Fungi**       | 61.0+/-0.2     | 61.4+/-0.4   | 46.1+/-0.4 | **64.8+/-0.3** |
| **Omniglot**    | 71.3+/-0.1     | 87.8+/-0.1   | 47.4+/-1.0 | **89.2+/-0.7** |
| **Textures**    | **83.2+/-0.4** | 76.7+/-1.6   | 73.1+/-0.4 | 81.4+/-0.6     |


Ablating to the standard Meta-Dataset evaluation procedure (with fully observed images for standard few-shot learning) as defined in the [original Meta-Dataset paper](https://arxiv.org/abs/1903.03096) the results final results achieved using the learners in this repository (with ResNet-18 backbones) are:

| **Test Source** | **Finetune** | **ProtoNet** | **MAML**   | **POEM**   |
|:---------------:|:------------:|:------------:|:----------:|:----------:|
| **Aircraft**    | 56.2+/-1.1   | 47.2+/-1.2   | 35.9+/-1.8 | 46.5+/-1.5 |
| **Birds**       | 52.6+/-1.8   | 78.3+/-0.5   | 65.2+/-0.3 | 79.4+/-0.3 |
| **Flowers**     | 80.1+/-2.0   | 84.2+/-0.7   | 70.4+/-0.4 | 83.6+/-1.3 |
| **Fungi**       | 33.6+/-1.7   | 84.7+/-0.2   | 18.9+/-0.2 | 81.0+/-0.1 |
| **Omniglot**    | 89.6+/-3.3   | 98.7+/-0.1   | 94.7+/-0.1 | 98.6+/-0.1 |
| **Textures**    | 60.4+/-1.0   | 65.3+/-1.2   | 56.1+/-0.3 | 65.7+/-0.8 |

This demonstrates that POEM provides additional benefits for unifying representations when dealing with partial observability, while remaining competitive at standard few-shot learning.
