# A Detailed Investigation into KAN-based Vision Transformers. 

This is the official repository containing all details and code related to the project currently in progress, which is benchmarking KAN-based Vision Transformers versus the vanilla Vision Transformers from this [paper](https://arxiv.org/abs/2010.11929).

## Installation

### Step 1: Setup a virtualenv / conda environment. 
For conda: 
```bash
$ conda create -n <env_name> 
```

For virtualenv:
```bash
$ python -m venv <env_name>
```
where `env_name` will be the name of the directory that you will use as the virtual environment. 

### Step 2: Activate the environment. 
For conda: 
```bash
$ conda install --file requirements.txt
```

For virualenv: 
```bash
$ pip install -r requirements.txt
```

### Step 3: Run train.py 
```bash
$ python train.py
```

To follow these steps correctly, make sure you are in the root directory of the repository. 


Control variables:
- **Dataset used: MNIST**
- **Transformations: None**
- **GPU Used: Tesla P100**

## Benchmark #1: Vanilla Testing
Time Taken / Epoch: 
- Vanilla ViT: ~ 4 minutes.
- KAN-ViT: ~ 29 minutes. (7.6x longer)

| Model | Date of Training | #Epochs | Test Accuracy | Balanced Test Accuracy | F1 Score | ROC/AUC Score |
|:-----:|:----------------:|:-------:|:-------------:|:----------------------:|:--------:|:-------------:|
| Vanilla ViT | 27-07-2024 | 5 | 52.48% | 51.42% | 0.497 | 0.815 |
| KAN-ViT | 03-08-2024 | 5 | 61.24% | 60.42% | 0.612 | 0.879 |
| |
| Vanilla ViT | 26-07-2024 | 8 | 49.44% | 48.45% | 0.465 | 0.809 |
| KAN-ViT | 26-07-2024 | 8 | 79.44% | 78.97% | 0.792 | 0.935 |
| |
| Vanilla ViT | 27-07-2024 | 10 | 56.78% | 55.64% | 0.543 | 0.830 | 
| KAN-ViT | 27-07-2024 | 10 | 82.14% | 82.01% | 0.818 | 0.950 |
||

## Benchmark #2: Using Flash Attention
The flash attention used for KAN_ViT, which contains MLP layers were replaced with KANLinear layers, taken from the efficient-kan repository.

Time Taken / Epoch: 
- Flash-ViT: ~ 2 minutes.
- Flash-KAN_ViT: ~ 3 minutes.

| Model | Date of Training | #Epochs | Test Accuracy | Balanced Test Accuracy | F1 Score | ROC/AUC Score |
|:-----:|:----------------:|:-------:|:-------------:|:----------------------:|:--------:|:-------------:|
| Vanilla ViT | 06-11-2024 | 5 | 64.35% | 63.38% | 0.609 | 0.886 |
| KAN-ViT | 07-11-2024 | 5 | 76.51% | 75.57% | 0.734 | 0.936 | 
| |
| Vanilla ViT | 06-11-2024 | 8 | 78.42% | 78.05% | 0.782 | 0.945 |
| KAN-ViT | 07-11-2024 | 8 | 74.61% | 73.60% | 0.715 | 0.930 | 
| |
| Vanilla ViT | 06-11-2024 | 10 | 74.83% | 74.37% | 0.747 | 0.928 |
| KAN-ViT | 07-11-2024 | 10 | 69.04% | 67.96% | 0.665 | 0.899 |
||

## Benchmark #3: KAN Variants
We will be using KAN variants in our Vision Transformers architecture and benchmark them against the test loss, test
accuracy and time taken per epoch. 

Control variables: 
- **Number of Epochs: 10**
- **Dataset used: MNIST**
- **Transformations: None**
- **Learning Rate: 0.003 - 0.005** 
NOTE: A handful of learning rates were picked out of the range and the result was added below based on performance

**Note: All variant names listed below are accurate with respect to each adaptation listed in the references/works cited.**
| Variant | Date of Training | Time Taken / Epoch | Test Accuracy | Balanced Test Accuracy | F1 Score | ROC/AUC Score |
|:-------:|:----------------:|:-------:|:-------------:|:----------------------:|:--------:|:-------------:|
| ChebyKAN | 28-10-2024 | ~ 11 minutes | 76.14% | 75.95% | 0.759 | 0.952 |
| FourierKAN | 29-10-2024 | ~ 5 minutes | 53.53% | 53.08% | 0.527 | 0.832 |
| efficient-kan | 27-10-2024 | ~ 30 minutes | 75.01% | 74.46% | 0.753 | 0.930 |
| fast-kan | 27-10-2024 | ~ 12 minutes | 80.14% | 79.12% | 0.805 | 0.934 |
| SineKAN | 28-10-2024 | ~ 11 minutes | 62.38% | 61.84% | 0.618 | 0.889 |
| |

## License
We give credit to all the other papers and projects that we have referenced in order to write the paper. This paper is covered by the MIT License, granting full access to using this repository for any use whatsoever. 