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

| Model | Date of Training | #Epochs | Test Accuracy |
|:-----:|:----------------:|:-------:|:-------------:|
| Vanilla ViT | 27-07-2024 | 5 | 54.49% |
| KAN-ViT | 03-08-2024 | 5 | 61.41% | 
| |
| Vanilla ViT | 26-07-2024 | 8 | 52.07% |
| KAN-ViT | 26-07-2024 | 8 | 79.44% |
| |
| Vanilla ViT | 27-07-2024 | 10 | 56.14% | 
| KAN-ViT | 27-07-2024 | 10 | 82.14% | 
||

## Benchmark #2: Using Flash Attention
The flash attention used for KAN_ViT, which contains MLP layers were replaced with KANLinear layers, taken from the efficient-kan repository.

Time Taken / Epoch: 
- Flash-ViT: ~ 2 minutes.
- Flash-KAN_ViT: ~ 3 minutes.

| Model | Date of Training | #Epochs | Test Accuracy |
|:-----:|:----------------:|:-------:|:-------------:|
| Vanilla ViT | 16-08-2024 | 5 | 64.35% |
| KAN-ViT | 17-08-2024 | 5 | 49.65% | 
| |
| Vanilla ViT | 16-08-2024 | 8 | 72.22% |
| KAN-ViT | 17-08-2024 | 8 | 61.63% | 
| |
| Vanilla ViT | 16-08-2024 | 10 | 74.83% |
| KAN-ViT | 17-08-2024 | 10 | 69.01% |
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
| Variant | Date of Training | Time Taken / Epoch | Test Accuracy |
|:-----:|:------------------:|:------------------:|:-------------:|
| ChebyKAN | 28-10-2024 | ~ 11 minutes | 76.14% |
| FourierKAN | 29-10-2024 | ~ 5 minutes | 60.10% |
| efficient-kan | 27-10-2024 | ~ 30 minutes | 84.50% |
| fast-kan | 27-10-2024 | ~ 12 minutes | 80.14% |
| SineKAN | 28-10-2024 | ~ 11 minutes | 62.53% |
| |

## License
We give credit to all the other papers and projects that we have referenced in order to write the paper. This paper is covered by the MIT License, granting full access to using this repository for any use whatsoever. 