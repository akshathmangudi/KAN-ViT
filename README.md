# A Detailed Investgation into KAN-based Vision Transformers. 

This is the official repository containing all details and code related to the project currently in progress, which is benchmarking KAN-based Vision Transformers versus the vanilla Vision Transformers from this [paper](https://arxiv.org/abs/2010.11929).

Control variables:
- **Dataset used: MNIST**
- **Optimizer: Adam, learning_rate = 0.005.**
- **Transformations: None**
- **GPU Used: Tesla P100**

## Benchmark #1: Vanilla Testing
Time Taken / Epoch: 
- Vanilla ViT: ~3.8 minutes. 
- KAN-ViT: ~29 minutes. (7.6x longer)

| Model | Date of Training | #Epochs | Test Loss | Test Accuracy |
|:-----:|:----------------:|:-------:|:---------:|:-------------:|
| Vanilla ViT | 27-07-24 | 5 | 1.95 | 54.49% |
| KAN-ViT | 03-08-24 | 5 | 1.85 | 61.41% | 
| |
| Vanilla ViT | 26-07-24 | 8 | 1.94 | 52.07% |
| KAN-ViT | 26-07-24 | 8 | 1.67 | 79.44% |
| |
| Vanilla ViT | 27-07-24 | 10 | 1.90 | 56.14% | 
| KAN-ViT | 27-07-24 | 10 | 1.76 | 70.02% |
||

## Benchmark #2: Using Flash Attention
The flash attention used for KAN_ViT, which contains MLP layers were replaced with KANLinear layers.  

Time Taken / Epoch: 
- Flash-ViT: ~2.1 minutes
- Flash-KAN_ViT: ~2.7 minutes

| Model | Date of Training | #Epochs | Test Loss | Test Accuracy |
|:-----:|:----------------:|:-------:|:---------:|:-------------:|
| Vanilla ViT | 27-07-24 | 5 | 2.36 | 10.28% |
| KAN-ViT | 27-07-24 | 5 | 1.79 | 76.00% | 
| |
| Vanilla ViT | TBD | 8 | TBD | TBD |
| KAN-ViT | 28-07-24 | 8 | 1.68 | 78.34% | 
| |
| Vanilla ViT | TBD | 10 | TBD | TBD |
| KAN-ViT | 28-07-24 | 10 | 1.62 | 84.50% | 
||

## Benchmark #3: KAN Variants
We will be using KAN variants in our Vision Transformers architecture and benchmark them against the test loss, test
accuracy and time taken per epoch. 

Control variables: 
- **Number of Epochs: 10**
- **Dataset used: MNIST**
- **Transformations: None**

**Note: All variant names listed below are accurate with respect to each adaptation listed in the references/works cited.**
| Variant | Time Taken / Epoch | Test Loss | Test Accuracy |
|:-----:|:------------------:|:---------:|:-------------:|
| ChebyKAN | | |
| FourierKAN | | |
| FusedFourierKAN | | |
| efficient-kan | ~29 minutes | 1.62 | 84.50% |
| fast-kan | ~12 minutes | 1.66 | 80.14% | 
| faster-kan | | |
| pykan | | | 
| RBF-KAN | | | 
| Wav-KAN | | |

