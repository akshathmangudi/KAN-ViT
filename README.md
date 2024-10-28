# A Detailed Investigation into KAN-based Vision Transformers. 

This is the official repository containing all details and code related to the project currently in progress, which is benchmarking KAN-based Vision Transformers versus the vanilla Vision Transformers from this [paper](https://arxiv.org/abs/2010.11929).

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
| Vanilla ViT | 27-07-2024 | 10 | 34.1% | 
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
| Vanilla ViT | 16-08-2024 | 74.83% |
| KAN-ViT | 17-08-2024 | 69.01% |
||

## Benchmark #3: KAN Variants
We will be using KAN variants in our Vision Transformers architecture and benchmark them against the test loss, test
accuracy and time taken per epoch. 

Control variables: 
- **Number of Epochs: 10**
- **Dataset used: MNIST**
- **Transformations: None**

**Note: All variant names listed below are accurate with respect to each adaptation listed in the references/works cited.**
| Variant | Date of Training | Time Taken / Epoch | Test Accuracy |
|:-----:|:------------------:|:------------------:|:-------------:|
| ChebyKAN | | | |
| FourierKAN* | | | |
| efficient-kan | 27-10-2024 | ~ 30 minutes | 84.50% |
| fast-kan | 27-10-2024 | ~ 12 minutes | 80.14% |
| SineKAN | | | |
| |

asterisk(*) indicates that the variant is converging too slow and learning rate has to be increased. However, that has not been done for fair benchmarking.

