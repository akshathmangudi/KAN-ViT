# A Detailed Investigation into KAN-based Vision Transformers. 

This is the official repository containing all details and code related to the project currently in progress, which is benchmarking KAN-based Vision Transformers versus the vanilla Vision Transformers from this [paper](https://arxiv.org/abs/2010.11929).

Control variables:
- **Dataset used: CIFAR**
- **Transformations: None**
- **GPU Used: Tesla P100**

## Benchmark #1: Vanilla Testing
Time Taken / Epoch: 
- Vanilla ViT: 5 minutes.
- KAN-ViT: 25 minutes. 

| Model | Date of Training | #Epochs | Test Loss | Test Accuracy |
|:-----:|:----------------:|:-------:|:---------:|:-------------:|
| Vanilla ViT | | | | |
| KAN-ViT | 19-10-2024 | 5 | 2.12 | 33.0% | 
| |
| Vanilla ViT | 19-10-2024 | 8 | 2.15 | 29.6% |
| KAN-ViT | 19-10-2024 | 8 | | |
| |
| Vanilla ViT | 18-10-2024 | 10 | 2.11 | 34.1% | 
| KAN-ViT | 18-10-2024 | 10 | 2.08 | 37.3% | 
||

## Benchmark #2: Using Flash Attention
The flash attention used for KAN_ViT, which contains MLP layers were replaced with KANLinear layers, taken from the efficient-kan repository.

Time Taken / Epoch: 
- Flash-ViT:
- Flash-KAN_ViT: 

| Model | Date of Training | #Epochs | Test Loss | Test Accuracy |
|:-----:|:----------------:|:-------:|:---------:|:-------------:|
| Vanilla ViT | | | | |
| KAN-ViT | | | | | 
| |
| Vanilla ViT | | | | |
| KAN-ViT | | | | | 
| |
| Vanilla ViT | | | | |
| KAN-ViT | | | | | 
||

## Benchmark #3: KAN Variants
We will be using KAN variants in our Vision Transformers architecture and benchmark them against the test loss, test
accuracy and time taken per epoch. 

Control variables: 
- **Number of Epochs: 10**
- **Dataset used: MNIST**
- **Transformations: None**

**Note: All variant names listed below are accurate with respect to each adaptation listed in the references/works cited.**
| Variant | Date of Training | Time Taken / Epoch | Test Loss | Test Accuracy |
|:-----:|:------------------:|:------------------:|:---------:|:-------------:|
| ChebyKAN | | | | | 
| FourierKAN* | | | | |
| FusedFourierKAN | | | | |
| efficient-kan | | | | |
| fast-kan | | | | |
| faster-kan | | | | |
| pykan | | | | |
| RBF-KAN | | | | |
| Wav-KAN | | | | |
| SineKAN | | | | |
| |

asterisk(*) indicates that the variant is converging too slow and learning rate has to be increased. However, that has not been done for fair benchmarking.

