# A Detailed Investgation into KAN-based Vision Transformers. 

This is the official repository containing all details and code related to the project currently in progress, which is benchmarking KAN-based Vision Transformers versus the vanilla Vision Transformers from this [paper](https://arxiv.org/abs/2010.11929).

## Benchmark #1: 
The only thing changing in these benchmarks is the number of epochs that we are training the model for. 
- **Dataset used: MNIST**
- **Optimizer: Adam, learning_rate = 0.005.**
- **GPU Used: Tesla P100 and Tesla T4**

Time Taken / Epoch: 
- Vanilla ViT: ~3.8 minutes. 
- KAN-ViT: ~29 minutes. (7.6x longer)

| Model | Date of Training | #Epochs | Test Loss | Test Accuracy |
|:-----:|:----------------:|:-------:|:---------:|:-------------:|
| Vanilla ViT | 27-07-24 | 5 | 1.95 | 54.49% |
| KAN-ViT | 27-07-24 | 5 | TBD | TBD | 
| |
| Vanilla ViT | 26-07-24 | 8 | 1.94 | 52.07% |
| KAN-ViT | 26-07-24 | 8 | 1.67 | 79.44% |
| |
| Vanilla ViT | TBD | 10 | TBD | TBD | 
| KAN-ViT | TBD | 10 | TBD | TBD |
||