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

## License
We give credit to all the other papers and projects that we have referenced in order to write the paper. This paper is covered by the MIT License, granting full access to using this repository for any use whatsoever. 