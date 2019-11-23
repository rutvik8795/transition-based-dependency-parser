This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Credits:

Most of this code has been directly taken from the authors of:

> From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)

We have adapted it to work with tensorflow2.0, and changed it to similar format as assignment 2. Also thanks to previous TA, Heeyoung Kwon who set up the original assignment.

# Overview

You will implement a neural Dependency Parsing model by writing code for the following:

From Incrementality in Deterministic Dependency Parsing (2004, Nivre)
- the arc-standard algorithm

From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)

- feature extraction
- the neural network architecture including activation function
- loss function


# Installation

The environment is same as assignment 2. But we would *strongly* encourage you to make a new environment for assignment 3.

This assignment is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-hw3 python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw3
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download glove wordvectors:
```
./download_glove.sh
```

**NOTE:** We will be using this environment to check your code, so please don't work in your default or any other python environment.


# Data

You have training, development and test set for dependency parsing in conll format. The `train.conll` and `dev.conll` are labeled whereas `test.conll` is unlabeled

For quick code development/debugging, this time we have explicitly provided small fixture dataset. You can use this as training and development dataset while working on the code.


# Code Overview


This repository largely follows the same interface as assignment 2.


## Train, Predict, Evaluate


You have three scripts in the repository `train.py`, `predict.py` and `evaluate.py` for training, predicting and evaluating a Dependency Parsing Model. You can supply `-h` flag to each of these to figure out how to use these scripts.

Here we show how to use the commands with the defaults.


#### Train a model
```
python train.py data/train.conll data/dev.conll

# stores the model by default at : serialization_dirs/default
```

#### Predict with model
```
python predict.py serialization_dirs/default \
                  data/dev.conll \
                  --predictions-file dev_predictions.conll
```

#### Evaluate model predictions

```
python evaluate.py serialization_dirs/default \
                   data/dev.conll \
                   dev_predictions.conll
```

**NOTE:** These scripts will not work until you fill-up the placeholders (TODOs) left out as part of the assignment.



## Dependency Parsing

  - `lib.model:` Defines the main model class of the neural dependency parser.

  - `lib.data.py`: Code dealing with reading, writing connl dataset, generating batches, extracting features and loading pretrained embedding file.

  - `lib.dependency_tree.py`: The dependency tree class file.

  - `lib.parsing_system.py`: This file contains the class for a transition-based parsing framework for dependency parsing.

  - `lib.configuration.py`: The configuration class file. Configuration reflects a state of the parser.

  - `lib.util.py`: This file contain function to load pretrained Dependency Parser.

  - `constants.py`: Sets project-wide constants for the project.


# Expectations

## What to write in code:

Like assignment 2 you have `TODO(Students) Start` and `TODO(Students) End` annotations. You are expected to write your code between those comment/annotations.

1. Implement the arc-standard algorithm in `parsing_system.py`: `apply` method
2. Implement feature extraction in `data.py`: `get_configuration_features` method
3. Implement neural network architecture in `model.py` in `DependencyParser` class: `__init__` and `call` method.
4. Implement loss function for neural network in `model.py` in `DependencyParser` class: `compute_loss` method.


## What experiments to try

You should try experiments to figure out the effects of following on learning:

1. activations (cubic vs tanh vs sigmoid)
2. pretrained embeddings
3. tunability of embeddings

and write your findings in the report.

The file `experiments.sh` enlists the commands you will need to train and save these models. In all you will need ~5 training runs, each taking about 30 minutes on cpu. See `colab_notes.md` to run experiments on gpu.

## As shown in the `experiments.sh`, you can use `--experiment-name` argument in the `train.py` to store the models at different locations in `serialization_dirs`. You can also use `--cache-processed-data` and `--use-cached-data` flags in `train.py` to not generate the training features everytime. Please look at training script for details. Lastly, after training your dev results will be stored in serialization directory of the experiment with name `metric.txt`.

**NOTE**: You will be reporting the scores on development set and submitting to use the test prediction of the configuration that you found the best. The labels of test dataset are hidden from you.


## What to turn in?

A single zip file containing the following files:

1. parsing_system.py
2. data.py
3. model.py
4. test_predictions.conll
5. gdrive_link.txt
6. report.pdf

`gdrive_link.txt` should have a link to the `serialization_dirs.zip` of your trained models.

We will release the exact zip format on piazza in a couple of days.

### Good Luck!
