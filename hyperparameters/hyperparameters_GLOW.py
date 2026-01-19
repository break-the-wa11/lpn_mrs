"""
In this file, we define the hyperparameters for the training_GLOW script.
The hyperparameters might depend on the regularizer and the task.
"""

import argparse

def get_GLOW_hyperparameters():
    args = argparse.Namespace()

    args.num_steps = 3000
    args.validate_every_n_steps = 200
    args.lr = 1e-4

    return args