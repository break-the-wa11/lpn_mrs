"""
In this file, we define the hyperparameters for the training_LPN script.
The hyperparameters might depend on the regularizer and the task.
"""

import argparse

def get_LPN_hyperparameters():
    args = argparse.Namespace()

    args.num_steps = 20000
    args.validate_every_n_steps = 500
    args.num_steps_pretrain = 2000
    args.pretrain_lr = 1e-3
    args.num_stages = 4
    args.gamma_init = 0.1
    args.lr = 1e-4

    return args