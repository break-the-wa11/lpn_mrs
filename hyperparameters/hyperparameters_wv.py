"""
In this file, we define the hyperparameters for the perturb script.
"""

def get_wv_hyperparameters():
    threshold_dict = {
        '0.05':0.4,
        '0.1':0.4,
        '0.15':0.5,
        '0.2':0.6}

    return threshold_dict