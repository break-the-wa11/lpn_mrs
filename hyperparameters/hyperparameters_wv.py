"""
In this file, we define the hyperparameters for the perturb script.
"""

def get_wv_hyperparameters():
    threshold_dict = {
        0.005:0.2,
        0.01:0.2,
        0.02:0.3,
        0.03:0.3}

    return threshold_dict