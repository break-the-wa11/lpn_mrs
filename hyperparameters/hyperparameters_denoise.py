"""
In this file, we define the hyperparameters for the perturb script.
"""

def get_denoise_hyperparameters():
    perturb_hyperparameter_list = [
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.005
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.01
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.02
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.03
            }
        }
    ]

    return perturb_hyperparameter_list