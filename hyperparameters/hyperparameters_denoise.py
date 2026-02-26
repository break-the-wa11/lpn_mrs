"""
In this file, we define the hyperparameters for the perturb script.
"""

def get_denoise_hyperparameters():
    perturb_hyperparameter_list = [
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.05,
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.1,
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.15,
            }
        },
        {
            "perturb_mode": "gaussian",
            "parameters": {
                "sigma": 0.2,
            }
        }
    ]

    return perturb_hyperparameter_list