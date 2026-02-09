"""
In this file, we define the hyperparameters for the perturb script.
"""

def get_perturb_hyperparameters():
    perturb_hyperparameter_list = [
        {
            "perturb_mode": "clean",
        },
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
        },
        {
            "perturb_mode": "baseline"
        },
        {       
            "perturb_mode": "low_lipid"
        },
        {       
            "perturb_mode": "borderline_lipid"
        },
        {       
            "perturb_mode": "high_lipid"
        },
        # {
        #     "perturb_mode": "slope",
        #     "parameters": {
        #         "start": [0,0.5],
        #         "end": [0,0.5]
        #     }
        # },
        # {
        #     "perturb_mode": "level",
        # },
        # {
        #     "perturb_mode": "spike",
        #     "parameters": {
        #         "num": [1,5],
        #         "stretch": [2,10],
        #         "magnitude": [0,0.2]
        #     }
        # },
        # {
        #     "perturb_mode": "magnitude change",
        #     "parameters": {
        #         "magnitude": [-0.2,0.2]
        #     }
        # },
        # {
        #     "perturb_mode": "lipid",
        # }
    ]

    return perturb_hyperparameter_list