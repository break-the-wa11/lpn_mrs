import numpy as np
import random
import os

from datasets import MRSDataset

def perturb_generator(n_samples, perturb_arg, raw_data_dir, save_dir):
    """
    Generate perturbed data samples and save to specified directory.
    Args:
        n_samples: int, number of samples to generate
        perturb_arg: dict, parameters for the perturbation
        raw_data_dir: str, directory of raw test_data
        save_dir: str, directory to save the perturbed samples
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if perturb_arg['perturb_mode'] == 'lipid':
        dataset = MRSDataset(root=raw_data_dir, split='test', data_type='metab_lipid')
        n_samples = min(n_samples, len(dataset))
        random_indices = random.sample(range(len(dataset)), n_samples)

        samples = []
        for idx in random_indices:
            samples.append(dataset[idx])

        samples = np.array(samples)
        np.save(os.path.join(save_dir, f"lipid.npy"), samples)
    else:
        dataset = MRSDataset(root='data/', split='test', data_type='metab')
        n_samples = min(n_samples, len(dataset))
        random_indices = random.sample(range(len(dataset)), n_samples)

        samples = []
        for idx in random_indices:
            clean_sample = dataset[idx]
            if perturb_arg['perturb_mode'] == 'gaussian':
                sigma = perturb_arg['parameters']['sigma']
                noisy_sample = clean_sample + np.random.normal(0, sigma, clean_sample.shape[-1])

            elif perturb_arg['perturb_mode'] == 'slope':
                start = random.uniform(perturb_arg['parameters']['start'][0],perturb_arg['parameters']['start'][1])
                end = random.uniform(perturb_arg['parameters']['end'][0],perturb_arg['parameters']['end'][1])
                slope = np.linspace(start, end, clean_sample.shape[-1])
                noisy_sample = clean_sample + slope

            elif perturb_arg['perturb_mode'] == 'level':
                s,e = random.sample(range(clean_sample.shape[-1]),2)
                start,end = min(s,e), max(s,e)
                noisy_sample = np.copy(clean_sample)
                noisy_sample[start:end] = 0

            elif perturb_arg['perturb_mode'] == 'spike':
                num = random.randint(perturb_arg['parameters']['num'][0],perturb_arg['parameters']['num'][1])
                pos = random.sample(range(clean_sample.shape[-1]-2),num)
                noisy_sample = np.copy(clean_sample)

                for id in pos:
                    stretch = min(random.randint(perturb_arg['parameters']['stretch'][0],perturb_arg['parameters']['stretch'][1]), clean_sample.shape[-1] - id - 1)
                    magnitude = random.uniform(perturb_arg['parameters']['magnitude'][0], perturb_arg['parameters']['magnitude'][1])
                    noisy_sample[id:(id+stretch)] += np.linspace(0, magnitude, stretch)

            elif perturb_arg['perturb_mode'] == 'magnitude change':
                pos = random.randint(0, clean_sample.shape[-1]-1)
                magnitude = random.uniform(perturb_arg['parameters']['magnitude'][0], perturb_arg['parameters']['magnitude'][1])
                noisy_sample = np.copy(clean_sample)
                noisy_sample[pos:(clean_sample.shape[-1])] += magnitude
            else:
                raise ValueError(f"Unknown perturbation mode: {perturb_arg['perturb_mode']}")
            
            samples.append(noisy_sample)

        samples = np.array(samples)
        if perturb_arg['perturb_mode'] == 'gaussian':
            np.save(os.path.join(save_dir, f"gaussian_{perturb_arg['parameters']['sigma']}.npy"), samples)
        else:
            np.save(os.path.join(save_dir, f"{perturb_arg['perturb_mode']}.npy"), samples)