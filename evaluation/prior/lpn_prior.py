import torch
from tqdm import tqdm
import numpy as np
from scipy.optimize import fmin_cg, minimize
import pandas as pd
import os

def prox(x, model, sigma=None):
    """
    Evaluate the learned proximal operator at x.

    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN_cond model

    Returns:
    y (numpy): (b, *), b outputs from LPN model
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)

    #LPN
    if sigma is None:
        return model(x).detach().cpu().numpy()
    
    #LPN_cond
    b = x.size(0)
    noise_levels = torch.full((b,1), sigma).to(device)
    return model(x, noise_levels).detach().cpu().numpy()

def invert_ls(x, model, maxiter, sigma=None, epsilon = 1e-6, verbose=True):
    """Invert the LPN model at x by least squares min_y||f_theta(y) - x||_2^2.
    
    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model.
    
    Returns:
    y (numpy): (b, *), b outputs, inverse of LPN model at x
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    y = torch.zeros(x.shape).float().to(device)
    y.requires_grad_(True)

    optimizer = torch.optim.Adam([y], lr=1e-2)

    b = x.size(0)
    noise_levels = torch.full((b,1), sigma).to(device)
    for i in tqdm(range(maxiter), disable=not verbose):
        optimizer.zero_grad()

        if sigma is None:
            loss = (model(y) - x).pow(2).mean()
        else:
            loss = (model(y, noise_levels) - x).pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            if loss.item() < epsilon:
                break
            print(f"iter {i}: mse {loss.item()}")
    print("final mse", loss.item())

    y = y.detach().cpu().numpy()
    return y

def torch_f_cg(x, model, z, sigma=None):
    """
    Compute the value of the objective function:
    psi(x) - <z, x>

    Params:
    x (tensor): (*) should match the input shape of the model
    model: ICNN
    z (tensor): (*) should match the input shape of the model
    
    Return: 
    v (tensor): shape (1,), objective function value
    """
    if sigma is None:
        v = model.scalar(x.unsqueeze(0)).squeeze() - torch.sum(z.reshape(-1) * x.reshape(-1))
    else:
        device = x.device
        noise_levels = torch.full((1,1), sigma, device=device)
        v = model.scalar(x.unsqueeze(0), noise_levels).squeeze() - torch.sum(z.reshape(-1) * x.reshape(-1))
    return v

def f(x, *args):
    """
    Objective function for invert_cvx_cg, i.e. f(x) = psi_theta(x) - <z,x>

    Params:
    x (numpy): (n,), flattened model input
    args: (model, z, sigma)
        model: LPN model
        z (numpy): (*), single input for inverse evaluation, the shape of each input should match the input shape of the model
        sigma (float) or None: conditional noise

    Return:
    v (numpy): float objective function value evaluated at x
    """
    if len(args) == 2:
        model, z = args
        sigma = None
    else:
        model, z, sigma = args

    device = next(model.parameters()).device
    x = torch.tensor(x, device=device, dtype=torch.float32).view(z.shape)
    z = torch.tensor(z, device=device, dtype=torch.float32)
    
    v = torch_f_cg(x, model, z, sigma=sigma)
    v = v.cpu().detach().numpy()
    return float(np.array(v).reshape(()))

def gradf(x, *args):
    """
    Gradient of the objective function for invert_cvx_cg

    Params:
    x (numpy): (n,), flattened model input
    args: (model, z)
        model: LPN model
        z (numpy): (*), single input for inverse evaluation, the shape of each input should match the input shape of the model
        sigma (float) or None: conditional noise

    Return:
    g (numpy): (n), flattened gradient
    """
    if len(args) == 2:
        model, z = args
        sigma = None
    else:
        model, z, sigma = args

    device = next(model.parameters()).device
    x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=True).view(z.shape)
    z = torch.tensor(z, device=device, dtype=torch.float32)

    v = torch_f_cg(x, model, z, sigma=sigma)
    v.backward()
    g = x.grad.cpu().numpy().flatten()
    return np.asarray(g, dtype=np.float64).flatten()

def invert_cvx_cg(x, model, sigma=None, maxiter=200, logger=None,
                  check_every=10, save_mse_path=None):
    """Invert the LPN model at x by convex optimization with CG.
    Solve min_y psi_theta(y) - <x,y>
    
    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model.
    sigma (float or None): conditional noise
    maxiter (int): maximum iterations
    logger: optional logger
    check_every (int): check MSE every this many iterations
    save_mse_path (str or None): if provided, save MSE history to this CSV file
    
    Returns:
    y (numpy): (b, *), b outputs, inverse of LPN model at x
    """
    y = np.zeros(x.shape)
    all_mse_records = []

    if save_mse_path is not None:
        # create folder if it doesn't exist
        save_dir = os.path.dirname(save_mse_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    for i in range(x.shape[0]):
        z = x[i].copy()
        x0 = z.copy().flatten()
        
        if sigma is None:
            args = (model, z)
        else:
            args = (model, z, sigma)
        
        # state tracking
        state = {
            'iter': 0,
            'mse_history': []
        }
        
        def _scipy_callback(xk):
            state['iter'] += 1
            
            # only check MSE every check_every iterations
            if state['iter'] % check_every != 0:
                return
            
            # compute MSE
            yi = np.asarray(xk).reshape(z.shape)
            prox_out = prox(np.expand_dims(yi, 0), model, sigma=sigma)
            mse = float(np.mean((prox_out - z) ** 2))
            state['mse_history'].append((state['iter'], mse))
        
        # run optimizer
        res = minimize(
            f, x0, args=args, method='CG', jac=gradf,
            callback=_scipy_callback,
            options={'maxiter': maxiter, 'gtol': 1e-5, 'disp': False}
        )
        
        # use final result
        y[i] = res.x.reshape(z.shape)

        # collect MSE records for this sample
        for it, mse in state['mse_history']:
            all_mse_records.append({
                'sample': i,
                'iter': it,
                'mse': mse
            })
        
        # logging
        if logger:
            mse_str = ', '.join([f"iter {it}: {mse:.6e}" for it, mse in state['mse_history']])
            logger.info(f"Sample {i}: {len(state['mse_history'])} MSE checks")
            logger.info(f"  MSE history: {mse_str}")

    # final batch MSE
    final_mse = np.mean((prox(y, model, sigma=sigma) - x) ** 2)
    print("final mse:", final_mse)
    if logger is not None:
        logger.info(f"Final batch mse: {final_mse}")

    # save MSE history to CSV if path provided
    if save_mse_path is not None:
        df = pd.DataFrame(all_mse_records)
        df.to_csv(save_mse_path, index=False)
        print(f"MSE history saved to {save_mse_path}")
    
    return y

def torch_f_gd(x, model, z, sigma = None):
    """
    Compute the value of the objective function:
    psi(x) - <z, x>
    
    Params:
    x (tensor): (b, *) should match the input shape of the model
    model: ICNN
    z (tensor): (b, *) should match the input shape of the model
    sigma (float): conditional noise
    
    Return: 
    v (tensor): shape (1,), objective function value
    """
    b = x.shape[0]
    if sigma is None:
        v = model.scalar(x).squeeze() - torch.sum(z.reshape(b, -1) * x.reshape(b, -1), dim=1)
    else:
        device = x.device
        noise_levels = torch.full((b,1), sigma, device=device)
        v = model.scalar(x, noise_levels).squeeze() - torch.sum(z.reshape(b, -1) * x.reshape(b, -1), dim=1)
    v = v.sum()
    return v

def invert_cvx_gd(x, model, sigma = None, maxiter = 200):
    """
    Invert the learned proximal operator at x by convex optimization with gradient descent.
    
    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model.
    sigma (float): conditional noise
    
    Return:
    y (numpy): (b, *), b outputs, inverse of LPN model at x
    """
    z = x.copy()
    device = next(model.parameters()).device
    z = torch.tensor(z).float().to(device)
    x = torch.zeros(z.shape).to(device)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=1e-2)

    for i in range(maxiter):
        optimizer.zero_grad()
        loss = torch_f_gd(x, model, z, sigma=sigma)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("loss", loss.item())

    print("final mse: ", (model(x) - z).pow(2).mean().item())
    x = x.detach().cpu().numpy()
    return x

def invert(x, model, inv_alg, sigma, **kwargs):
    """
    Invert the LPN model at x.

    Params:
    x (numpy): (b,*), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model
    inv_alg: Inversion algorithm, choose from ['ls', 'cvx_cg', 'cvx_gd']

    Returns:
    y (numpy): (b,*), b outputs, inverse of LPN model at x
    """
    if inv_alg == "ls":
        return invert_ls(x, model, sigma=sigma, **kwargs)
    elif inv_alg == "cvx_cg":
        return invert_cvx_cg(x, model, sigma=sigma, **kwargs)
    elif inv_alg == "cvx_gd":
        return invert_cvx_gd(x, model, sigma=sigma, **kwargs)
    else:
        raise ValueError("Unknown inversion algorithm:", inv_alg)
    

def eval_lpn_prior(x, model, inv_alg, sigma, **kwargs):
    """
    Evaluate the learned prior at x.

    Params:
        x: (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
        model: LPN model
        inv_alg: Inversion algorithm, choose from ['ls', 'cvx_cg', 'cvx_gd']
        sigma (float or None): conditional noise

    Returns:
        p: (b, ), numpy.ndarray, the prior value at x
        y: (b, *), numpy.ndarray, the inverse of model at x
        fy: (b, *), numpy.ndarray, the model output at y

    Note: The shape of x should match the input shape of model.

    Formula: phi(f(y)) = <y, f(y)> - 1/2 ||f(y)||^2 - psi(y)

    """
    b = x.shape[0]
    device = next(model.parameters()).device

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        

    # invert 
    y = invert(x, model, inv_alg, sigma=sigma, **kwargs)
    y_torch = torch.tensor(y).float().to(device)
    if sigma is None:
        fy = model(y_torch).detach().cpu().numpy()
        psi = model.scalar(y_torch).squeeze(1).detach().cpu().numpy()
    else:
        noise_levels = torch.full((b,1), sigma).to(device)
        fy = model(y_torch, noise_levels).detach().cpu().numpy()
        psi = model.scalar(y_torch, noise_levels).squeeze(1).detach().cpu().numpy()

    # compute prior
    q = 0.5 * np.sum(x.reshape(b,-1)**2, axis=1)
    ip = np.sum(y.reshape(b,-1) * x.reshape(b,-1), axis=1)
    p = ip - q - psi

    return p,y,fy