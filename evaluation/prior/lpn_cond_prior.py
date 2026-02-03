import torch
from tqdm import tqdm
import numpy as np
from scipy.optimize import fmin_cg

def prox(x, model, sigma = 0.01):
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
    b = x.size(0)
    noise_levels = torch.full((b,1), sigma).to(device)
    return model(x, noise_levels).detach().cpu().numpy()

def cvx(x, model, sigma = 0.01):
    """
    Evaluate the learned convex function at x.

    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model

    Returns:
    y (numpy): (b), a vector of b values
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    b = x.size(0)
    noise_levels = torch.full((b,1), sigma).to(device)
    return model.scalar(x, noise_levels).squeeze(1).detach().cpu().numpy()

def invert_ls(x, model, sigma = 0.01, max_iter=1000, epsilon = 1e-6, verbose=True):
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
    for i in tqdm(range(max_iter), disable=not verbose):
        optimizer.zero_grad()
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

def torch_f_cg(x, model, z, sigma = 0.01):
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
    noise_levels = torch.full((1,1), sigma)
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
        sigma (float): conditional noise

    Return:
    v (numpy): (1,), scaler objective function value evaluated at x
    """
    model, z, sigma = args
    x = torch.tensor(x).view(z.shape).float()
    z = torch.tensor(z).float()
    device = next(model.parameters()).device
    x,z = x.to(device), z.to(device)
    v = torch_f_cg(x, model, z, sigma=sigma)
    v = v.cpu().detach().numpy()
    return v

def gradf(x, *args):
    """
    Gradient of the objective function for invert_cvx_cg

    Params:
    x (numpy): (n,), flattened model input
    args: (model, z)
        model: LPN model
        z (numpy): (*), single input for inverse evaluation, the shape of each input should match the input shape of the model
        sigma (float): conditional noise

    Return:
    g (numpy): (n), flattened gradient
    """
    model, z, sigma = args
    x = torch.tensor(x).view(z.shape).float()
    z = torch.tensor(z).float()
    device = next(model.parameters()).device
    x,z = x.to(device), z.to(device)
    x.requires_grad_(True)
    v = torch_f_cg(x, model, z, sigma=sigma)
    v.backward()
    g = x.grad.cpu().numpy().flatten()
    return g

def invert_cvx_cg(x, model, sigma = 0.01):
    """Invert the LPN model at x by convex optimization with conjugate gradient.
    Solve min_y psi_theta(y) - <x,y>
    
    Params:
    x (numpy): (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
    model: LPN model.
    sigma: conditional noise
    
    Returns:
    y (numpy): (b, *), b outputs, inverse of LPN model at x
    """
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z = x[i].copy()

        x0 = z.copy().flatten() # Initial point for cvx_cg
        args = (model, z, sigma)
        x_list = []
        callback = lambda x: x_list.append(x)
        res = fmin_cg(
            f, x0, fprime=gradf, args=args, full_output=True, disp=0, callback=callback, maxiter = 200
        )
        # print(
        #     f"fopt: {res[1]}, func_calls: {res[2]}, grad_calls: {res[3]}, warnflag: {res[4]}"
        # )
        y[i] = x_list[-1].reshape(z.shape)

    print("final mse: ", np.mean((prox(y, model) - x) ** 2))
    return y

def torch_f_gd(x, model, z, sigma = 0.01):
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
    noise_levels = torch.full((b,1), sigma)
    v = model.scalar(x, noise_levels).squeeze() - torch.sum(z.reshape(b, -1) * x.reshape(b, -1), dim=1)
    v = v.sum()
    return v

def invert_cvx_gd(x, model, sigma = 0.01):
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

    for i in range(2000):
        optimizer.zero_grad()
        loss = torch_f_gd(x, model, z, sigma=sigma)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("loss", loss.item())

    print("final mse: ", (model(x) - z).pow(2).mean().item())
    x = x.detach().cpu().numpy()
    return x

def invert(x, model, inv_alg, sigma = 0.01, **kwargs):
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
        return invert_cvx_cg(x, model, sigma=sigma)
    elif inv_alg == "cvx_gd":
        return invert_cvx_gd(x, model, sigma=sigma)
    else:
        raise ValueError("Unknown inversion algorithm:", inv_alg)
    

def eval_lpn_cond_prior(x, model, inv_alg, sigma = 0.01, **kwargs):
    """
    Evaluate the learned prior at x.

    Params:
        x: (b, *), b inputs for LPN model, the shape of each input should match the input shape of the model
        model: LPN model
        inv_alg: Inversion algorithm, choose from ['ls', 'cvx_cg', 'cvx_gd']
        sigma (float): conditional noise

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

    noise_levels = torch.full((b,1), sigma).to(device)

    # invert 
    y = invert(x, model, inv_alg, sigma=sigma, **kwargs)
    y_torch = torch.tensor(y).float().to(device)
    fy = model(y_torch, noise_levels).detach().cpu().numpy()

    # compute prior
    psi = model.scalar(y_torch, noise_levels).squeeze(1).detach().cpu().numpy()
    q = 0.5 * np.sum(x.reshape(b,-1)**2, axis=1)
    ip = np.sum(y.reshape(b,-1) * x.reshape(b,-1), axis=1)
    p = ip - q - psi

    return p,y,fy