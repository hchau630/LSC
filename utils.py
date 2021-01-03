import numpy as np
import torch
import time

device = torch.device("cuda")

def trapz(y, delta_x):
    """
    1-dimensional integral with trapezoidal rule
    y - y values of function - (N,...)
    delta_x - (1)
    N is the number of sample points
    """
    return (delta_x * (y[1:] + y[:-1]) / 2.0).sum(dim=0) # (...)

def trapz_n(y, delta_x):
    """
    n-dimensional integral with trapezoidal rule
    y - y values of function - (N1,...,Nn,...)
    delta_x - (n)
    N is the number of sample points along each dimension
    """
    n = delta_x.size()[0]
    result = y
    for i in range(n):
        result = trapz(result, delta_x[i]) # Don't use trapezoidal rule to avoid loop.
    return result

def simple_integral_n(y, delta_x):
    """
    n-dimensional integral
    y - y values of function - (N1-1,...,Nn-1,...)
    delta_x - (n)
    N is the number of sample points along each dimension
    """
    n = delta_x.size()[0]
    dxn = torch.prod(delta_x)
    return torch.sum(y, dim=tuple(np.arange(n)))*dxn

def to_cartesian(r, theta):
    return (r*torch.cos(theta)).float(), (r*torch.sin(theta)).float()
    
def to_polar(a, b):
    return (a**2 + b**2).sqrt().float(), torch.atan2(b, a).float()

def grid(N, n, a=0, b=2*np.pi, include_end=False):
    if include_end:
        sn = torch.linspace(a, b, steps=N, device=device).repeat(n,1) # (n,N)
        return torch.stack(torch.meshgrid(*sn),dim=n) # (N,...,N,n) where there are n number of Ns.
    else:
        sn = torch.linspace(a, b, steps=N, device=device)[:-1].repeat(n,1) # (n,N-1)
        return torch.stack(torch.meshgrid(*sn),dim=n) # (N-1,...,N-1,n) where there are n number of N-1s.

def T_multi(j, s):
    """
    j - (J, n)
    s - (N1, ..., Nn, n)
    returns (J, N1, ..., Nn)
    """
    js = torch.einsum('jn,...n->j...',j,s)
    return torch.cos(js), torch.sin(js)

def exp_unnormalized_circular_moment_numint_multi(k, m, o1, o2, N, return_scale=False, map_est=False):
    """
    Computes unnormalized circular moment for the n-parameter commutative transformation
    int_0^2pi exp(eta \dot T(o1,s))cos(o2 \dot s) is numerically unstable because of the exponential, so instead we compute
    int_0^2pi exp(eta \dot T(o1,s) - max_s{eta \dot T(o1,s)})cos(o2 \dot s) 
    (need to ensure max_s{eta \dot T(s)} remains the same when calculating Z(eta))
    k, m - kappa, mu (B,J), (B,J)
    o1 - omegas of distribution (J,n), where n is the number of parameters of the transformation
    o2 - omegas of moments (M,n)
    N - number of sample points to evaluate integral
    """
    B, J, M, n = k.size()[0], k.size()[1], o2.size()[0], o1.size()[1]
    grid_s = grid(N, n) # (N-1,...,N-1, n) # Don't create new grid everytime
    a, b = to_cartesian(k, m) # (B,J), (B,J)
    tc, ts = T_multi(o1.float(),grid_s) # (J,N,...,N), (J,N,...,N)
    pre_exp = torch.einsum('bj,j...->b...',a,tc) + torch.einsum('bj,j...->b...',b,ts) # (B,N,...,N)
    pre_exp_max = pre_exp.view(B,-1).max(dim=1)[0][(..., )+(None,)*n] # (B,1,...,1). [(..., )+(None,)*n] unsqueezes n dimensions.
    if map_est:
#         print("HI")
        pre_exp_max_idx = pre_exp.view(B,-1).max(dim=1)[1] # (B)
        exp_term = torch.zeros_like(pre_exp,device=device) # (B,N,...,N)
        exp_term.reshape(B,-1)[torch.arange(B),pre_exp_max_idx] = 1.0
    else:
        exp_term = torch.exp(pre_exp - pre_exp_max) # (B,N,...,N)
    mc, ms = T_multi(o2.float(),grid_s) # (M,N,...,N), (M,N,...,N)
    yc, ys = torch.einsum('b...,j...->...bj',exp_term,mc), torch.einsum('b...,j...->...bj',exp_term,ms) #(N,...N,B,M),(N,...N,B,M)
#     print(yc.size(), ys.size())
    delta_x = torch.ones(n,device=device)*2*np.pi/(N-1) # (n)
    result_c, result_s = simple_integral_n(yc,delta_x), simple_integral_n(ys,delta_x) # (B,M), (B,M)
    if return_scale:
        return result_c, result_s, pre_exp_max.squeeze() # (B,M), (B,M), (B)
    return result_c, result_s # (B,M), (B,M)

def circular_moment_numint_multi(k, m, o1, o2, N=100, map_est=False):
    """
    k, m - (B,J), (B,J)
    o1 - omegas of distribution (J,n)
    o2 - omegas of moments (M,n)
    N - number of sample points to evaluate integral
    Returns 1/Z(eta) * int_0^2pi exp(eta \dot T(o1,s))cos(o2 \dot s)
    """
    exp_un_c, exp_un_s = exp_unnormalized_circular_moment_numint_multi(k, m, o1, o2, N, map_est=map_est) # (B,M), (B,M)
    exp_norm_constant = exp_Z_numint_multi(k, m, o1, N, map_est=map_est).unsqueeze(1) # (B,1)
    return exp_un_c/exp_norm_constant, exp_un_s/exp_norm_constant # (B,M), (B,M)

def exp_Z_numint_multi(k, m, o1, N=100, map_est=False):
    """
    k, m - (B,J), (B,J)
    o1 - omegas of distribution (J,n)
    returns exp(- max_s{eta \dot T(s)})Z(eta) - (B)
    """
    n = o1.size()[1]
    o2 = torch.zeros(1,n,device=device) # (M=1,n)
    result = exp_unnormalized_circular_moment_numint_multi(k, m, o1, o2, N, map_est=map_est)[0].squeeze(1) # (B)
    return result

def log_Z_numint_multi(k, m, o1, N=100, map_est=False):
    """
    k, m - (B,J), (B,J)
    o1 - omegas of distribution (J,n)
    Returns ln Z(eta) - (B)
    """
    n = o1.size()[1]
    o2 = torch.zeros(1,n,device=device) # (M=1,n)
    exp_un_c, _, scale = exp_unnormalized_circular_moment_numint_multi(k, m, o1, o2, N, return_scale=True, map_est=map_est) # (B,1),(B,1),(B)
    return exp_un_c.squeeze(1).log() + scale # (B)

def get_distribution(k, m, o1, N=100):
    """
    Computes n-dimensional distribution evaluated at N^n number of points in [-pi, pi]^n
    k, m - (B,J), (B,J)
    o1 - (J,n)
    Returns: result (B,N,...,N)
    torch cuda input, numpy output
    """
    B, J, n = k.size()[0], k.size()[1], o1.size()[1]
    Z = log_Z_numint_multi(k, m, o1, N=N).exp()[(..., )+(None,)*n] # (B,1,...,1)
    s = grid(N, n, a=-np.pi, b=np.pi, include_end=True) # (N,...,N, n)
    a, b = to_cartesian(k, m) # (B,J), (B,J)
    tc, ts = T_multi(o1.float(),s) # (J,N,...,N), (J,N,...,N)
    unnormalized_dist = torch.exp(torch.einsum('bj,j...->b...',a,tc) + torch.einsum('bj,j...->b...',b,ts)) # (B,N,...,N)
    return (unnormalized_dist/Z).cpu().numpy() # Covert to numpy to save GPU space

def marginalize_distribution(post):
    B, N, n = post.shape[0], post.shape[1], post.ndim - 1
    if n == 1:
        return post[:,:,np.newaxis]
    post = torch.from_numpy(post).cuda()
    marginals = torch.zeros(B,N,n).cuda()
    for i in range(n):
        marginals[:,:,i] = torch.sum(post, dim=tuple(1+np.delete(np.arange(n),i)))
    post = post.cpu().numpy()
    return marginals.cpu().numpy()

def get_MAP(post):
    """
    Computes MAP estimate given the n-dimensional posterior distribution
    post - posterior distributions evaluated at N^n number of points in [-pi, pi]^n - (B,N,...,N)
    numpy input, torch cuda output
    """
    B, N, n = post.shape[0], post.shape[1], post.ndim - 1
    flat_idx = np.argmax(post.reshape((B,N**n)), axis=1) # (B)
    shape = tuple([N for _ in range(n)])   
    idx = np.array(np.unravel_index(flat_idx, shape)).T # (B,n)
    s = grid(N, n, a=-np.pi, b=np.pi, include_end=True) # (N,...,N, n)
    s_hat = torch.zeros(B,n).cuda() # (B,n)
    for b in range(B):
        s_hat[b] = s[tuple(idx[b])]
    # set s_hat to 0 if it is just a uniform distribution, which would imply the number of unique elements in post is 1.
    for i in range(B):
        if len(set(post[i].flatten())) <= 1:
            s_hat[i] = 0.0
    return s_hat # (B,n)