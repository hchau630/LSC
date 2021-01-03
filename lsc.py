import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as f
import utils
from IPython import display
import time

device = torch.device("cuda")

def combine_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
    """
    Compute variance of X given mean and variances of A and B, where X = A union B.
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#cite_note-:0-10
    """
    if count_a + count_b <= 1:
        return torch.zeros(var_a.size()).cuda()
    delta = avg_b - avg_a
    M2_a = var_a * (count_a - 1)
    M2_b = var_b * (count_b - 1)
    M2 = M2_a + M2_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    return M2 / (count_a + count_b - 1)

def combine_mean(avg_a, count_a, avg_b, count_b):
    """
    Compute variance of X given mean and variances of A and B, where X = A union B.
    """
    if count_a + count_b == 0:
        return torch.zeros(avg_a.size()).cuda()
    return (count_a * avg_a + count_b * avg_b)/(count_a + count_b)

def statistics(data):
    """
    Compute mean and variance of data along first axis
    """
    B = data.size()[0]
    avg = data.sum(dim=0)/B
    if B == 1:
        variance = torch.zeros(avg.size()).cuda()
    else:
        variance = ((data - avg)**2).sum(dim=0)/(B-1)
    return (avg, variance)

class RunningStatistics:
    """
    Compute running mean and variance
    """
    def __init__(self, size):
        self.avg = torch.zeros(size).cuda()
        self.variance = torch.zeros(size).cuda()
        self.stderr = torch.zeros(size).cuda()
        self.count = 0
        
    def update(self, data):
        """
        data - (batch_size, ...(any number of dimensions here)...)
        """
        new_avg, new_variance = statistics(data)
        new_count = data.size()[0]
        updated_avg = combine_mean(self.avg, self.count, new_avg, new_count)
        updated_variance = combine_variance(self.avg, self.count, self.variance, new_avg, new_count, new_variance)
        self.avg, self.variance = updated_avg, updated_variance
        self.count = self.count + new_count
        self.stderr = (self.variance/self.count)**0.5
        
    def print_stats(self):
        ratio = torch.abs(self.stderr/self.avg)
        print("Standard error ratio range: {} - {}".format(ratio.min(), ratio.max()))

def orthogonalize(w):
    u, s, v = torch.svd(w)
    return torch.mm(u, v.t())

def eta(T, eta_0=0.25):
    return eta_0/np.sqrt(T)

def to_cartesian(r, theta):
    return (r*torch.cos(theta)).float(), (r*torch.sin(theta)).float()
    
def to_polar(a, b):
    return (a**2 + b**2).sqrt().float(), torch.atan2(b, a).float()

def rot(delta_theta, a, b):
    r, theta = to_polar(a, b)
    return to_cartesian(r, theta+delta_theta)

def reconstruct(x, w, s_hat, omega):
    """
    x - (B,D)
    w - (B,L*2)
    s_hat - (B,n)
    omega - (L,n)
    returns y_hat - (B,D)
    """
    w1, w2 = w[:,::2], w[:,1::2] # (D,L), (D,L)
    u1 = torch.einsum('dl, bd -> bl', w1, x) # (B,L)
    u2 = torch.einsum('dl, bd -> bl', w2, x) # (B,L)
    m_hat = torch.einsum('ln,bn->bl', omega.float(), s_hat.float()) # (B,L)
    
    B, L = u1.size()[0], u1.size()[1]
    ru = torch.zeros((B,L*2)).cuda() # (B,L*2)
    ru1, ru2 = rot(m_hat, u1, u2) # (B,L), (B,L)
    ru[:,::2], ru[:,1::2] = ru1, ru2
    y_hat = torch.einsum('dj, bj -> bd', w, ru) # (B,D)
    return y_hat

def posterior_phi(u, v, omega, s, k=None, m=None):
    """
    Inputs:
    u, v - (B,L*2)
    omega - (L,n)
    k, m - (B,J)
    Returns:
    k_hat, m_hat - (B,J), where J = number of unique omegas excluding omega = (0,0)
    """
    u1, u2 = u[:,::2], u[:,1::2] # (B,L)
    v1, v2 = v[:,::2], v[:,1::2] # (B,L)
    B, L, n = u1.size()[0], u1.size()[1], omega.size()[1]
    assert isinstance(omega, torch.cuda.IntTensor)
    
    unique_omega, inverse_idx = torch.unique(omega, dim=0, return_inverse=True) # (J,n), (L)
    J = unique_omega.size()[0]
    a = (u1*v1+u2*v2)/s**2 # (B,L)
    b = (u1*v2-u2*v1)/s**2 # (B,L)

    graph1, graph2 = torch.zeros(B,L,J,device=device), torch.zeros(B,L,J,device=device)
    graph1[:,torch.arange(L,device=device),inverse_idx] = a
    graph2[:,torch.arange(L,device=device),inverse_idx] = b
    eta_posterior_1, eta_posterior_2 = graph1.sum(dim=1), graph2.sum(dim=1) # (B,J)
    zero_inds = (unique_omega == 0).all(dim=1)
    eta_posterior_1[:,zero_inds], eta_posterior_2[:,zero_inds] = 0, 0
    
    if k is not None and m is not None:
        eta_prior_1, eta_prior_2 = to_cartesian(k, m) # (B,J), (B,J)
        eta_posterior_1 += eta_prior_1
        eta_posterior_2 += eta_prior_2

    k_hat, m_hat = to_polar(eta_posterior_1, eta_posterior_2) # (B,J), (B,J)
    return k_hat, m_hat

def compute_q(u, v, omega, k_hat, m_hat, N=100, map_est=False):
    """
    Inputs:
    u, v - (B,L*2)
    omega - (L,n)
    k_hat, m_hat - (B,J)
    """
    B, L = u.size()[0], int(u.size()[1]/2)
    unique_omega, inverse_idx = torch.unique(omega, dim=0, return_inverse=True) # (J,n), (L)
    c, s = utils.circular_moment_numint_multi(k_hat, m_hat, unique_omega, unique_omega, N=N, map_est=map_est) # (B,J), (B,J) (0.0013s)
    c, s = c[:,inverse_idx], s[:,inverse_idx] # (B,L), (B,L)
    qc, qs = torch.empty(B,L*2,device=device), torch.empty(B,L*2,device=device)
    qc[:,::2], qc[:,1::2] = c.clone(), c.clone()
    qs[:,::2], qs[:,1::2] = s.clone(), s.clone()
    return qc, qs # (B,L*2),(B,L*2),((B,L*2),(B,L*2))

def compute_aux_var(y, psi, w, alpha, omega, sigma, k=None, m=None, N=100, map_est=False):
    x = torch.einsum('dk,bk->bd',psi,alpha) # (B,D)
    u = torch.einsum('dl,bd->bl',w,x) # (B,L*2)
    v = torch.einsum('dl,bd->bl',w,y) # (B,L*2)
    k_hat, m_hat = posterior_phi(u, v, omega, sigma, k=k, m=m) # (B,J), (B,J) (0.0013s)
    q = compute_q(u, v, omega, k_hat, m_hat, N=N, map_est=map_est) # (0.0017s)
    res = y - torch.einsum('dl,bl->bd',w,mul_q(q, u)) # (B,D)
    tres = torch.einsum('dl,bl->bd',w,mul_q(q, torch.einsum('dl,bd->bl',w,res), transpose=True))
    return x, u, v, k_hat, m_hat, q, res, tres

def mul_q(q, x, transpose=False):
    qc, qs = q[0], q[1]
    xc = x.clone()
    xs = torch.zeros(x.size(),device=device)
    xs[:,::2], xs[:,1::2] = -x[:,1::2].clone(), x[:,::2].clone()
    if transpose:
        xs = -xs
    result = qc*xc + qs*xs
    return result

def grad_w(x, res, w, q, sigma):
    B = x.size()[0]
    wres, wx = torch.einsum('dl,bd->bl',w,res), torch.einsum('dl,bd->bl',w,x)
    qwres, qwx = mul_q(q, wres, transpose=True), mul_q(q, wx, transpose=False)
    return (torch.einsum('bd,bl->dl',x,qwres) + torch.einsum('bd,bl->dl',res,qwx)) / (sigma**2*B)

def grad_w_modified(x, res, w, q, sigma):
    B = x.size()[0]
    wres, wx = torch.einsum('dl,bd->bl',w,res), torch.einsum('dl,bd->bl',w,x)
    qwres, qwx = mul_q(q, wres, transpose=True), mul_q(q, wx, transpose=False)
    return (torch.einsum('bd,bl->dl',x,qwres) + torch.einsum('bd,bl->dl',res,qwx)) / (sigma**2*B)

def grad_alpha(tres, psi, alpha, prev_alpha, sigma, lamb, lamb2):
    return torch.einsum('dk,bd->bk',psi,tres) / (sigma**2) - lamb*torch.sign(alpha) - 2*lamb2*(alpha - prev_alpha)

def grad_alpha_modified(tres, psi, alpha, prev_alpha, sigma, lamb, lamb2):
    return torch.einsum('dk,bd->bk',psi,tres) / (sigma**2) - lamb*torch.sign(alpha) - 2*lamb2*(alpha - prev_alpha)

def grad_psi(tres, alpha, sigma):
    B = tres.size()[0]
    return torch.einsum('bd,bk->dk',tres,alpha) / (sigma**2 * B)

def grad_psi_modified(psi, tres, alpha, sigma, lamb):
    B = tres.size()[0]
    reg_term = torch.einsum('dk,k->dk',psi,torch.diag(torch.einsum('dk,dl->kl',psi,psi))-1)
    return torch.einsum('bd,bk->dk',tres,alpha) / (sigma**2 * B) - lamb*reg_term

def alpha_update_FISTA(y, psi, w, omega, alpha, sigma, lamb, lamb2, steps, k=None, m=None, eta_alpha=0.001, N=100, modified=False, plot=False, map_est=False, adaptive=True, adaptive_steps=False):
    D, K = psi.size()[0], psi.size()[1]
    tk_n, tk = 1.0, 1.0
    prev_alpha, alpha_y = alpha.clone(), alpha.clone()
    if adaptive:
        wpsi = torch.mm(w.t(),psi)
        lipschitz = 1.5*torch.max(torch.symeig(torch.mm(wpsi.t(),wpsi),eigenvectors=False)[0])/sigma**2
        eta_alpha = 1.0/lipschitz
#         print(1/eta_alpha)
        if adaptive_steps:
            steps = int(21*torch.sqrt(0.001/eta_alpha))-1
#             print(eta_alpha, int(21*torch.sqrt(0.001/eta_alpha))-1)
    if plot:
        fig = plt.figure(figsize=(10,5))
    for t in range(steps):
        # From here till end excluding compute_aux_var (0.0033s)
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        alpha_pre = alpha.clone()
        x, u, v, k_hat, m_hat, q, res, tres = compute_aux_var(y, psi, w, alpha_y, omega, sigma, k=k, m=m, N=N, map_est=map_est) # (0.0032s)
        if modified:
            dalpha = eta(1, eta_0=eta_alpha)*grad_alpha_modified(tres, psi, alpha_y, prev_alpha, sigma, lamb, lamb2)
        else:
            dalpha = eta(1, eta_0=eta_alpha)*grad_alpha(tres, psi, alpha_y, prev_alpha, sigma, lamb, lamb2)
        alpha = (alpha_y + dalpha).clamp(min=0.)
        alpha_y = alpha + (tk-1)/tk_n * (alpha - alpha_pre)
        
        # Plot loss and alpha
        if plot and t % 1 == 0:
            nll = -log_likelihood(x, y, u, v, psi, alpha, k_hat, m_hat, omega, lamb, lamb2, sigma, mean=True, N=N, modified=modified)
            fig.add_subplot(1,2,1)
            plt.scatter(t, nll)
            fig.add_subplot(1,2,2)
            plt.scatter(np.ones(alpha.size()[1])*t, alpha[0].cpu().numpy(), c=np.arange(K))
            display.clear_output(wait=True)
            display.display(plt.gcf())
    if plot:
        display.clear_output()
    return alpha

def psi_update(psi, alpha, tres, sigma, eta_psi, lamb, modified=False):
    if modified:
        dpsi = eta(1, eta_0=eta_psi)*grad_psi_modified(psi, tres, alpha, sigma, lamb) # (D,K)
        new_psi = psi + dpsi
    else:
        dpsi = eta(1, eta_0=eta_psi)*grad_psi(tres, alpha, sigma) # (D,K)
        new_psi = f.normalize(psi + dpsi, dim=0)
#     if torch.isnan(new_psi).any():
#         print("Infinity in new_psi. Setting dpsi = 0.")
#         new_psi = psi.clone()
    dpsi_length = torch.mean(torch.norm(new_psi - psi, dim=0))
    psi = new_psi.clone()
    return psi, dpsi_length

def w_update_riemann(optimizer, x, w, q, res, sigma, modified=False):
    """
    optimizer - geoopt.optim.RiemannianAdam/geoopt.optim.RiemannianSGD
    psi, w - geoopt.ManifoldParameter
    """
    w_old = w.clone()
    optimizer.zero_grad()
    if modified:
        w.grad = -grad_w_modified(x, res, w, q, sigma)
    else:
        w.grad = -grad_w(x, res, w, q, sigma)
    optimizer.step()
    dw_norm = torch.norm(w - w_old)
    return w, dw_norm

def log_likelihood(x, y, u, v, psi, alpha, k_hat, m_hat, omega, lamb, lamb2, s, mean=True, N=100, modified=False, map_est=False):
    """
    Computes log likelihood.
    x, y - (B,D)
    u, v - (B,L*2)
    omega - (L,n)
    k_hat, m_hat - (B,J)
    Assume k = 0, m = 0
    """
    D = x.size()[1]
    u1, u2 = u[:,::2], u[:,1::2]
    v1, v2 = v[:,::2], v[:,1::2]
    k, m = torch.zeros(k_hat.size(),device=device), torch.zeros(m_hat.size(),device=device) # (B,J), (B,J)
    unique_omega = torch.unique(omega, dim=0) # (J)
    Z_term = utils.log_Z_numint_multi(k_hat, m_hat, unique_omega, N=N, map_est=map_est) - utils.log_Z_numint_multi(k, m, unique_omega, N=N, map_est=map_est) # (B)
    constant_term = - ((u**2).sum(dim=1)+(y**2).sum(dim=1))/(2*s**2) # (B)
    zero_inds = (omega == 0).all(dim=1)
    u10, u20 = u1[:,zero_inds], u2[:,zero_inds]
    v10, v20 = v1[:,zero_inds], v2[:,zero_inds]
    if len(zero_inds) > 0:
        zero_term = (torch.einsum('bn,bn->b',u10,v10) + torch.einsum('bn,bn->b',u20,v20))/s**2 # (B)
    else:
        zero_term = torch.zeros(x.size()[0],device=device) # (B)
        
    batch_log_likelihood = Z_term + constant_term + zero_term - lamb*alpha.norm(p=1,dim=1) # (B)
    if modified:
        batch_log_likelihood -= lamb2*torch.norm(torch.diag(torch.einsum('dk,dl->kl',psi,psi))-1)**2 # Psi regularization term
    if mean:
        return torch.mean(batch_log_likelihood).cpu().numpy() # (1)
    else:
        return batch_log_likelihood.cpu().numpy() # (B)