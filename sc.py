# Yubei Chen, Sparse Manifold Coding Lib Ver 0.1
"""
This file contains multiple method to sparsify the coefficients
"""
import time
import numpy as np
import numpy.linalg as la
#import utility
from IPython import display
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# import cupy as cp


def quadraticBasisUpdate(basis, Res, ahat, lowestActivation, HessianDiag, stepSize, sigma, constraint = 'L2', Noneg = False):
    """
    This matrix update the basis function based on the Hessian matrix of the activation.
    It's very similar to Newton method. But since the Hessian matrix of the activation function is often ill-conditioned, we takes the pseudo inverse.
    Note: currently, we can just use the inverse of the activation energy.
    A better idea for this method should be caculating the local Lipschitz constant for each of the basis.
    The stepSize should be smaller than 1.0 * min(activation) to be stable.
    """
    dBasis = stepSize/sigma**2*torch.mm(Res, ahat.t())/ahat.size(1)
    dBasis = dBasis.div_(HessianDiag+lowestActivation)
    basis = basis.add_(dBasis)
    if Noneg:
        basis = basis.clamp(min = 0.)
    if constraint == 'L2':
        basis = basis.div_(basis.norm(2,0))
    return basis

def ISTA_PN(I,basis,lambd,num_iter,eta=None, useMAGMA=True):
    # This is a positive-negative PyTorch-Ver ISTA solver
    # MAGMA uses CPU-GPU hybrid method to solve SVD problems, which is great for single task. When running multiple jobs, this flag should be turned off to leave the svd computation on only GPU.
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat_sign = torch.sign(ahat)
        ahat.abs_()
        ahat.sub_(eta * lambd).clamp_(min = 0.)
        ahat.mul_(ahat_sign)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res

def FISTA(I,basis,lambd,num_iter,sigma,eta=None, useMAGMA=True, plot=False):
    # This is a positive-only PyTorch-Ver ISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
#             L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])/sigma**2
            jittered_basis = basis + torch.rand(basis.size()).cuda()*1.0e-15
            L = torch.from_numpy(np.array([np.nanmax(torch.symeig(torch.mm(jittered_basis,jittered_basis.t()),eigenvectors=False)[0].cpu().numpy())])).cuda()/sigma**2
            eta = 1./L
#         else: # Need to change with sigma
#             eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
#             eta = torch.from_numpy(eta.astype('float32')).cuda()

    tk_n = 1.
    tk = 1.
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)
    ahat_y = torch.cuda.FloatTensor(M,batch_size).fill_(0)
    
    if plot:
        fig = plt.figure(figsize=(10,5))
    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - torch.mm(basis,ahat_y)
#         print("ahat_y", torch.isnan(ahat_y).any())
#         print("RES", torch.isnan(Res).any())
#         print("eta", torch.isnan(eta).any())
        ahat_y = ahat_y.add(eta * basis.t().mm(Res)/sigma**2)
#         print(torch.isnan(ahat_y).any())
        ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
#         print(torch.isnan(ahat).any())
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
#         print(torch.isnan(ahat_y).any())
        
        # Plot snr and alpha
        if plot and t % 1 == 0:
            snr = ((I**2).sum(dim=0)/(Res**2).sum(dim=0)).mean()
            fig.add_subplot(1,2,1)
            plt.scatter(t, snr.cpu().numpy())
            fig.add_subplot(1,2,2)
            plt.scatter(np.ones(ahat.size()[0])*t, ahat[:,0].cpu().numpy(), c=np.arange(ahat.size()[0]))
            display.clear_output(wait=True)
            display.display(plt.gcf())
    if plot:
        display.clear_output()
    Res = I - torch.mm(basis,ahat)
    return ahat, Res

def ISTA(I,basis,lambd,num_iter,eta=None, useMAGMA=True):
    # This is a positive-only PyTorch-Ver ISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat = ahat.sub(eta * lambd).clamp(min = 0.)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res

def ISTA_tempo(I,P,basis,D,tilde,lambd,num_iter,step_mod = 1.0,eta=None, useMAGMA=True):
    # This is a positive-only PyTorch-Ver ISTA solver
    # temporal regularization: we use pooling to regularize the sparse inference
    # objective: min ||I - Phi A||_2^2 + \tilde ||PAD||_2^2 + \lambda ||A||_1
    # \tilde is the temporal regularization coeff
    # \lambd is the sparse regularization coeff

    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])
            eta = step_mod/L
        else:
            eta = step_mod/cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()
    DD = torch.mm(D,D.t())

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ADD = torch.mm(ahat,DD)
        PADD = torch.mm(P,ADD)
        PPADD = torch.mm(P.t(),PADD)
        ahat = ahat.sub(tilde * eta * PPADD)
        ahat = ahat.sub_(eta * lambd).clamp(min = 0.)
        Res = I - torch.mm(basis,ahat)

    #For testing purpose:
    PAD = torch.mm(P,torch.mm(ahat,D))
    return ahat, Res, PAD

def FISTA_adapt(I,basis,lambd,weights,num_iter,eta=None, useMAGMA=True):
    # This is a positive-only PyTorch-Ver ISTA solver
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    tk_n = 1.
    tk = 1.
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)
    ahat_y = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in range(num_iter):
        tk = tk_n
        tk_n = (1+np.sqrt(1+4*tk**2))/2
        ahat_pre = ahat
        Res = I - torch.mm(basis,ahat_y)
        ahat_y = ahat_y.add(eta * basis.t().mm(Res))
        ahat = ahat_y.sub(eta * lambd * weights).clamp(min = 0.)
        ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
    Res = I - torch.mm(basis,ahat)
    return ahat, Res

def ISTA_adapt(I,basis,lambd,weights,num_iter,eta=None,useMAGMA=True):
    # This is a PyTorch-Ver ISTA solver
    # The adaptive version ISTA will have a different shrinkage, determined by weights, on each dimensional.
    dtype = basis.type()
    batch_size=I.size(1)
    M = basis.size(1)
    if eta is None:
        if useMAGMA:
            L = torch.max(torch.symeig(torch.mm(basis,basis.t()),eigenvectors=False)[0])
            eta = 1./L
        else:
            eta = 1./cp.linalg.eigvalsh(cp.asarray(torch.mm(basis,basis.t()).cpu().numpy())).max().get().reshape(1)
            eta = torch.from_numpy(eta.astype('float32')).cuda()

    #Res = torch.zeros(I.size()).type(dtype)
    #ahat = torch.zeros(M,batch_size).type(dtype)
    Res = torch.cuda.FloatTensor(I.size()).fill_(0)
    ahat = torch.cuda.FloatTensor(M,batch_size).fill_(0)

    for t in xrange(num_iter):
        ahat = ahat.add(eta * basis.t().mm(Res))
        ahat = ahat.sub_(eta * lambd * weights).clamp(min = 0.)
        Res = I - torch.mm(basis,ahat)
    return ahat, Res