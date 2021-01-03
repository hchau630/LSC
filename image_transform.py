import numpy as np
from math import floor
import torch
import time
from scipy.signal import convolve2d
import torch.nn.functional as f

CONSTANT = 0
REPLICATE = 1

def gaussian_kernels(B, dim, sigma):
    """
    sigma - (B)
    """
    assert dim % 2 == 1
    c = (dim-1)/2
    i = torch.arange(dim).cuda().float()
    j = torch.arange(dim).cuda().float()
    gi, gj = torch.meshgrid(i,j) # (dim,dim), (dim,dim)
    gi, gj = gi.repeat(B,1,1), gj.repeat(B,1,1) # (B,dim,dim), (B,dim,dim)
    sigma = sigma.cuda().unsqueeze(1).unsqueeze(2).float() # (B,1,1)
    kernel = torch.exp(-0.5*((gi-c)**2+(gj-c)**2)/sigma**2)
    return kernel

def gaussian_blur(x, sigma):
    """
    x - (B,dim,dim)
    sigma - (B)
    """
    B, dim = x.size()[0], x.size()[1]
    x = x.cuda()
    y = torch.zeros(B,dim,dim).cuda()
    kernels = gaussian_kernels(B,dim-1,sigma) # (B,dim-1,dim-1)
    for i in range(B):
        x_input = x[i].unsqueeze(0).unsqueeze(0) # (1,1,dim,dim)
        kernel = kernels[i].unsqueeze(0).unsqueeze(0) # (1,1,dim-1,dim-1)
        padding = int((dim-2)/2)
        y[i] = f.conv2d(x_input,kernel,padding=padding)
#     y = y / y.view(B,dim**2).norm(dim=1).unsqueeze(1).unsqueeze(2) * 1.5*x.view(B,dim**2).norm(dim=1).unsqueeze(1).unsqueeze(2)
    return y

def thicken(x, sigma):
    """
    x - (B,dim,dim)
    sigma - (B)
    """
    y = gaussian_blur(x,sigma)
    y = y.clamp(min=0.0,max=1.0)
    return y.cpu()

def get_affine_transform(x, y, angle, scale, center):
    cx, cy = center[0], center[1]
    angle = angle.cuda().float() / 180 * np.pi
    scale = scale.cuda().float()
    x, y = x.cuda().float(), y.cuda().float()
    a = scale * torch.cos(angle)
    b = scale * torch.sin(angle)
    M1 = torch.stack((a,-b,(1-a)*cx+b*cy+x),dim=1) # (B,3)
    M2 = torch.stack((b,a,-b*cx+(1-a)*cy+y),dim=1) # (B,3)
    Ms = torch.stack((M1,M2),dim=2).permute(0,2,1)
    return Ms

def affine_2D(imgs, x, y, angle, scale, wrap_around=False):
    """
    Rotate image around center by ANGLE, scale image by SCALE, then translate by X and Y
    imgs - (B,dimx,dimy), torch no cuda
    x - (B), torch no cuda
    y - (B), torch no cuda
    angle - (B), torch no cuda, in degrees
    scale - (B), torch no cuda
    Output - (B,dimx,dimy), torch no cuda
    """
    B, dimx, dimy = imgs.size()[0], imgs.size()[1], imgs.size()[2]
    center = ((dimx-1)/2, (dimy-1)/2)
    Ms = get_affine_transform(-x, -y, -angle, 1/scale, center)
    transformed_imgs = transform(imgs,Ms,wrap_around=wrap_around)
    return transformed_imgs

def translate_2D(imgs, x, y, wrap_around=False):
    """
    Translate image in 2D plane
    imgs - (B,dimx,dimy), no cuda
    x - (B), np array or torch no cuda
    y - (B), np array or torch no cuda
    Output - (B,dimx,dimy), no cuda
    """
    B, dimx, dimy = imgs.size()[0], imgs.size()[1], imgs.size()[2]
    center = ((dimx-1)/2, (dimy-1)/2)
    Ms = get_affine_transform(-x, -y, torch.zeros(B), torch.ones(B), center)
    transformed_imgs = transform(imgs,Ms,wrap_around=wrap_around)
    return transformed_imgs

def transform(images, Ms, wrap_around=False, fill_mode=CONSTANT, fill_color=0.0):
    B, dimx, dimy = images.size()[0], images.size()[1], images.size()[2]
    B_max = 100
    # to prevent using too much GPU memory, resample images at most B_max images at a time
    transformed_images = torch.zeros(B,dimx,dimy)
    for i in range(B//B_max):
        s, t = i*B_max,(i+1)*B_max
        grid_coords = get_grid_coords(B_max, dimx, dimy)
        transformed_grid_coords = transform_coords(Ms[s:t], grid_coords, wrap_around=wrap_around)
        transformed_images[s:t] = resample(images[s:t], transformed_grid_coords, fill_mode=fill_mode, fill_color=fill_color)
    if B%B_max != 0:
        s, t = B_max*(B//B_max), B
        grid_coords = get_grid_coords(t-s, dimx, dimy)
        transformed_grid_coords = transform_coords(Ms[s:t], grid_coords, wrap_around=wrap_around)
        transformed_images[s:t] = resample(images[s:t], transformed_grid_coords, fill_mode=fill_mode, fill_color=fill_color)
    return transformed_images

def get_grid_coords(B, dimx, dimy):
    sx, sy = torch.arange(dimx).float().cuda(), torch.arange(dimy).float().cuda() # (dimx), (dimy)
    grid_coords = torch.stack(torch.meshgrid(sx, sy),dim=2) # (dimx, dimy, 2)
    repeated_grid_coords = grid_coords.repeat(B,1,1,1) # (B,dimx,dimy,2)
    return repeated_grid_coords

def transform_coords(Ms, coords, wrap_around=False):
    """
    Ms - affine transform matrices, (B,2,3), assumes x-y coordinate system in math
    coords - (B,dimx,dimy,2)
    """
    B, dimx, dimy = coords.size()[0], coords.size()[1], coords.size()[2]
    augmented_coords = torch.cat((coords, torch.ones(B,dimx,dimy,1).cuda()),dim=3) # (B,dimx,dimy,3)
    new_coords = torch.einsum('bij,bxyj->bxyi',Ms,augmented_coords) # (B,dimx,dimy,2)
    if wrap_around:
        new_coords[:,:,:,0] = new_coords[:,:,:,0] % dimx
        new_coords[:,:,:,1] = new_coords[:,:,:,1] % dimy
    return new_coords

def resample(images, coords, fill_mode=CONSTANT, fill_color=0.0):
    """
    images - (B,dimx,dimy)
    coords - (B,dimx,dimy,2)
    """
    B, dimx, dimy = images.size()[0], images.size()[1], images.size()[2]
    
    # Rotate image from python array coordinate system to typical x-y coordinate system used in math
    images = images.cuda()
    images = torch.rot90(images, -1, [1, 2])
    
    # Compute the four nearest neighboring pixels for each coordinate
    x, y = coords[:,:,:,0].cuda(), coords[:,:,:,1].cuda() # (B,dimx, dimy), (B,dimx, dimy)
    p00 = (torch.floor(x).long(), torch.floor(y).long()) # (2,B,dimx,dimy)
    p01 = (p00[0], p00[1]+1) # (2,B,dimx,dimy)
    p10 = (p00[0]+1, p00[1]) # (2,B,dimx,dimy)
    p11 = (p00[0]+1, p00[1]+1) # (2,B,dimx,dimy)
    ps = (p00, p01, p10, p11) # (4,2,B,dimx,dimy)
    
    # Compute the coefficients for bilinear interpolation
    x0, x1 = p00[0].float(), p11[0].float()
    y0, y1 = p00[1].float(), p11[1].float()
    
    w00 = (x1-x)*(y1-y) # (B,dimx,dimy)
    w01 = (x1-x)*(y-y0) # (B,dimx,dimy)
    w10 = (x-x0)*(y1-y) # (B,dimx,dimy)
    w11 = (x-x0)*(y-y0) # (B,dimx,dimy)
    bs = torch.stack((w00,w01,w10,w11),dim=3)
    
    # Pad original image according to fill_mode
    D = dimx*dimy
    llx = torch.min(torch.min(x0), torch.zeros(1).cuda()).long() # (1)
    lly = torch.min(torch.min(y0), torch.zeros(1).cuda()).long() # (1)
    urx = torch.max(torch.max(x1), (dimx-1)*torch.ones(1).cuda()).long() # (1)
    ury = torch.max(torch.max(y1), (dimx-1)*torch.ones(1).cuda()).long() # (1)
    padded_dimx = urx - llx + 1 # (1)
    padded_dimy = ury - lly + 1 # (1)
    padded_images = torch.zeros(B,padded_dimx,padded_dimy).cuda()
    offset_x, offset_y = -llx, -lly # (1), (1)
    if fill_mode == CONSTANT:
        padded_images = padded_images + fill_color
        padded_images[:,offset_x:offset_x+dimx,offset_y:offset_y+dimy] = images
#     if fill_mode == REPLICATE:
#         padded_image[offset_x:offset_x+dimx][offset_y:offset_y+dimy] = image
    
    # Compute each of the four neighboring pixels values for each coordinate in the padded original image
    fs = torch.zeros(B, dimx, dimy, 4).cuda()
    idx = torch.arange(B).repeat(dimx,dimy,1).permute(2,0,1).cuda()
    for i in range(4):
        fs[:,:,:,i] = padded_images[idx,ps[i][0]+offset_x,ps[i][1]+offset_y]
    
    # Compute new image by dot product of the calculated bilinear interpolation coefficients and the pixel values
    new_images = torch.einsum('bxyi,bxyi->bxy',bs,fs)
    
    # Rotate image from typical x-y coordinate system used in math back to python array coordinate system
    new_images = torch.rot90(new_images, 1, [1, 2]).cpu()
    
    return new_images
