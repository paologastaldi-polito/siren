import scipy
import numpy as np
import math
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, ToPILImage, GaussianBlur
import matplotlib.pyplot as plt

def _psnr(pred, gt, max_pixel=1.):
    '''PSNR formula'''
    mse = ((pred - gt)**2).mean().item()
    rmse = math.sqrt(mse)
    psnr = -999.
    if mse != 0.:
        psnr = 20 * math.log10(max_pixel/rmse)
    return psnr

# --- IMAGE ---

def monocromatic_image(color=255, sidelength=256):
    img = np.zeros([sidelength,sidelength,1],dtype=np.uint8)
    img[:] = color
    transform = Compose([
        ToPILImage(),
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def blur_image(img):
    return GaussianBlur(99., (0.1, 99.))(img)

def _init_image_psnr(in_img):
    in_img = in_img.detach().cpu().numpy() # tensors do not have to be attached to the graph and running on the GPU anymore
    out_img = in_img.transpose(1, 2, 0)
    out_img = (out_img / 2.) + 0.5 # move [-1, 1] in [0, 1]
    out_img = np.clip(out_img, a_min=0., a_max=1.)
    return out_img

def image_psnr(img, gt):
    '''Compute PSNR over image'''
    if type(img) is dict:
        img = img['pixels']
    if type(gt) is dict:
        gt = gt['pixels']
    img = _init_image_psnr(img)
    gt = _init_image_psnr(gt)
    return _psnr(img, gt)

def plot_image(img, gt, sidelength=256):
    _, axes = plt.subplots(1,2, figsize=(18,6))
    axes[0].imshow(img.cpu().view(sidelength,sidelength).detach().numpy())
    axes[1].imshow(gt.cpu().view(sidelength,sidelength).detach().numpy())
    plt.show()

# --- GRADIENTS ---

def sobel_filter(img):
    '''Generate gradients with the sobel operator'''
    grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
    grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
    grads = np.sqrt(np.power(grads_x, 2) + np.power(grads_y, 2))    
    return { 'grads' : torch.from_numpy(grads), 'grads_x' : torch.from_numpy(grads_x), 'grads_y' : torch.from_numpy(grads_y) }

def _init_grads_psnr(in_grads):
    # why 2. and 4.?
    # input [-1., +1.]
    # gradient matrix [-2., +2.]
    out_grads = (in_grads +2.) / 4. # move [-2., 2.] in [0., 1.]
    out_grads = np.clip(out_grads, a_min=0., a_max=1.)
    return out_grads

def grads_psnr(img_grads, gt_grads):
    '''Compute PSNR over gradients'''
    if type(img_grads) is dict:
        img_grads = img_grads['grads']
    if type(gt_grads) is dict:
        gt_grads = gt_grads['grads']
    img_grads = _init_grads_psnr(img_grads)
    gt_grads = _init_grads_psnr(gt_grads)
    return _psnr(img_grads, gt_grads)

def plot_grads(img_grads, gt_grads, sidelength=256):
    _, axes = plt.subplots(1,2, figsize=(18,6))
    axes[0].imshow(img_grads.cpu().norm(dim=-1).view(sidelength,sidelength).detach().numpy())
    axes[1].imshow(gt_grads.cpu().norm(dim=-1).view(sidelength,sidelength).detach().numpy())
    plt.show()

# --- LAPLACIAN ---

def laplace_filter(img):
    '''Get laplacian with ND laplacian operator'''
    img_laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
    img_laplace = torch.from_numpy(img_laplace).view(-1, 1)
    return { 'laplace' : img_laplace }

def _init_grads_psnr(in_laplace):
    # why 1. and 5.?
    # input: [-1., +1.]
    # laplacian matrix [-1., +4]
    out_laplace = (in_laplace + 1.) / 5. # move [-1., +4.] in [0., 1.]
    out_laplace = np.clip(out_laplace, a_min=0., a_max=1.)
    return out_laplace

def laplace_psnr(img_laplace, gt_laplace):
    '''Compute PSNR over laplacian'''
    if type(img_laplace) is dict:
        img_laplace = img_laplace['laplace']
    if type(gt_laplace) is dict:
        gt_laplace = gt_laplace['laplace']
    img_laplace = _init_grads_psnr(img_laplace)
    gt_laplace = _init_grads_psnr(gt_laplace)
    return _psnr(img_laplace, gt_laplace)

def plot_laplace(img_laplace, gt_laplace, sidelength=256):
    _, axes = plt.subplots(1,2, figsize=(18,6))
    axes[0].imshow(img_laplace.cpu().view(sidelength,sidelength).detach().numpy())
    axes[1].imshow(gt_laplace.cpu().view(sidelength,sidelength).detach().numpy())
    plt.show()